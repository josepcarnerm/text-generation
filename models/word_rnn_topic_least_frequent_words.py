import string

import nltk
nltk.download('wordnet')

import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import is_remote, zeros
from nltk.corpus import wordnet as wn
from models.word_rnn_topic_provided import Model as WordRNNModelTopic


class Model(WordRNNModelTopic):

    def __init__(self, opt):
        super(Model, self).__init__(opt)

    def analyze_topics(self, batch):
        batch_size, sentence_len = len(batch[0]), len(batch) - 1

        # Analyze select_topics
        examples = []
        # Is noun, adjective, verb or adverb?
        is_nava = lambda word: len(wn.synsets(word)) != 0

        # Select "topic" as the least common noun, verb, adjective or adverb in each sentence
        print('Analyzing topic candidates......')
        for i in range(len(batch[0])):
            sentence = [batch[j][i] for j in range(len(batch))]
            words_sorted = sorted([(self.word_count[word], word) for word in set(sentence) if is_nava(word)])
            examples.append({'sentence': sentence, 'topic candidates': words_sorted})

        for e in examples:
            try:
                print('Sentence: {}. Topic candidates: {}.'.format(' '.join(e['sentence']), e['topic candidates']))
            except:
                print('Exception when printing')

    def get_test_topic(self):
        return self.select_topics([['happy']])

    def analyze_closeness_to_topics(self, batch):
        batch_size, sentence_len = len(batch[0]), len(batch) - 1

        # Analyze closeness to topics
        print('Analyzing predictions......')
        examples = []

        self.copy_weights_encoder()
        topics, topic_words = self.select_topics(batch)
        if len(batch) ==2:
            batch = batch[1]
        inp, target = self.get_input_and_target(batch)
        if self.opt.bidirectional:
            topic_enc = self.encoder(topics.view(-1, 1)).view(self.opt.n_layers_rnn*2, batch_size, self.opt.hidden_size_rnn)
        else:
            topic_enc = self.encoder(topics.view(-1, 1)).view(self.opt.n_layers_rnn, batch_size, self.opt.hidden_size_rnn)
        hidden = topic_enc, topic_enc.clone()

        for i in range(len(batch[0])):
            sentence = [batch[j][i] for j in range(len(batch))]
            examples.append(
                {'sentence': ' '.join(sentence), 'topic': topic_words[i], 'preds and dist': []})

        # Encode/Decode sentence
        for w in range(sentence_len):
            output, hidden = self.forward(inp[:, w], hidden)

            # Topic closeness loss: Weight each word contribution by the inverse of it's frequency
            # _, words_i = output.max(1)
            # Sample from the network as a multinomial distribution
            output_dist = output.div(0.8).exp()
            words_i = torch.multinomial(output_dist, 1)

            loss_topic_weights = Variable(torch.from_numpy(numpy.array(
                [1 / self.word_count[self.inverted_word_dict[i.data[0]]] for i in words_i]
            )).unsqueeze(1)).float()
            loss_topic_weights = loss_topic_weights.cuda() if is_remote() else loss_topic_weights
            closeness = self.closeness_to_topics(output, topics)

            for i in range(len(batch[0])):
                examples[i]['preds and dist'].append({
                    'predicted word': self.inverted_word_dict[words_i[i].data[0]],
                    'word weight in topic loss': loss_topic_weights[i].data[0],
                    'closeness to sentence topic': closeness[i].data[0]
                })
        for e in examples:
            try:
                print('Sentence: {}. Topic: {}. Predictions, weights and closeness: {}.'.format(
                    ' '.join(e['sentence']), e['topic'], '\n\t' + '\n\t'.join([str(x) for x in e['preds and dist']])
                ))
            except:
                print('Exception when printing')

    def select_topics(self, batch):

        # batch is weirdly ordered due to Pytorch Dataset class from which we inherit in each dataloader : sizes are sentence_len x batch_size

        # Is noun, adjective, verb or adverb?
        is_nava = lambda word: len(wn.synsets(word)) != 0
        batch_size = len(batch[0])

        # Select "topic" as the least common noun, verb, adjective or adverb in each sentence
        topics = torch.LongTensor(self.opt.n_layers_rnn, batch_size, 1)
        topics_words = []
        for i in range(batch_size):
            sentence = [batch[j][i] for j in range(len(batch))]
            words_sorted = sorted([(self.word_count[word], word) for word in set(sentence) if is_nava(word) and word in self.word2idx])
            if len(words_sorted) < self.opt.n_layers_rnn:
                n_more = self.opt.n_layers_rnn - len(words_sorted)
                for i in range(n_more):
                    words_sorted.append(words_sorted[0])
            for j in range(self.opt.n_layers_rnn):
                topics[j,i] = self.from_string_to_tensor([words_sorted[j][1]])
            topics_words.append(tuple([w[1] for w in words_sorted[:self.opt.n_layers_rnn]]))

        if self.opt.bidirectional:
            topics = torch.cat([topics, topics], 2)
        topics = Variable(topics).cuda() if is_remote() else Variable(topics)
        return topics, topics_words

    def closeness_to_topics(self, words_weights, topics):

        _, words = words_weights.max(1)

        dist = nn.PairwiseDistance(p=2)

        closeness = []

        for i in range(topics.size(1)):
            closeness_batch = []
            for j in range(topics.size(0)):
                topic_str = self.inverted_word_dict[topics[j,i].data[0]]
                topic = topics[j,i]
                word = words[i]
                synonyms = [topic.data[0]]
                for syn in wn.synsets(topic_str):
                    synonyms += [self.word2idx[l.name()] for l in syn.lemmas() if l.name() in self.word2idx]
                synonyms = torch.from_numpy(numpy.array(synonyms))
                synonyms = Variable(synonyms).cuda() if is_remote() else Variable(synonyms)
                closeness_batch.append(
                    torch.mean(torch.stack([dist(self.encoder(syn), self.encoder(word)) for syn in synonyms]))
                )
            closeness_batch = torch.cat(closeness_batch)
            closeness.append(closeness_batch.mean())

        return torch.stack(closeness)

    def evaluate(self, batch):

        loss_reconstruction = 0
        loss_topic = 0

        self.copy_weights_encoder()
        topics, topics_words = self.select_topics(batch)
        if self.opt.bidirectional:
            topics_enc = self.encoder(topics.view(-1, 1)) \
                .view(self.opt.n_layers_rnn*2, self.opt.batch_size, self.opt.hidden_size_rnn)
        else:
            topics_enc = self.encoder(topics.view(-1, 1)) \
                             .view(self.opt.n_layers_rnn, self.opt.batch_size, self.opt.hidden_size_rnn)
        inp, target = self.get_input_and_target(batch)

        # Topic is provided as an initialization to the hidden state
        hidden = topics_enc, topics_enc.clone()

        # Encode/Decode sentence
        loss_topic_total_weight = 0
        last_output = inp[:, 0]  # Only used if "reuse_pred" is set
        for w in range(self.opt.sentence_len):

            x = last_output if self.opt.reuse_pred else inp[:, w]
            output, hidden = self.forward(x, hidden)

            # Reconstruction Loss
            loss_reconstruction += self.criterion(output.view(self.opt.batch_size, -1), target[:, w])

            # Topic closeness loss: Weight each word contribution by the inverse of it's frequency
            _, words_i = output.max(1)
            loss_topic_weights = Variable(torch.from_numpy(numpy.array(
                [1/self.word_count[self.inverted_word_dict[i.data[0]]] for i in words_i]
            )).unsqueeze(1)).float()
            loss_topic_weights = loss_topic_weights.cuda() if is_remote() else loss_topic_weights
            loss_topic_total_weight += loss_topic_weights
            loss_topic += self.closeness_to_topics(output, topics) * loss_topic_weights

        loss_topic = torch.mean(loss_topic/loss_topic_total_weight)

        self.losses_reconstruction.append(loss_reconstruction.data[0])
        self.losses_topic.append(loss_topic.data[0])

        ratio = float(loss_reconstruction.detach().cpu().data.numpy()[0] / loss_topic.detach().cpu().data.numpy()[0])
        return self.opt.loss_alpha * loss_reconstruction + (1 - self.opt.loss_alpha) * loss_topic * ratio

    def test(self, prime_words, predict_len, temperature=0.8):

        self.copy_weights_encoder()
        topic, _ = self.get_test_topic()
        if self.opt.bidirectional:
            topic_enc = self.encoder(topic.view(-1, 1)).view(self.opt.n_layers_rnn*2, 1, self.opt.hidden_size_rnn)
        else:
            topic_enc = self.encoder(topic.view(-1, 1)).view(self.opt.n_layers_rnn, 1, self.opt.hidden_size_rnn)
        hidden = topic_enc, topic_enc.clone()
        prime_input = Variable(self.from_string_to_tensor(prime_words).unsqueeze(0))

        if is_remote():
            prime_input = prime_input.cuda()
        predicted = ' '.join(prime_words)

        # Use priming string to "build up" hidden state
        for p in range(len(prime_words) - 1):
            _, hidden = self.forward(prime_input[:, p], hidden)

        inp = prime_input[:, -1]

        for p in range(predict_len):
            output, hidden = self.forward(inp, hidden)

            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            # Add predicted character to string and use as next input
            predicted_word = self.from_predicted_index_to_string(top_i)
            predicted += ' ' + predicted_word
            inp = Variable(self.from_string_to_tensor([predicted_word]).unsqueeze(0))
            if is_remote():
                inp = inp.cuda()

        return predicted