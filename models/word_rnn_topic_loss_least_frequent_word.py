import string

import nltk
nltk.download('wordnet')

import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import is_remote, zeros
from nltk.corpus import wordnet as wn
from models.word_rnn import Model as WordRNNModel


class Model(WordRNNModel):

    def __init__(self, opt):
        super(Model, self).__init__(opt)

        self.encoder_topic = nn.Embedding(self.N_WORDS, self.opt.hidden_size_rnn)
        for p in self.encoder_topic.parameters(): p.requires_grad = False  # Freeze weights

        self.submodules = self.submodules + [self.encoder_topic]

        self.losses_reconstruction = []
        self.losses_topic = []

    def closeness_to_topics(self, words_weights, topics):

        _, words = words_weights.max(1)

        dist = nn.PairwiseDistance(p=2)

        closeness = []

        for i in range(len(topics)):
            topic_str = self.inverted_word_dict[topics[i].data[0]]
            topic = topics[i]
            word = words[i]
            synonyms = [topic.data[0]]
            for syn in wn.synsets(topic_str):
                synonyms += [self.word_dict[l.name()] for l in syn.lemmas() if l.name() in self.word_dict]

            synonyms = torch.from_numpy(numpy.array(synonyms))
            synonyms = Variable(synonyms).cuda() if is_remote() else Variable(synonyms)

            closeness.append(torch.mean(torch.stack([dist(self.encoder(syn), self.encoder(word)) for syn in synonyms])))

        return torch.stack(closeness)

    def analyze(self, batch):

        # Analyze select_topics
        examples = []
        # Is noun, adjective, verb or adverb?
        is_nava = lambda word: len(wn.synsets(word)) != 0

        # Select "topic" as the least common noun, verb, adjective or adverb in each sentence
        for i in range(len(batch[0])):
            sentence = [batch[j][i] for j in range(len(batch))]
            words_sorted = sorted([(self.word_count[word], word) for word in set(sentence) if is_nava(word)])
            least_common_word = words_sorted[0][1] if len(words_sorted) > 0 else sentence[0]
            examples.append({'sentence': sentence, 'topic candidates': words_sorted})

        for e in examples:
            print('Sentence: {}. Topic candidates: {}.'.format(e['sentence'], e['topic_candidates']))

        # Analyze closeness to topics
        examples = []

        self.copy_weights_encoder()
        topics = self.select_topics(batch)
        inp, target = self.get_input_and_target(batch)
        # Topic is provided as an initialization to the hidden state
        hidden = torch.cat([self.encoder(topics) for _ in range(self.opt.n_layers_rnn)], 1).permute(1, 0, 2)  # N_layers x batch_size x N_hidden
        for i in range(len(batch[0])):
            sentence = [batch[j][i] for j in range(len(batch))]
            examples.append({'sentence': sentence, 'topic': self.inverted_word_dict[topics[i].data[0]], 'preds and dist':[]})

        # Encode/Decode sentence
        for w in range(self.opt.sentence_len):
            output, hidden = self.forward(inp[:, w], hidden)

            # Topic closeness loss: Weight each word contribution by the inverse of it's frequency
            _, words_i = output.max(1)
            loss_topic_weights = Variable(torch.from_numpy(numpy.array(
                [1 / self.word_count[self.inverted_word_dict[i.data[0]]] for i in words_i]
            )).unsqueeze(1)).float()
            loss_topic_weights = loss_topic_weights.cuda() if is_remote() else loss_topic_weights
            closeness = self.closeness_to_topics(output, topics)

            for i in range(len(batch[0])):
                examples[i]['preds and dist'].append([{
                    'pred': self.inverted_word_dict[words_i[i].data[0]], 'w': loss_topic_weights[i], 'closeness': closeness
                }])
        for e in examples:
            print('Sentence: {}. Topic: {}. Predictions, weights and closeness: {}.'.format(
                e['sentence'], e['topic'], e['preds and dist']
            ))

    def select_topics(self, batch):

        # batch is weirdly ordered due to Pytorch Dataset class from which we inherit in each dataloader : sizes are sentence_len x batch_size

        # Is noun, adjective, verb or adverb?
        is_nava = lambda word: len(wn.synsets(word)) != 0
        batch_size = len(batch[0])

        # Select "topic" as the least common noun, verb, adjective or adverb in each sentence
        topics = torch.LongTensor(batch_size, 1)
        for i in range(batch_size):
            sentence = [batch[j][i] for j in range(len(batch))]
            words_sorted = sorted([(self.word_count[word], word) for word in set(sentence) if is_nava(word)])
            least_common_word = words_sorted[0][1] if len(words_sorted) > 0 else sentence[0]
            topics[i] = self.from_string_to_tensor([least_common_word])

        return Variable(topics).cuda() if is_remote() else Variable(topics)

    def copy_weights_encoder(self):
        # Copy weights in encoder to to encoder_topic so that both embedding layers are the same but the latter
        # is not influenced by the topic loss
        self.encoder_topic.load_state_dict(self.encoder.state_dict())

    def evaluate(self, batch):

        loss_reconstruction = 0
        loss_topic = 0

        self.copy_weights_encoder()
        topics = self.select_topics(batch)
        inp, target = self.get_input_and_target(batch)
        # Topic is provided as an initialization to the hidden state
        hidden = torch.cat([self.encoder(topics) for _ in range(self.opt.n_layers_rnn)], 1)\
                      .permute(1, 0, 2)  # N_layers x batch_size x N_hidden

        # Encode/Decode sentence
        loss_topic_total_weight = 0
        for w in range(self.opt.sentence_len):

            output, hidden = self.forward(inp[:, w], hidden)

            # Reconstruction Loss
            loss_reconstruction += self.criterion(output, target[:, w])

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

        return self.opt.loss_alpha*loss_reconstruction + (1-self.opt.loss_alpha)*loss_topic

    def test(self, prime_words, predict_len, temperature=0.8):

        self.copy_weights_encoder()
        topic = self.select_topics([['happy']])
        hidden = torch.cat([self.encoder(topic) for _ in range(self.opt.n_layers_rnn)], 1) \
                      .permute(1, 0, 2)  # N_layers x 1 x N_hidden
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
            predicted += ' '+predicted_word
            inp = Variable(self.from_string_to_tensor([predicted_word]).unsqueeze(0))
            if is_remote():
                inp = inp.cuda()

        return predicted

    # Debug
    def get_avg_losses(self):
        return numpy.mean(self.losses_reconstruction), numpy.mean(self.losses_topic)

    def parameters(self):
        params = (p for p in super(Model, self).parameters() if p.requires_grad)
        return params