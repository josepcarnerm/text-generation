import string, pdb

import nltk
nltk.download('wordnet')

import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import is_remote, zeros, flatten
from nltk.corpus import wordnet as wn
from models.word_rnn_withncharsgram_FAKE import Model as WordRNNModel


class Model(WordRNNModel):

    def __init__(self, opt):
        super(Model, self).__init__(opt)

        self.load_topic_word_counts()
        self.encoder_topic = nn.Embedding(self.N_WORDS, self.opt.hidden_size_rnn)
        if self.opt.use_pretrained_embeddings:
            embeddings = torch.zeros((len(self.word2idx)), self.word_dict_dim).float()
            for k, v in self.word2idx.items():
                embeddings[v] = self.word_dict[k]
            self.encoder_topic.weight = nn.Parameter(embeddings)
            self.encoder_topic.weight.requires_grad = False
            del self.word_dict  # Clear the memory
        for p in self.encoder_topic.parameters(): p.requires_grad = False  # Freeze weights

        self.submodules = self.submodules + [self.encoder_topic]

        self.losses_reconstruction = []
        self.losses_topic = []

    def load_topic_word_counts(self):
        if self.opt.use_pretrained_embeddings:
            self.topic_word_count = torch.load(self.opt.topic_folder_path + '.topic.sentences.g_word_count')

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
                synonyms += [self.word2idx[l.name()] for l in syn.lemmas() if l.name() in self.word2idx]

            synonyms = torch.from_numpy(numpy.array(synonyms))
            synonyms = Variable(synonyms).cuda() if is_remote() else Variable(synonyms)

            closeness.append(torch.mean(torch.stack([dist(self.encoder(syn), self.encoder(word)) for syn in synonyms])))

        return torch.stack(closeness)

    def analyze_topics(self, batch):
        pass

    def analyze_closeness_to_topics(self, batch):

        batch_size, sentence_len = len(batch[0]), len(batch) - 1

        # Analyze closeness to topics
        print('Analyzing predictions......')
        examples = []

        self.copy_weights_encoder()
        topics, _ = self.select_topics(batch)
        if len(batch) ==2:
            batch = batch[1]
        inp, target = self.get_input_and_target(batch)
        # Topic is provided as an initialization to the hidden state
        hidden = torch.cat([self.encoder(topics) for _ in range(self.opt.n_layers_rnn)], 1).permute(1, 0, 2), \
                 torch.cat([self.encoder(topics) for _ in range(self.opt.n_layers_rnn)], 1).permute(1, 0, 2)  # N_layers x batch_size x N_hidden

        for i in range(len(batch[0])):
            sentence = [batch[j][i] for j in range(len(batch))]
            examples.append(
                {'sentence': ' '.join(sentence), 'topic': self.inverted_word_dict[topics[i].data[0]], 'preds and dist': []})

        # Encode/Decode sentence
        for w in range(sentence_len):
            output, hidden = self.forward(inp[:, w], hidden)

            # Topic closeness loss: Weight each word contribution by the inverse of it's frequency
            _, words_i = output.max(1)
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
            print('Sentence: {}. Topic: {}. Predictions, weights and closeness: {}.'.format(
                ' '.join(e['sentence']), e['topic'], '\n\t' + '\n\t'.join([str(x) for x in e['preds and dist']])
            ))

    def analyze(self, batch):

        # self.analyze_topics(batch)
        # self.analyze_closeness_to_topics(batch)

        # Analyze losses
        print('Analyzing Losses......')
        # print('Average topic loss: {}. Average reconstruction loss: {}'.format(
        #     numpy.mean(self.losses_topic), numpy.mean(self.losses_reconstruction)
        # ))

    # def select_topics(self, batch):
    #     try:
    #         batch_size = max(batch[1][0], key=len)
    #     except:
    #         import pdb; pdb.set_trace()


    #     topics_words = batch[1]
    #     import pdb; pdb.set_trace()
    #     topics = self.from_string_to_tensor(topics_words) #.view(batch_size, 1)
    #     return Variable(topics).cuda() if is_remote() else Variable(topics) #, topics_words
    
    def select_topics(self, batch):

        # batch is weirdly ordered due to Pytorch Dataset class from which we inherit in each dataloader : sizes are sentence_len x batch_size

        # Is noun, adjective, verb or adverb?
        is_nava = lambda word: len(wn.synsets(word)) != 0
        batch_size = len(batch[0])

        # Select "topic" as the least common noun, verb, adjective or adverb in each sentence
        topics = torch.LongTensor(batch_size, 1)
        topics_words = []
        # import pdb; pdb.set_trace()
        for i in range(batch_size):
            sentence = flatten([batch[j][i] for j in range(len(batch))])
            try:
                words_sorted = sorted([(self.topic_word_count[word], word) for word in set(sentence) if is_nava(word) and word in self.word2idx])
            except:
                import pdb; pdb.set_trace()
            least_common_word = words_sorted[0][1] if len(words_sorted) > 0 else sentence[0]
            topics[i] = self.from_string_to_tensor([least_common_word])
            topics_words.append(least_common_word)
        return Variable(topics).cuda() if is_remote() else Variable(topics), topics_words

    def copy_weights_encoder(self):
        # Copy weights in encoder to to encoder_topic so that both embedding layers are the same but the latter
        # is not influenced by the topic loss
        if not self.opt.use_pretrained_embeddings:
            self.encoder_topic.load_state_dict(self.encoder.state_dict())

    # def evaluate(self, batch):

    #     loss_reconstruction = 0
    #     loss_topic = 0

    #     self.copy_weights_encoder()
    #     topic = self.select_topics(batch)
    #     # topics, topics_words = self.select_topics(batch)
    #     inp, target = self.get_input_and_target(batch)
    #     inp = inp + topic

    #     # Topic is provided as an initialization to the hidden state
    #     # topic_enc = torch.cat([self.encoder(topics) for _ in range(self.opt.n_layers_rnn)], 1) \
    #     #                  .permute(1, 0, 2)  # N_layers x batch_size x N_hidden
    #     # hidden = topic_enc, topic_enc.clone()



    #     # Encode/Decode sentence
    #     loss_topic_total_weight = 0
    #     last_output = inp[:, 0]  # Only used if "reuse_pred" is set
    #     for w in range(self.opt.sentence_len):

    #         x = last_output if self.opt.reuse_pred else inp[:, w]
    #         output, hidden = self.forward(x, hidden)

    #         # Reconstruction Loss
    #         loss_reconstruction += self.criterion(output, target[:, w])

    #         # Topic closeness loss: Weight each word contribution by the inverse of it's frequency
    #         # _, words_i = output.max(1)
    #         # loss_topic_weights = Variable(torch.from_numpy(numpy.array(
    #         #     [1/self.word_count[self.inverted_word_dict[i.data[0]]] for i in words_i]
    #         # )).unsqueeze(1)).float()
    #         # loss_topic_weights = loss_topic_weights.cuda() if is_remote() else loss_topic_weights
    #         # loss_topic_total_weight += loss_topic_weights
    #         # loss_topic += self.closeness_to_topics(output, topics) * loss_topic_weights

    #     # loss_topic = torch.mean(loss_topic/loss_topic_total_weight)

    #     self.losses_reconstruction.append(loss_reconstruction.data[0])
    #     # self.losses_topic.append(loss_topic.data[0])

    #     return self.opt.loss_alpha*loss_reconstruction #+ (1-self.opt.loss_alpha)*loss_topic

    def evaluate(self, batch):
        
        topic, _ = self.select_topics(batch[0])
        inp, target = self.get_input_and_target(batch)
        hidden = self.init_hidden(self.opt.batch_size)
        loss = 0
        last_output = inp[:, 0]  # Only used if "reuse_pred" is set
        count = 1
        # import pdb; pdb.set_trace()
        for w in range(self.opt.sentence_len):
            x = last_output if self.opt.reuse_pred else inp[:, w]
            output, hidden = self.forward(x, hidden, topic)
            last_output = self.select_word_index_from_output(output)
            if (count == 21):
                pdb.set_trace()
            else:
                count += 1
            loss += self.criterion(output.view(self.opt.batch_size, -1), target[:, w,0])

        return loss

    def forward(self, input, hidden, topic):
        batch_size = input.size(0)  # Will be self.opt.batch_size at train time, 1 at test time
        # test
        if batch_size == 2:
            encoded = self.word_encoder(input[:1])
            encoded_chars = self.char_encoder(input[1:2])
            encoded_topic = self.encoder_topic(topic[:, 0])
            encoded = torch.cat([encoded, encoded_chars, encoded_topic], 1)
            output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
            output = self.decoder(output.view(1, -1))
        # train
        else:
            encoded = self.word_encoder(input[:, 0])
            encoded_chars = self.char_encoder(input[:, 1])
            encoded_topic = self.encoder_topic(topic[:, 0])
            encoded = torch.cat([encoded, encoded_chars, encoded_topic], 1)
            output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
            output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def get_test_topic(self):
        return self.select_topics((['happy'], [['happy']]))

    # def test(self, prime_words, predict_len, temperature=0.8):

    #     self.copy_weights_encoder()
    #     topic, _ = self.get_test_topic()
    #     topic_enc = torch.cat([self.encoder(topic) for _ in range(self.opt.n_layers_rnn)], 1) \
    #                      .permute(1, 0, 2)  # N_layers x 1 x N_hidden
    #     hidden = topic_enc, topic_enc.clone()
    #     prime_input = Variable(self.from_string_to_tensor(prime_words).unsqueeze(0))

    #     if is_remote():
    #         prime_input = prime_input.cuda()
    #     predicted = ' '.join(prime_words)

    #     # Use priming string to "build up" hidden state
    #     for p in range(len(prime_words) - 1):
    #         _, hidden = self.forward(prime_input[:, p], hidden)

    #     inp = prime_input[:, -1]

    #     for p in range(predict_len):
    #         output, hidden = self.forward(inp, hidden)

    #         # Sample from the network as a multinomial distribution
    #         output_dist = output.data.view(-1).div(temperature).exp()
    #         top_i = torch.multinomial(output_dist, 1)[0]

    #         # Add predicted character to string and use as next input
    #         predicted_word = self.from_predicted_index_to_string(top_i)
    #         predicted += ' '+predicted_word
    #         inp = Variable(self.from_string_to_tensor([predicted_word]).unsqueeze(0))
    #         if is_remote():
    #             inp = inp.cuda()

    #     return predicted

    def test(self, prime_words, predict_len, temperature=0.8):

        hidden = self.init_hidden(1)
        prime_input = self.concat_word_with_ngram(Variable(self.from_string_to_tensor(prime_words).unsqueeze(0)))
        inp = prime_input[-2:]
        topic, _ = self.get_test_topic()

        if is_remote():
            prime_input = prime_input.cuda()
            inp = Variable(inp).cuda()

        predicted = ' '.join(prime_words)

        # Use priming string to "build up" hidden state
        # import pdb; pdb.set_trace()
        for p in range(len(prime_words) - 1):
            _, hidden = self.forward(prime_input[:, p], hidden, topic)

        for p in range(predict_len):
            output, hidden = self.forward(inp, hidden, topic)

            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            # Add predicted character to string and use as next input
            predicted_word = self.from_predicted_index_to_string(top_i)
            predicted += ' '+predicted_word
            # pdb.set_trace()
            inp = self.concat_word_with_ngram(None, predicted_word)
            if is_remote():
                inp = Variable(inp).cuda()

        return predicted


    # Debug
    def get_avg_losses(self):
        return numpy.mean(self.losses_reconstruction), numpy.mean(self.losses_topic)

    def parameters(self):
        params = (p for p in super(Model, self).parameters() if p.requires_grad)
        return params