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
        self.opt = opt
        self.word_dict = torch.load(self.opt.input_file_train + '.word_dict')
        self.inverted_word_dict = {i: w for w, i in self.word_dict.items()}
        self.word_count = torch.load(self.opt.input_file_train + '.word_count')
        self.N_WORDS = len(self.word_dict)

        self.encoder = nn.Embedding(self.N_WORDS, self.opt.hidden_size_rnn)
        self.rnn = nn.GRU(self.opt.hidden_size_rnn, self.opt.hidden_size_rnn, self.opt.n_layers_rnn)
        self.decoder = nn.Linear(self.opt.hidden_size_rnn, self.N_WORDS)
        self.encoder_topic = nn.Embedding(self.N_WORDS, self.opt.hidden_size_rnn)

        self.criterion = nn.CrossEntropyLoss()

        self.submodules = [self.encoder, self.rnn, self.decoder, self.criterion]

        self.losses_reconstruction = []
        self.losses_topic = []

    def analyze(self, batch):

        examples = []
        # Is noun, adjective, verb or adverb?
        is_nava = lambda word: len(wn.synsets(word)) != 0

        # Select "topic" as the least common noun, verb, adjective or adverb in each sentence
        for i in range(len(batch[0])):
            sentence = [batch[j][i] for j in range(len(batch))]
            words_sorted = sorted([(self.word_count[word], word) for word in set(sentence) if is_nava(word)])
            least_common_word = words_sorted[0][1] if len(words_sorted) > 0 else sentence[0]
            examples.append({'sentence': sentence, 'topic candidates': words_sorted})

        print(examples)

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

    def evaluate(self, batch):

        topics = self.select_topics(batch)
        inp, target = self.get_input_and_target(batch)
        # Topic is provided as an initialization to the hidden state
        hidden = torch.cat([self.encoder(topics) for _ in range(self.opt.n_layers_rnn)], 1)\
                      .permute(1, 0, 2)  # N_layers x batch_size x N_hidden

        # Encode/Decode sentence
        loss = 0
        for w in range(self.opt.sentence_len):
            output, hidden = self.forward(inp[:, w], hidden)
            loss += self.criterion(output, target[:, w])  # From documentation: The losses are averaged across observations for each minibatch.

        return loss

    def perplexity(self, batch):
        loss = eval(batch)
        return torch.exp(loss.data[0])

    def RNN_output_to_word(self, one_hot, temperature=0.8):
        word_dist = one_hot.div(temperature).exp()
        _, top_indexes = word_dist.max(1)
        # top_indexes = torch.multinomial(word_dist, 1)
        return top_indexes

    def test(self, prime_words, predict_len, temperature=0.8):

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