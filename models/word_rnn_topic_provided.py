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

    def evaluate(self, batch):

        topics, sentences = batch
        inp, target = self.get_input_and_target(sentences)
        # Topic is provided as an initialization to the hidden state
        hidden = torch.cat([self.encoder(topics) for _ in range(self.opt.n_layers_rnn)], 1)\
                      .permute(1, 0, 2)  # N_layers x batch_size x N_hidden

        # Encode/Decode sentence
        loss = 0
        for w in range(self.opt.sentence_len):
            output, hidden = self.forward(inp[:, w], hidden)
            loss += self.criterion(output, target[:, w])  # From documentation: The losses are averaged across observations for each minibatch.

        return loss

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