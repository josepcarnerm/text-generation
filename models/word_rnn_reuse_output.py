import string

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import is_remote, zeros
from models.word_rnn import Model as WordRNNModel


class Model(WordRNNModel):
    """
    WordRNN but instead of inputing the next word at training time, we input the previous prediction (as in test time),
    so that the model generalizes better
    """

    def __init__(self, opt):
        super(Model, self).__init__(opt)

    @staticmethod
    def select_word_index_from_output(output, temperature=0.8):
        # Output: batch_size x N_words
        _, top_indexes = output.max(1)  # Select word as the with "highest probability"
        # top_indexes.size() = (batch_size)
        return top_indexes

    def evaluate(self, batch):
        
        inp, target = self.get_input_and_target(batch)
        hidden = self.init_hidden(self.opt.batch_size)
        loss = 0
        last_output = inp[:, 0]

        for w in range(self.opt.sentence_len):
            output, hidden = self.forward(last_output, hidden)
            last_output = self.select_word_index_from_output(output)
            loss += self.criterion(output.view(self.opt.batch_size, -1), target[:, w])

        return loss
