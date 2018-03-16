# External modules imports
import string, torch
import torch.nn as nn
from torch.autograd import Variable


# Project imports
from utils import move, zeros, to_variable, to_string, is_remote


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.criterion = nn.CrossEntropyLoss()

        self.opt.n_characters = len(string.printable)

        self.encoder = nn.Embedding(self.opt.n_characters, self.opt.hidden_size_rnn)
        self.gru = nn.GRU(self.opt.hidden_size_rnn, self.opt.hidden_size_rnn, self.opt.n_layers_rnn)
        self.decoder = nn.Linear(self.opt.hidden_size_rnn, self.opt.n_characters)

        self.submodules = [self.encoder, self.gru, self.decoder, self.criterion]
        move(gpu=is_remote(), tensor_list=self.submodules)

    def init_hidden(self):
        return zeros(gpu=is_remote(), sizes=[self.opt.n_layers_rnn, 1, self.opt.hidden_size_rnn])

    def forward(self, sentence, num_chars_encoder=0):

        input = to_variable(gpu=is_remote(), sentence=sentence)
        seq_len = input.size(0)
        h = self.init_hidden()

        input_emb = self.encoder(input) \
            .view(seq_len, 1, self.opt.hidden_size_rnn)  # Batch size x N channels x Width (Seq len) x Height (Emb dim)
        output_rnn, h = self.gru(input_emb, h)
        output = self.decoder(output_rnn.squeeze())  # Seq len x n_chars (one hot vector of output sentence)
        preds = output[num_chars_encoder:-1]

        return preds

    def evaluate(self, sentence):
        loss = 0
        preds = self.forward(sentence)
        target = to_variable(gpu=is_remote(), sentence=sentence)
        for i in range(len(sentence) - 1):  # First pred char is not eval
            loss += self.criterion(preds[i].unsqueeze(0), target[i + 1])
        return loss

    def test(self, start, predict_len=100, temperature=0.8):

        start = to_variable(gpu=is_remote(), sentence=start)
        h = self.init_hidden()

        for c in start:
            c = c.view(1, 1)
            embedded_cs = self.encoder(c)
            output, h = self.gru(embedded_cs.view(1, 1, self.opt.hidden_size_rnn), h)

        preds = []
        output_dist = output.view(-1).div(temperature).exp()
        c = torch.multinomial(output_dist, 1)
        for _ in range(predict_len):
            c = c.view(1, 1)
            embedded_cs = self.encoder(c)
            output, h = self.gru(embedded_cs.view(1, 1, self.opt.hidden_size_rnn), h)
            output_dist = output.view(-1).div(temperature).exp()
            c = torch.multinomial(output_dist, 1)
            preds.append(to_string(c))

        return ''.join(preds)

