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

    def init_hidden(self, batch_size):
        return zeros(gpu=is_remote(), sizes=[self.opt.n_layers_rnn, batch_size, self.opt.hidden_size_rnn])

    def forward(self, sentences, num_chars_encoder=0):

        input = torch.stack([to_variable(gpu=is_remote(), sentence=sentence) for sentence in sentences])
        h = self.init_hidden(self.opt.batch_size)


        input_emb = self.encoder(input).permute(1, 0, 2)\
                        .contiguous()\
                        .view(self.opt.sentence_len, self.opt.batch_size, self.opt.hidden_size_rnn)
        output_rnn, h = self.gru(input_emb, h)
        output = self.decoder(output_rnn)\
                     .permute(1,0,2)\
                     .contiguous()\
                     .view(self.opt.batch_size, self.opt.sentence_len, self.opt.n_characters)

        preds = output[:,num_chars_encoder:-1]

        return preds

    def evaluate(self, sentences):
        loss = 0
        preds = self.forward(sentences)
        targets = torch.stack([to_variable(gpu=is_remote(), sentence=sentence) for sentence in sentences])
        targets = targets[:, 1:]
        for i in range(targets.size(1)):  # First pred char is not eval
            loss += self.criterion(preds[:,i], targets[:,i])
        return loss/targets.size(1)

    def test(self, start, predict_len=100, temperature=0.8):

        start = to_variable(gpu=is_remote(), sentence=start)
        h = self.init_hidden(1)

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

