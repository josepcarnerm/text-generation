# External modules imports
import string, torch
import torch.nn as nn
from torch.autograd import Variable


# Project imports
from utils import move, zeros, to_variable, to_string, is_remote


class Model(nn.Module):
    # WORK IN PROGRESS

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.criterion = nn.CrossEntropyLoss()

        self.opt.n_characters = len(string.printable)
        self.opt.emb_dimension = self.opt.hidden_size_rnn
        self.opt.window_size = 5

        if self.opt.window_size % 2 == 0:
            raise Exception("Window size must be odd")

        self.encoder_emb = nn.Embedding(self.opt.n_characters, self.opt.emb_dimension)
        self.encoder_cnn = nn.Conv2d(
            in_channels = 1,
            out_channels = self.opt.hidden_size_rnn,
            kernel_size = (self.opt.window_size, self.opt.hidden_size_rnn),
            padding = (self.opt.window_size//2, 0) # That's why window size must be odd, so that padding "matches" and output length is still the same
        )
        self.gru = nn.GRU(self.opt.hidden_size_rnn, self.opt.hidden_size_rnn, self.opt.n_layers_rnn)
        self.decoder = nn.Linear(self.opt.hidden_size_rnn, self.opt.n_characters)

        self.submodules = [self.encoder_emb, self.encoder_cnn, self.gru, self.decoder, self.criterion]
        move(gpu=is_remote(), tensor_list=self.submodules)

    def init_hidden(self):
        return zeros(gpu=is_remote(), sizes=[self.opt.n_layers_rnn, 1, self.opt.hidden_size_rnn])

    def forward(self, sentence, num_chars_encoder=0):

        input = to_variable(gpu=is_remote(), sentence=sentence)
        seq_len = input.size(0)
        h = self.init_hidden()

        input_emb = self.encoder_emb(input)\
                        .view(1,1, seq_len, self.opt.emb_dimension) # Batch size x N channels x Width (Seq len) x Height (Emb dim)
        input_rnn = self.encoder_cnn(input_emb)[:,:,:]\
                        .permute(2,0,1,3).contiguous()\
                        .view(seq_len, 1, self.opt.hidden_size_rnn) # Seq len x Batch size x N hidden (= n_filters)
        output_rnn, h = self.gru(input_rnn, h)
        output = self.decoder(output_rnn.squeeze())  # Seq len x n_chars (one hot vector of output sentence)
        preds = output[num_chars_encoder:-1]
        return preds

    def evaluate(self, sentence):
        loss = 0
        preds = self.forward(sentence)
        target = to_variable(gpu=is_remote(), sentence=sentence)
        for i in range(len(sentence)-1): # First pred char is not eval
            loss += self.criterion(preds[i].unsqueeze(0), target[i+1])
        return loss

    def test(self, start, predict_len=100, temperature=0.8):

        # preds_dist = preds.div(temperature).exp()
        # chars_indexes = torch.multinomial(preds_dist, 1).size()
        # return to_string(chars_indexes)

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
            import pdb; pdb.set_trace()

        return ''.join(preds)

