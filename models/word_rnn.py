import string, torch
import torch.nn as nn
from torch.autograd import Variable

# Project imports
from utils import move, zeros, to_variable, to_string, is_remote
import pdb



class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.criterion = nn.CrossEntropyLoss()

        self.opt.emb_dim = \
            self.get_embedding_dimensions_from_filename(self.opt.input_pretrained_vector)

        self.encoder = nn.Embedding(self.opt.emb_dim, self.opt.hidden_size_rnn)
        self.lstm = nn.LSTM(input_size=self.opt.emb_dim, hidden_size=self.opt.hidden_size_rnn,\
                            num_layers=self.opt.n_layers_rnn, dropout=self.opt.dropout_rate)
        self.decoder = nn.Linear(self.opt.hidden_size_rnn, self.opt.emb_dim)

        self.submodules = [self.encoder, self.gru, self.decoder, self.criterion]
        move(gpu=is_remote(), tensor_list=self.submodules)

    def init_hidden(self, batch_size):
        return zeros(gpu=is_remote(), sizes=[self.opt.n_layers_rnn, batch_size, self.opt.hidden_size_rnn])

    def forward(self, sentences, num_words_encoder=0):
        input = torch.stack([to_variable(gpu=is_remote(), sentence=sentence) for sentence in sentences])
        h = self.init_hidden(self.opt.batch_size)

        input_emb = self.encoder(input).permute(1,0,2)\
            .contiguous()\
            .view(self.opt.sentence_len, self.opt.batch_size, self.opt.hidden_size_rnn)
        output_rnn, h = self.lstm(input_emb, h)
        output = self.decoder(output_rnn)\
            .permute(1,0,2)\
            .contiguous()\
            .view(self.opt.batch_size, self.opt.sentence_len, self.opt.emb_dim)

        preds = output[:,num_words_encoder:-1]
        return preds


    def evaluate(self, sentences):
        loss = 0
        pdb.set_trace()
        preds = self.forward(sentences)
        targets = torch.stack([to_variable(gpu=is_remote(), sentence=sentence) for sentence in sentences])
        targets = targets[:, 1:]
        for i in range(targets.size(1)):  # First pred word not evaluated
            loss += self.criterion(preds[:,i], targets[:,i])
        return loss/targets.size(1)

    def test(self, start, predict_len=100, temperature=0.8):

        start = to_variable(gpu=is_remote(), sentence=start)
        pdb.set_trace()
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

    # HACK; TODO: better way to get glove vector size
    def get_embedding_dimensions_from_filename(self, fname):
        vec_type = fname.split('/')[1]

        if vec_type == 'glove.6B':
            this_glove_file = fname.split('/')[-1]
            return int(this_glove_file.split('.')[-2][:-1])
        return self.opt.hidden_size_rnn
