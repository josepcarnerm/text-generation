import string

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import is_remote, zeros

class Model(nn.Module):

    N_CHARS = len(string.printable)
    ALL_CHARS = string.printable

    def __init__(self, opt):
        super(Model, self).__init__()

        self.opt = opt

        self.encoder = nn.Embedding(self.N_CHARS, self.opt.hidden_size_rnn)
        self.rnn = nn.GRU(self.opt.hidden_size_rnn, self.opt.hidden_size_rnn, self.opt.n_layers_rnn)
        self.decoder = nn.Linear(self.opt.hidden_size_rnn, self.N_CHARS)

        self.criterion = nn.CrossEntropyLoss()

        self.submodules = [self.encoder, self.rnn, self.decoder, self.criterion]

    def from_string_to_tensor(self, string):
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            try:
                tensor[c] = self.ALL_CHARS.index(string[c])
            except:
                continue
        return tensor

    def from_predicted_index_to_string(self, index):
        return self.ALL_CHARS[index]

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def forward2(self, input, hidden):
        encoded = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def evaluate(self, batch):

        inp = torch.LongTensor(self.opt.batch_size, self.opt.sentence_len)
        target = torch.LongTensor(self.opt.batch_size, self.opt.sentence_len)
        for i, sentence in enumerate(batch):
            inp[i] = self.from_string_to_tensor(sentence[:-1])
            target[i] = self.from_string_to_tensor(sentence[1:])
        inp = Variable(inp)
        target = Variable(target)
        if is_remote():
            inp = inp.cuda()
            target = target.cuda()

        hidden = self.init_hidden(self.opt.batch_size)
        loss = 0

        for c in range(self.opt.sentence_len):
            output, hidden = self.forward(inp[:, c], hidden)
            loss += self.criterion(output.view(self.opt.batch_size, -1), target[:, c])

        return loss

    def init_hidden(self, batch_size):
        return zeros(gpu=is_remote(), sizes=(self.opt.n_layers_rnn, batch_size, self.opt.hidden_size_rnn))

    def test(self, prime_str='A', predict_len=100, temperature=0.8):

        hidden = self.init_hidden(1)
        prime_input = Variable(self.from_string_to_char_tensor(prime_str).unsqueeze(0))

        if is_remote():
            prime_input = prime_input.cuda()
        predicted = prime_str

        # Use priming string to "build up" hidden state
        for p in range(len(prime_str) - 1):
            _, hidden = self.forward(prime_input[:, p], hidden)

        inp = prime_input[:, -1]

        for p in range(predict_len):
            output, hidden = self.forward(inp, hidden)

            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            # Add predicted character to string and use as next input
            predicted_char = self.from_predicted_index_to_string(top_i)
            predicted += predicted_char
            inp = Variable(self.from_string_to_tensor(predicted_char).unsqueeze(0))
            if is_remote():
                inp = inp.cuda()

        return predicted

