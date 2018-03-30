import string

import torch, pdb
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from utils import is_remote, zeros

class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()

        self.opt = opt

        # Load word dict from file
        self.word_dict = torch.load(self.opt.input_file_train + '.g_word_dict')
        self.word_dict_dim = self.word_dict['the'].size(0)
        self.word2idx = self.get_word_to_idx()
        self.inverted_word_dict = {idx: word for word, idx in self.word2idx.items()}

        self.N_WORDS = len(self.word2idx)

        self.encoder = nn.Embedding(self.N_WORDS, self.word_dict_dim)
        self.encoder.weight = nn.Parameter(self.vocab_embedding())
        self.encoder.weight.requires_grad = False
        # Clear the memory
        del self.word_dict

        self.rnn = nn.GRU(self.opt.hidden_size_rnn, self.opt.hidden_size_rnn, self.opt.n_layers_rnn)
        self.decoder = nn.Linear(self.opt.hidden_size_rnn, self.N_WORDS)

        self.criterion = nn.CrossEntropyLoss()

        self.submodules = [self.encoder, self.rnn, self.decoder, self.criterion]

    def get_word_to_idx(self):
        return {word: idx for idx, word in enumerate((self.word_dict.keys()))}

    def vocab_embedding(self):
        embeddings = torch.zeros((len(self.word2idx)), self.word_dict_dim).float()
        for k, v in self.word2idx.items():
            embeddings[v] = self.word_dict[k]
        return embeddings

    def from_string_to_tensor(self, sentence):
        tensor = torch.LongTensor(len(sentence))
        for i, word in enumerate(sentence):
            try:
                tensor[i] = self.word2idx[word]
            except:
                tensor[i] = self.word2idx['unk']
        return tensor

    def from_predicted_index_to_string(self, index):
        return self.inverted_word_dict[index]

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

    def get_input_and_target(self, batch):

        inp = torch.LongTensor(self.opt.batch_size, (2 * self.opt.sentence_len) + 1)
        target = torch.LongTensor(self.opt.batch_size, (2 * self.opt.sentence_len) + 1)
        for i, sentence in enumerate(batch):
            inp[:, i] = self.from_string_to_tensor(sentence)
            target[:, i] = self.from_string_to_tensor(sentence)
        inp = inp[:, :-1]
        target = target[:, 1:]
        inp = Variable(inp)
        target = Variable(target)
        if is_remote():
            inp = inp.cuda()
            target = target.cuda()

        return inp, target

    def evaluate(self, batch):

        inp, target = self.get_input_and_target(batch)
        hidden = self.init_hidden(self.opt.batch_size)
        loss = 0

        for w in range((2 * self.opt.sentence_len)):
            output, hidden = self.forward(inp[:, w], hidden)
            loss += self.criterion(output.view(self.opt.batch_size, -1), target[:, w])

        return loss

    def init_hidden(self, batch_size):
        return zeros(gpu=is_remote(), sizes=(self.opt.n_layers_rnn, batch_size, self.opt.hidden_size_rnn))

    def test(self, prime_words, predict_len, temperature=0.8):

        hidden = self.init_hidden(1)
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

