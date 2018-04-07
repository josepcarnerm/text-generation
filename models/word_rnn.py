import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import is_remote, zeros


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()

        self.opt = opt
        self.load_word_dicts()
        self.N_WORDS = len(self.word2idx)

        self.encoder = nn.Embedding(self.N_WORDS, self.opt.hidden_size_rnn)
        self.rnn = nn.LSTM(self.opt.hidden_size_rnn, self.opt.hidden_size_rnn, self.opt.n_layers_rnn, dropout=self.opt.dropout)
        self.decoder = nn.Linear(self.opt.hidden_size_rnn, self.N_WORDS)

        if self.opt.use_pretrained_embeddings:
            embeddings = torch.zeros((len(self.word2idx)), self.word_dict_dim).float()
            for k, v in self.word2idx.items():
                embeddings[v] = self.word_dict[k]
            self.encoder.weight = nn.Parameter(embeddings)
            self.encoder.weight.requires_grad = False
            if self.opt.model == 'word_rnn':
                del self.word_dict  # Clear the memory

        self.criterion = nn.CrossEntropyLoss()

        self.submodules = [self.encoder, self.rnn, self.decoder, self.criterion]

    def load_word_dicts(self):
        if self.opt.use_pretrained_embeddings:
            self.word_dict = torch.load(self.opt.data_dir + self.opt.input_file + '.sentences.g_word_dict')
            self.opt.hidden_size_rnn = self.word_dict['the'].size(0)
            self.word_dict_dim = self.opt.hidden_size_rnn
            self.word2idx = {word: idx for idx, word in enumerate((self.word_dict.keys()))}
        else:
            self.word2idx = torch.load(self.opt.data_dir + self.opt.input_file + '.sentences.word_dict')

        self.inverted_word_dict = {i: w for w, i in self.word2idx.items()}

    def from_string_to_tensor(self, sentence):
        tensor = torch.LongTensor(len(sentence))
        for i, word in enumerate(sentence):
            try:
                tensor[i] = self.word2idx[word]
            except:
                continue
        return tensor

    @staticmethod
    def select_word_index_from_output(output, temperature=0.8):
        # Output: batch_size x N_words
        _, top_indexes = output.max(1)  # Select word as the with "highest probability"
        # top_indexes.size() = (batch_size)
        return top_indexes

    def from_predicted_index_to_string(self, index):
        return self.inverted_word_dict[index]

    def forward(self, input, hidden):
        batch_size = input.size(0)  # Will be self.opt.batch_size at train time, 1 at test time
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.contiguous().view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def forward2(self, input, hidden):
        encoded = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def get_input_and_target(self, batch):

        if len(batch) == 2:  # Topic included. Batch is: topics, sentences
            batch = batch[1]

        batch_size, sentence_len = len(batch[0]), len(batch)-1
        inp = torch.LongTensor(batch_size, sentence_len + 1)
        target = torch.LongTensor(batch_size, sentence_len + 1)
        for i in range(sentence_len + 1):
            sentence = batch[i]
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
        last_output = inp[:, 0]  # Only used if "reuse_pred" is set

        for w in range(self.opt.sentence_len):
            x = last_output if self.opt.reuse_pred else inp[:, w]
            output, hidden = self.forward(x, hidden)
            last_output = self.select_word_index_from_output(output)
            loss += self.criterion(output.view(self.opt.batch_size, -1), target[:, w])

        return loss

    def perplexity(self, batch):
        loss = eval(batch)
        return torch.exp(loss.data[0])

    def init_hidden(self, batch_size):
        return (zeros(gpu=is_remote(), sizes=(self.opt.n_layers_rnn, batch_size, self.opt.hidden_size_rnn)),
                    zeros(gpu=is_remote(), sizes=(self.opt.n_layers_rnn, batch_size, self.opt.hidden_size_rnn)))

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
