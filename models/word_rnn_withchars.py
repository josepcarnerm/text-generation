import pdb

import torch, string
import torch.nn as nn
from torch.autograd import Variable
from utils import is_remote, zeros


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()

        self.opt = opt
        self.load_word_dicts()
        self.N_WORDS = len(self.word2idx)
        self.char_dict = self.make_char_gram_dict()

        self.N_CHARS = len(self.char_dict)

        self.word_encoder = nn.Embedding(self.N_WORDS, self.opt.hidden_size_rnn)
        self.char_encoder = nn.Embedding(self.N_CHARS, self.opt.hidden_size_rnn)
        self.rnn = nn.LSTM(self.opt.hidden_size_rnn*2, self.opt.hidden_size_rnn, self.opt.n_layers_rnn, dropout=self.opt.dropout)
        self.decoder = nn.Linear(self.opt.hidden_size_rnn, self.N_WORDS)

        if self.opt.use_pretrained_embeddings:
            embeddings = torch.zeros((len(self.word2idx)), self.word_dict_dim).float()
            for k, v in self.word2idx.items():
                embeddings[v] = self.word_dict[k]
            self.word_encoder.weight = nn.Parameter(embeddings)
            self.word_encoder.weight.requires_grad = False
            if self.opt.model == 'word_rnn':
                del self.word_dict  # Clear the memory

        self.criterion = nn.CrossEntropyLoss()

        self.submodules = [self.word_encoder, self.char_encoder, self.rnn, self.decoder, self.criterion]

    def make_char_gram_dict(self):
        ret_val, i = {}, 0
        for word in self.word2idx.keys():
            chargram = word[-self.opt.char_ngram:]
            if ret_val.get(chargram) is None:
                ret_val[chargram] = i
                i += 1
        return ret_val

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

    def from_chars_to_tensor(self, sentence):
        tensor = torch.zeros(len(sentence))
        for i, word in enumerate(sentence):
            tensor[i] = self.char_dict[word[-self.opt.char_ngram:]]
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
        # test
        if batch_size == 2:
            encoded = self.word_encoder(input[:1])
            encoded_chars = self.char_encoder(input[1:2])
            encoded = torch.cat([encoded, encoded_chars], 1)
            output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
            output = self.decoder(output.view(1, -1))
        # train
        else:
            encoded = self.word_encoder(input[:, 0])
            encoded_chars = self.char_encoder(input[:, 1])
            encoded = torch.cat([encoded, encoded_chars], 1)
            output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
            output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    # def forward2(self, input, hidden):
    #     encoded = self.word_encoder(input.view(1, -1))
    #     output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
    #     output = self.decoder(output.view(1, -1))
    #     return output, hidden

    def get_input_and_target(self, batch):

        if len(batch) == 2:  # Topic included. Batch is: topics, sentences
            batch = batch[1]

        batch_size, sentence_len = len(batch[0]), len(batch)-1
        inp = torch.LongTensor(batch_size, sentence_len + 1, 2)
        target = torch.LongTensor(batch_size, sentence_len + 1, 2)
        for i in range(sentence_len + 1):
            sentence = batch[i]
            inp[:, i,0] = self.from_string_to_tensor(sentence)
            inp[:, i,1:] = self.from_chars_to_tensor(sentence)
            target[:, i,0] = self.from_string_to_tensor(sentence)
            target[:, i,1:] = self.from_chars_to_tensor(sentence)
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
            # pdb.set_trace()
            loss += self.criterion(output.view(self.opt.batch_size, -1), target[:, w,0])

        return loss

    def perplexity(self, batch):
        loss = eval(batch)
        return torch.exp(loss.data[0])

    def init_hidden(self, batch_size):
        return (zeros(gpu=is_remote(), sizes=(self.opt.n_layers_rnn, batch_size, self.opt.hidden_size_rnn)),
                    zeros(gpu=is_remote(), sizes=(self.opt.n_layers_rnn, batch_size, self.opt.hidden_size_rnn)))

    def concat_word_with_ngram(self, tensor, string_val=None):
        if string_val:
            char_idx = self.char_dict.get(string_val[-self.opt.char_ngram:])
            word_idx = self.word2idx[string_val]
            return torch.LongTensor([word_idx, char_idx])

        else:
            word_idx = tensor[0].data[0]
            char_idx = self.char_dict.get(self.inverted_word_dict[word_idx][-self.opt.char_ngram:])

        if tensor.size(0) == 1:
            return torch.LongTensor([word_idx, char_idx])
        else:
            ret_val = torch.LongTensor(tensor.size[0], 2)
            for i in range(tensor.size(0)):
                word_idx = tensor[i].data[0]
                char_idx = self.char_dict.get(self.inverted_word_dict[word_idx][-self.opt.char_ngram:])
                ret_val[i,0] = word_idx
                ret_val[i,1] = char_idx
            return Variable(ret_val)


    def test(self, prime_words, predict_len, temperature=0.8):

        hidden = self.init_hidden(1)
        prime_input = self.concat_word_with_ngram(Variable(self.from_string_to_tensor(prime_words).unsqueeze(0)))
        inp = prime_input[-2:]

        if is_remote():
            prime_input = prime_input.cuda()
            inp = Variable(inp).cuda()
        predicted = ' '.join(prime_words)

        # Use priming string to "build up" hidden state
        for p in range(len(prime_words) - 1):
            _, hidden = self.forward(prime_input[:, p], hidden)

        for p in range(predict_len):
            output, hidden = self.forward(inp, hidden)

            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            # Add predicted character to string and use as next input
            predicted_word = self.from_predicted_index_to_string(top_i)
            predicted += ' '+predicted_word
            # pdb.set_trace()
            inp = self.concat_word_with_ngram(None, predicted_word)
            if is_remote():
                inp = Variable(inp).cuda()

        return predicted
