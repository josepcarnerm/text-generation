import string

import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import is_remote, zeros
from nltk.corpus import wordnet as wn

class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()

        self.opt = opt
        self.word_dict = torch.load(self.opt.input_file_train + '.word_dict')
        self.inverted_word_dict = {i:w for w,i in self.word_dict.items()}
        self.word_count = torch.load(self.opt.input_file_train + '.word_count')
        self.N_WORDS = len(self.word_dict)

        self.encoder = nn.Embedding(self.N_WORDS, self.opt.hidden_size_rnn)
        self.rnn = nn.GRU(self.opt.hidden_size_rnn, self.opt.hidden_size_rnn, self.opt.n_layers_rnn)
        self.decoder = nn.Linear(self.opt.hidden_size_rnn, self.N_WORDS)

        self.criterion = nn.CrossEntropyLoss()

        self.submodules = [self.encoder, self.rnn, self.decoder, self.criterion]

        self.losses_reconstruction = []
        self.losses_topic = []

    def from_string_to_tensor(self, sentence):
        tensor = torch.zeros(len(sentence)).long()
        for word_i in range(len(sentence)):
            try:
                tensor[word_i] = self.word_dict[sentence[word_i]]
            except:
                continue
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

    def closeness_to_topics(self, words_weights, topics):

        _, words = words_weights.max(1)

        dist = nn.PairwiseDistance(p=2)

        closeness = []

        for i in range(len(topics)):
            topic_str = self.inverted_word_dict[topics[i].data[0]]
            topic = topics[i]
            word = words[i]
            synonyms = [topic.data[0]]
            for syn in wn.synsets(topic_str):
                synonyms += [self.word_dict[l.name()] for l in syn.lemmas() if l.name() in self.word_dict]

            synonyms = torch.from_numpy(numpy.array(synonyms))
            synonyms = Variable(synonyms).cuda() if is_remote() else Variable(synonyms)

            closeness.append(torch.mean(torch.stack([dist(self.encoder(syn), self.encoder(word)) for syn in synonyms])))

        return torch.stack(closeness)

    def select_topics(self, batch):

        # batch is weirdly ordered due to Pytorch Dataset class from which we inherit in each dataloader : sizes are sentence_len x batch_size

        # Is noun, adjective, verb or adverb?
        is_nava = lambda word: len(wn.synsets(word)) != 0

        # Select "topic" as the least common noun, verb, adjective or adverb in each sentence
        topics = torch.LongTensor(self.opt.batch_size, 1)
        for i in range(len(batch[0])):
            sentence = [batch[j][i] for j in range(len(batch))]
            words_sorted = sorted([(self.word_count[word], word) for word in set(sentence) if is_nava(word)])
            least_common_word = words_sorted[0][1] if len(words_sorted) > 0 else sentence[0]
            topics[i] = self.from_string_to_tensor([least_common_word])

        return Variable(topics).cuda() if is_remote() else Variable(topics)

    def encode_sentences_to_var(self, batch):

        # Convert sentences (list of words - strings) in batch to tensors, each word is now represented as an integer
        inp = torch.LongTensor(self.opt.batch_size, self.opt.sentence_len + 1)
        target = torch.LongTensor(self.opt.batch_size, self.opt.sentence_len + 1)
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

        topics = self.select_topics(batch)
        inp, target = self.encode_sentences_to_var(batch)

        loss_reconstruction = 0
        loss_topic = 0

        # Topic is provided as an initialization to the hidden state
        hidden = self.encoder(topics)

        # Encode/Decode sentence
        loss_topic_total_weight = 0
        for w in range(self.opt.sentence_len):

            output, hidden = self.forward(inp[:, w], hidden)

            # Reconstruction Loss
            loss_reconstruction += self.criterion(output, target[:, w])

            # Topic closeness loss: Weight each word contribution by the inverse of it's frequency
            _, words_i = output.max(1)
            loss_topic_weights = Variable(torch.from_numpy(numpy.array(
                [1/self.word_count[self.inverted_word_dict[i.data[0]]] for i in words_i]
            )).unsqueeze(1)).float()
            loss_topic_weights = loss_topic_weights.cuda() if is_remote() else loss_topic_weights
            loss_topic_total_weight += loss_topic_weights
            loss_topic += self.closeness_to_topics(output, topics) * loss_topic_weights


        loss_topic = torch.mean(loss_topic/loss_topic_total_weight)

        self.losses_reconstruction.append(loss_reconstruction.data[0])
        self.losses_topic.append(loss_topic.data[0])

        return self.opt.loss_alpha*loss_reconstruction + (1-self.opt.loss_alpha)*loss_topic

    def init_hidden(self, batch_size):
        return zeros(gpu=is_remote(), sizes=(self.opt.n_layers_rnn, batch_size, self.opt.hidden_size_rnn))

    def select_word_index_from_one_hot_encoding(self, one_hot, temperature=0.8):
        word_dist = one_hot.div(temperature).exp()
        _, top_indexes = word_dist.max(1)
        # top_indexes = torch.multinomial(word_dist, 1)
        return top_indexes

    def test(self, prime_words, predict_len, temperature=0.8):

        topic = self.select_topics([['happy']])
        hidden = self.encoder(topic)

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

    # Debug
    def get_avg_losses(self):
        return numpy.mean(self.losses_reconstruction), numpy.mean(self.losses_topic)
