import numpy
import torch, unidecode, random, os, pdb
from torch.utils.data import Dataset
from collections import Counter

from utils import ALL_CHARS, is_remote, glove2dict


class MyDataset(Dataset):

    NON_DIGITS = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n\t\x0b\x0c"
    END_SENTENCE = "!?."

    def __init__(self, opt, train):
        self.opt = opt
        self.train = train

        self.file = open(self.opt.data_dir + self.opt.input_file, "r",encoding='utf-8', errors='ignore').read()

        if self.opt.use_pretrained_embeddings:
            self.glv_dict = glove2dict(self.opt.glove_dir)

        self.preprocess_sentences()
        self.get_words()

        if self.opt.use_pretrained_embeddings:
            self.create_word_dict_glove()
            self.create_word_count_glove()
        else:
            self.create_word_dict()
            self.create_word_count()
        self.len = len(self.sentences)

    def preprocess_sentences(self):
        sentences_file = self.opt.data_dir + self.opt.input_file + \
                         ('.sentences.preprocess' if not self.opt.use_pretrained_embeddings else '.sentences.g_preprocess')
        if not os.path.isfile(sentences_file):
            self.words = str(self.file)
            for c in self.NON_DIGITS:
                self.words = self.words.replace(c, ' ' + c + ' ')

            for c in self.END_SENTENCE:
                self.words = self.words.replace(c, self.END_SENTENCE[0])
            self.sentences = self.words.split(self.END_SENTENCE[0])

            self.sentences = [
                [word.lower() for word in sentence.split(' ') if word.strip() != '']
                for sentence in self.sentences
            ]

            # When using pretrained glove vectors, only pick sentences whose words (all of them) have its corresponding glove vector
            if self.opt.use_pretrained_embeddings:
                self.sentences = [
                    sentence for sentence in self.sentences if all(self.glv_dict.get(word) is not None for word in sentence)
                ]

            self.sentences = [sentence for sentence in self.sentences if len(sentence) > self.opt.sentence_len]

            self.sentences_all = {'train': [], 'test': []}
            for sentence in self.sentences:
                if numpy.random.uniform() > 0.75:
                    self.sentences_all['test'].append(sentence)
                else:
                    self.sentences_all['train'].append(sentence)


            # numpy.random.shuffle(self.sentences)
            # n_train = int(len(self.sentences)*0.75)
            # self.sentences_all = {'train': self.sentences[:n_train], 'test': self.sentences[n_train:]}
            torch.save(self.sentences_all, sentences_file)
        else:
            self.sentences_all = torch.load(sentences_file)

        if self.train:
            self.sentences = self.sentences_all['train']
        else:
            self.sentences = self.sentences_all['test']

    def get_words(self):
        self.words = []
        for sentence in self.sentences_all['train'] + self.sentences_all['test']:
            self.words += list(sentence)

    def create_word_dict(self):
        word_dict_file = self.opt.data_dir + self.opt.input_file + '.sentences.word_dict'
        word_dict = {w: i for i, w in enumerate(set(self.words))}
        torch.save(word_dict, word_dict_file)

    def create_word_count(self):
        word_count_file = self.opt.data_dir + self.opt.input_file + '.sentences.word_count'
        word_count = Counter(self.words)
        torch.save(word_count, word_count_file)

    def create_word_dict_glove(self):
        word_dict_file = self.opt.data_dir + self.opt.input_file + '.sentences.g_word_dict'
        word_dict = {w: self.glv_dict.get(w) for w in set(self.words)}
        torch.save(word_dict, word_dict_file)

    def create_word_count_glove(self):
        word_count_file = self.opt.data_dir + self.opt.input_file + '.sentences.g_word_count'
        word_count = Counter(self.words)
        torch.save(word_count, word_count_file)

    def __getitem__(self, index):
        random.seed(index)
        i = random.randint(0, (self.len - 1))
        while len(self.sentences[i]) <= self.opt.sentence_len:
            i = random.randint(0, (self.len - 1))
        return self.sentences[i]

    def __len__(self):
        return self.len

