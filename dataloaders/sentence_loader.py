import unidecode
from torch.utils.data import Dataset
import re, spacy
import numpy as np

from collections import defaultdict
from torch.utils.data import Dataset

import glove_utils

class MyDataset(Dataset):

    def __init__(self, opt, train):
        self.opt = opt
        if train:
            with open(self.opt.input_file_train, 'r', encoding='utf-8', errors='ignore') as f:
                self.file = f.read()
        else:
            with open(self.opt.input_file_test, 'r', encoding='utf-8', errors='ignore') as f:
                self.file = f.read()

        self.nlp = spacy.load('en')
        self.tokens = self.make_tokens()

        self.glv_dict = glove_utils.glove2dict(opt.input_pretrained_vector)
        print("Getting pretrained vectors from: {} ".format(opt.input_pretrained_vector))
        self.words_matrix = self.get_pretrained_vectors()
        print(self.words_matrix)
        print("Number of tokens in document = {}".format(self.len))

    def __getitem__(self, index):
        return self.words_matrix[index:(index+self.opt.sentence_len)]

    def __len__(self):
        return self.len-self.opt.sentence_len

    def tokenizer(self, sentences):
        return [tok.text.lower() for tok in self.nlp.tokenizer(sentences)]

    def prepare_text(self):
        # remove excess spaces, \n characters
        self.file = re.sub(" {1,}", " ", self.file)
        self.file = re.sub("\n", " ", self.file)

    def make_tokens(self):
        self.prepare_text()
        tokens = self.tokenizer(self.file)
        for i in range(len(tokens)):
            # sub out special words such as kill'd for killed
            if tokens[i][-2:] == "'d'":
                tokens[i] = tokens[i][:-2] + "ed"
            # fix some spacy specific tokenization bugs around '-'
            if tokens[i][-2:] == '.-':
                tokens[i] = tokens[i][:-2]
                tokens.insert(i, '.')
                tokens.insert(i+1, '-')
            if tokens[i][-2:] == ',-':
                tokens[i] = tokens[i][:-2]
                tokens.insert(i, ',')
                tokens.insert(i+1, '-')
        return tokens

    # return an np.matrix of size vocab * glove_dimension
    def get_pretrained_vectors(self):
        vocab = defaultdict(int)
        retVal = []
        for tok in self.tokens:
            # replace unknown words with <unk>
            if self.glv_dict.get(tok) is None:
                vocab['unk'] += 1
                retVal.append(self.glv_dict.get('unk'))
            else:
                vocab[tok] += 1
                retVal.append(self.glv_dict.get(tok))
        # set length and vocab size
        self.vocab_size = len(vocab)
        self.len = len(retVal)
        return np.stack(retVal, axis=0)