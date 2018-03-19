import os, re, spacy, torch
import numpy as np

from collections import defaultdict
from torch.utils.data import Dataset

import glove_utils
from utils import is_remote

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

        if self.use_existing_wordvecs():
            print("Loaded pretrained wordvecs from {}".format(self.saved_wordvecs))
            self.words_matrix = torch.load(self.saved_wordvecs)
        else:
            self.words_matrix = self.get_pretrained_vectors()
            print("Saved pretrained wordvecs to {}".format(self.saved_wordvecs))
            torch.save(self.words_matrix, self.saved_wordvecs)
        self.len = self.words_matrix.size()[0]


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
        return torch.stack(retVal)

    # try to load pretrained word vector from file (to save time on local development)
    def use_existing_wordvecs(self):

        vec_dir = self.opt.wordvec_out_dir
        pretrained_vec_dim = self.opt.input_pretrained_vector.split('/')[-1].split('.')[-2]
        pretrained_vec_file = self.opt.input_pretrained_vector.split('/')[1]+"embeddings"+pretrained_vec_dim+".txt"
        self.saved_wordvecs = os.path.join(vec_dir, pretrained_vec_file)

        if self.opt.force_reload_wordvecs == 'yes':
            return False

        try:
            file = open(self.saved_wordvecs, 'r')
            file.close()
            return True
        except FileNotFoundError:
            print("Unable to load {}. Doesn't exist or bad path.".format(self.input_file))
            return False
