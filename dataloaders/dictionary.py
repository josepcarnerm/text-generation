from torch.utils.data import Dataset
import random, os, torch
from PyDictionary import PyDictionary
from random_words import RandomWords


class MyDataset(Dataset):

    NON_DIGITS = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n\t\x0b\x0c"

    def __init__(self, opt, train):
        self.opt = opt
        self.rw = RandomWords()
        self.dictionary=PyDictionary()
        self.prob_repeat_example = 0.25
        self.n_items = int(1/self.prob_repeat_example)
        self.word_dict()

    def word_dict(self):
        word_dict_file = self.opt.save_dir+ 'dictionary.word_dict'
        if not os.path.isfile(word_dict_file):
            n_words = self.opt.batch_size*self.opt.n_epochs*self.opt.epoch_size*self.n_items
            self.words = [word for word in self.rw.random_words(count=n_words) if self.dictionary.meaning(word)] + ['happy']
            word_dict = {w: i for i, w in enumerate(set(self.words))}
            torch.save(word_dict, word_dict_file)
        else:
            word_dict = torch.load(word_dict_file)
            self.words = word_dict.keys()
        self.len = len(self.words)

    def __getitem__(self, index):
        random.seed(index)
        i = random.randint(0, self.len)
        w = self.words[i]
        return w, self.dictionary.meaning(w)

    def __len__(self):
        return self.len
