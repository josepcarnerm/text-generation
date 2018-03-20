import torch, unidecode, random, os
from torch.autograd import Variable
from torch.utils.data import Dataset

from utils import ALL_CHARS, is_remote


class MyDataset(Dataset):

    NON_DIGITS = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n\t\x0b\x0c"

    def __init__(self, opt, train):
        self.opt = opt
        self.train = train
        self.file = open(self.opt.input_file_train, "r",encoding='utf-8', errors='ignore').read()

        self.words = str(self.file)
        for c in self.NON_DIGITS:
            self.words = self.words.replace(c, ' '+c+' ')
        self.words = [word.lower() for word in self.words.split(' ') if word.strip() != '']

        self.create_word_dict()
        self.len = len(self.words)

    def create_word_dict(self):
        word_dict_file = self.opt.input_file_train + '.word_dict'
        if not os.path.isfile(word_dict_file):
            word_dict = {w: i for i, w in enumerate(set(self.words))}
            torch.save(word_dict, word_dict_file)

    def __getitem__(self, index):
        random.seed(index)
        start_index = random.randint(0, self.len - self.opt.sentence_len)
        end_index = start_index + self.opt.sentence_len+1
        sentence = self.words[start_index:end_index]
        return sentence

    def __len__(self):
        return self.len-self.opt.sentence_len

