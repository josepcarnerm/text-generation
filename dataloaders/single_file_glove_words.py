import torch, random, os, re, pdb
from torch.utils.data import Dataset
from collections import Counter

from utils import ALL_CHARS, is_remote, build_glove, glove2dict, word_to_idx, word_to_tensor


class MyDataset(Dataset):

    NON_DIGITS = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n\t\x0b\x0c"

    def __init__(self, opt, train):
        self.opt = opt
        self.train = train
        self.file = open(self.opt.input_file_train, "r",encoding='utf-8', errors='ignore').read()

        self.words = str(self.file)

        self.words = re.sub(' {2,}', ' ', self.words)
        self.words = re.sub('\n{1,}', '\n', self.words)

        for c in self.NON_DIGITS:
            self.words = self.words.replace(c, ' '+c+' ')

        self.words = [word.lower() for word in self.words.split(' ') if word.strip() != '']

        self.glv_dict = glove2dict(self.opt.glove_dir)

        self.process_unknown_words()
        self.create_word_dict()
        self.create_word_count()

        self.len = len(self.words)

    def process_unknown_words(self):
        self.words = [word if self.glv_dict.get(word) is not None else 'unk' for word in self.words]

    def create_word_dict(self):
        word_dict_file = self.opt.input_file_train + '.g_word_dict'
        if not os.path.isfile(word_dict_file):
            word_dict = {w: self.glv_dict.get(w) for w in set(self.words)}
            torch.save(word_dict, word_dict_file)

    def create_word_count(self):
        word_count_file = self.opt.input_file_train + '.word_count'
        if not os.path.isfile(word_count_file):
            word_count = Counter(self.words)
            torch.save(word_count, word_count_file)

    def __getitem__(self, index):
        random.seed(index)
        start_index = random.randint(0, self.len - self.opt.sentence_len)
        end_index = start_index + self.opt.sentence_len+1
        sentence = self.words[start_index:end_index]
        return sentence

    def __len__(self):
        return self.len-self.opt.sentence_len

