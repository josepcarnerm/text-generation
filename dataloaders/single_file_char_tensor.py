import torch
import unidecode
from torch.autograd import Variable
from torch.utils.data import Dataset
import random
from utils import ALL_CHARS, is_remote

class MyDataset(Dataset):

    def __init__(self, opt, train):
        self.opt = opt
        if train:
            self.file = open(self.opt.input_file_train, "r",encoding='utf-8', errors='ignore').read()
        else:
            self.file = open(self.opt.input_file_test, "r",encoding='utf-8', errors='ignore').read()
        self.len = len(self.file)

    def __getitem__(self, index):
        # random.seed(index)
        inp = torch.LongTensor(self.opt.sentence_len)
        target = torch.LongTensor(self.opt.sentence_len)
        start_index = random.randint(0, self.len - self.opt.sentence_len)
        end_index = start_index + self.opt.sentence_len + 1
        chunk = self.file[start_index:end_index]
        inp = self.char_tensor(chunk[:-1])
        target = self.char_tensor(chunk[1:])
        return inp, target

    def __len__(self):
        return self.len-self.opt.sentence_len

