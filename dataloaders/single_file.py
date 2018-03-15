import unidecode
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, opt, train):
        self.opt = opt
        if train:
            self.file = open(self.opt.input_file_train, "r",encoding='utf-8', errors='ignore').read()
        else:
            self.file = open(self.opt.input_file_test, "r",encoding='utf-8', errors='ignore').read()
        self.len = len(self.file)

    def __getitem__(self, index):
        return self.file[index:(index+self.opt.sentence_len)]

    def __len__(self):
        return self.len-self.opt.sentence_len

