# import unidecode
from torch.utils.data import Dataset


class MyDataset(Dataset):

    def __init__(self, opt, train):
        self.opt = opt
        if train:
            self.file = open(self.opt.input_file_train).read()
        else:
            self.file = open(self.opt.input_file_test).read()
        self.len = len(self.file)

    def __getitem__(self, index):
        return self.file[index:(index+self.opt.sentence_len)]

    def __len__(self):
        return self.len-self.opt.sentence_len

