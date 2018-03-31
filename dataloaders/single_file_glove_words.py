import torch, random, os, re, nltk
from torch.utils.data import Dataset
from collections import Counter

from utils import glove2dict


class MyDataset(Dataset):

    def __init__(self, opt, train):
        self.opt = opt
        self.train = train

        with open(self.opt.input_file_train, 'r', encoding='utf-8', errors='ignore') as f:
            self.words = str(f.read())

        self.words = re.sub(' {2,}', ' ', self.words)
        self.words = re.sub('\n{1,}', '\n', self.words)

        try:
            self.words = nltk.word_tokenize(self.words)
        except LookupError:
            nltk.download('punkt')
            self.words = nltk.word_tokenize(self.words)

        self.glv_dict = glove2dict(self.opt.glove_dir)

        self.words = self.process_unknown_words()
        self.create_word_dict()
        self.create_word_count()

        self.len = len(self.words)

    def process_unknown_words(self):
        '''
        This function converts shakespeare words ending in 'd to words ending in ed
        example: kill'd -> killed
        \n Also replaces words not in glove dict with 'unk'
        --------
        :return: new word list
        '''
        new_words = []
        for word in self.words:
            if word.endswith('\'d'):
                new_words.append(word.lower()[:-2]+'ed')
            else:
                new_words.append(word.lower())

        return [word if self.glv_dict.get(word) is not None else 'unk' for word in new_words]

    def create_word_dict(self):
        '''
        Create word->GloVe vector dictionary given opt.glove_dir
        --------
        :return: nothing. Serializes the word dict to file to be read by the model
        '''
        word_dict_file = self.opt.input_file_train + '.g_word_dict'
        if not os.path.isfile(word_dict_file):
            word_dict = {w: self.glv_dict.get(w) for w in set(self.words)}
            torch.save(word_dict, word_dict_file)

    def create_word_count(self):
        '''
        Create word count file
        --------
        :return: nothing. Serializes the word count to file to be read by the model
        '''
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

