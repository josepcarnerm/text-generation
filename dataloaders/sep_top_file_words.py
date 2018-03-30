import torch, unidecode, random, os
from torch.autograd import Variable
from torch.utils.data import Dataset
from collections import Counter

from utils import ALL_CHARS, is_remote


class MyDataset(Dataset):

    NON_DIGITS = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n\t\x0b\x0c"

    def __init__(self, opt, train):
        self.opt = opt
        self.train = train
        self.file = open(self.opt.input_file_train, "r",encoding='utf-8', errors='ignore').read()
        self.topic_file = open(self.opt.topic_file, "r", encoding='utf-8', errors='ignore').read()

        self.words = str(self.file)
        for c in self.NON_DIGITS:
            self.words = self.words.replace(c, ' '+c+' ')
        self.words = [word.lower() for word in self.words.split(' ') if word.strip() != '']

        self.topic_words = str(self.file)
        for c in self.NON_DIGITS:
            self.topic_words = self.topic_words.replace(c, ' '+c+' ')
        self.topic_words = [word.lower() for word in self.topic_words.split(' ') if word.strip() != '']

        self.create_word_dict(self.opt.input_file_train, self.words)
        self.create_word_count(self.opt.input_file_train, self.words)
        self.create_word_dict(self.opt.topic_file, self.topic_words)
        self.create_word_count(self.opt.topic_file, self.topic_words)
        self.len = len(self.words)
        self.topic_len = len(self.topic_words)

    def create_word_dict(self, path, words):
        # word_dict_file = self.opt.input_file_train + '.word_dict'
        word_dict_file = path + '.word_dict'
        if not os.path.isfile(word_dict_file):
            word_dict = {w: i for i, w in enumerate(set(words))}
            torch.save(word_dict, word_dict_file)

    def create_word_count(self, path, words):
        word_count_file = path + '.word_count'
        if not os.path.isfile(word_count_file):
            word_count = Counter(words)
            torch.save(word_count, word_count_file)

    def __getitem__(self, index):
        random.seed(index)
        start_index = random.randint(0, self.len - self.opt.sentence_len)
        end_index = start_index + self.opt.sentence_len+1
        sentence = self.words[start_index:end_index]

        start_index_topic = random.randint(0, self.topic_len - self.opt.sentence_len)
        end_index_topic = start_index_topic + self.opt.sentence_len+1
        topic_sentence = self.topic_words[start_index_topic:end_index_topic]
        print("Sentence being fed in: " + " ".join(topic_sentence))
        return topic_sentence # + sentence

    def __len__(self):
        return self.len-self.opt.sentence_len

