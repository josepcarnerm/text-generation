
import numpy
import torch, nltk, random, os, re, glob, pdb, pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from torch.autograd import Variable
from torch.utils.data import Dataset
from collections import Counter

from utils import ALL_CHARS, is_remote, glove2dict

nltk.download('punkt')

class MyDataset(Dataset):

    NON_DIGITS = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n\t\x0b\x0c"
    END_SENTENCE = "!?."

    def __init__(self, opt, train):

        self.opt = opt
        self.train = train

        if self.opt.use_pretrained_embeddings:
            self.glv_dict = glove2dict(self.opt.glove_dir)

        self.preprocess_sentences()
        self.get_words()

        if self.opt.use_pretrained_embeddings:
            self.create_word_dict_glove()
            self.create_word_count_glove()
        else:
            self.create_word_dict()
            self.create_word_count()
        self.len = len(self.sentences)

        del self.sentences_all

    def preprocess_sentences(self):
        sentences_file = self.opt.input_folder_path + \
                         ('.sentences.preprocess' if not self.opt.use_pretrained_embeddings else '.sentences.g_preprocess')

        if not os.path.isfile(sentences_file):
            folder_path = self.opt.input_folder_path + "/"
            self.sentences = []
            for filename in glob.glob(folder_path+'*.txt'):
                words = open(filename, "r",encoding='utf-8', errors='ignore').read()

                words = re.sub(' {2,}', ' ', words)
                words = re.sub('\n{1,}', ' ', words)

                sentence_tokenized_words = sent_tokenize(words)

                use_last = True
                while sentence_tokenized_words:
                    sentence = sentence_tokenized_words.pop(0)
                    sentence = word_tokenize(sentence)
                    sentence = [word.lower() for word in sentence if word.strip() != '']
                    while len(sentence) < self.opt.sentence_len:
                        if not sentence_tokenized_words:
                            use_last = False
                            break
                        sentence = sentence_tokenized_words.pop(0)
                        sentence = word_tokenize(sentence)
                        sentence = [word.lower() for word in sentence if word.strip() != '']
                    if use_last:
                        if len(sentence) > self.opt.sentence_len:
                            self.sentences.append(sentence)

            # When using pretrained glove vectors, only pick sentences whose words (all of them) have its corresponding glove vector
            if self.opt.use_pretrained_embeddings:
                self.sentences = [
                    sentence for sentence in self.sentences if all(self.glv_dict.get(word) is not None for word in sentence)
                ]

            numpy.random.shuffle(self.sentences)
            n_train = int(len(self.sentences)*0.75)
            self.sentences_all = {'train': self.sentences[:n_train], 'test': self.sentences[n_train:]}
            # torch.save(self.sentences_all, sentences_file)

            output = open(sentences_file, 'wb')
            pickle.dump(self.sentences_all, output)
            output.close()

            self.topic_len = len(self.sentences)

        else:
            self.sentences_all = torch.load(sentences_file)

        if self.train:
            self.sentences = self.sentences_all['train']
        else:
            self.sentences = self.sentences_all['test']


    def get_words(self):
        self.words = []
        for sentence in self.sentences_all['train'] + self.sentences_all['test']:
            self.words += list(sentence)

    def create_word_dict(self):
        word_dict_file = self.opt.input_folder_path + '.sentences.word_dict'
        word_dict = {w: i for i, w in enumerate(set(self.words))}
        # torch.save(word_dict, word_dict_file)

        output = open(word_dict_file, 'wb')
        pickle.dump(word_dict, output)
        output.close()

    def create_word_count(self):
        word_count_file = self.opt.input_folder_path + '.sentences.word_count'
        word_count = Counter(self.words)
        # torch.save(word_count, word_count_file)

        output = open(word_count_file, 'wb')
        pickle.dump(word_count, output)
        output.close()

    def create_word_dict_glove(self):
        word_dict_file = self.opt.input_folder_path + '.sentences.g_word_dict'
        word_dict = {w: self.glv_dict.get(w) for w in set(self.words)}
        # torch.save(word_dict, word_dict_file)

        output = open(word_dict_file, 'wb')
        pickle.dump(word_dict, output)
        output.close()

    def create_word_count_glove(self):
        word_count_file = self.opt.input_folder_path + '.sentences.g_word_count'
        word_count = Counter(self.words)
        # torch.save(word_count, word_count_file)

        output = open(word_count_file, 'wb')
        pickle.dump(word_count, output)
        output.close()

    def __getitem__(self, index):
        random.seed(index)
        return random.sample(self.sentences, 1)[0]

    def __len__(self):
        return self.len