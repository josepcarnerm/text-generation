import torch, re
import nltk
import numpy as np
from torch.utils.data import Dataset
from collections import Counter
import random
from utils import glove2dict

class MyDataset(Dataset):

    def __init__(self, opt, train):
        self.opt = opt

        if train:
            with open(self.opt.input_file_train, 'r', encoding='utf-8', errors='ignore') as f:
                self.file = f.read()
        else:
            with open(self.opt.input_file_test, 'r', encoding='utf-8', errors='ignore') as f:
                self.file = f.read()

        self.file = re.sub(' {2,}', ' ', self.file)
        self.file = re.sub('\n{1,}', '\n', self.file)

        try:
            tokens = nltk.word_tokenize(self.file)
        except LookupError:
            nltk.download('punkt')
            tokens = nltk.word_tokenize(self.file)

        # Load glove vectors to process unknown words (this is wasteful but it helps)
        self.glv_dict = glove2dict(self.opt.glove_dir)
        tokens = self.process_unknown_words(tokens)
        self.vocab = set(tokens)
        # Clear the memory
        del self.glv_dict

        self.documents = self.split_tokens_into_documents(tokens)
        self.len = len(tokens)

    def generate_doc_term_matrix(self):
        '''
        This function converts self.documents to numpy word-doc matrix
        :return: the word-document matrix
        '''
        i, words = enumerate(self.vocab)
        word_to_index = {w: j for w, j in zip( words, i )}
        rownames = np.array([[w] for w in words])

        n_docs = len(self.documents)
        colnames = np.array(['D'+ n for n in range(n_docs)])
        matrix = np.zeros([len(rownames), len(colnames)])
        for doc in self.documents:
            word_count = Counter(doc)
            for word, count in word_count.items():
                matrix[word_to_index[word]] = count
        return matrix


    def process_unknown_words(self, words):
        '''
        This function converts shakespeare words ending in 'd to words ending in ed
        example: kill'd -> killed
        \n Also replaces words not in glove dict with 'unk'
        --------
        :return: new word list
        '''
        new_words = []
        for word in words:
            if word.endswith('\'d'):
                new_words.append(word.lower()[:-2]+'ed')
            else:
                new_words.append(word.lower())

        return [word if self.glv_dict.get(word) is not None else 'unk' for word in new_words]

    def split_tokens_into_documents(self, tokens):
        '''
        Takes a list of tokens and returns a list of lists such that self.opt.docsize sentences is a 'document' for consideration in LDA
        Todo: generalize this for multiple documents in the next iteration
        \n :param tokens: a list of tokens tokenized by nltk
        \n :return: list of lists [ [docsize] for docsize in single-file corpus].
        '''
        documents, sentence, i = [], [], 1
        punct = ['.', '!', '?']
        for tok in tokens:
            sentence.append[tok]
            if tok in punct:
                i += 1
            if i % self.opt.docsize == 0:
                documents.append(sentence)
                sentence.clear()
        return documents




    def __getitem__(self, index):
        return self.doc_word_matrix

    def __len__(self):
        return 1

