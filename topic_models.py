import numpy as np
import nltk, pdb, re
from collections import Counter
from utils import glove2dict
from sklearn import decomposition
from scipy import linalg


class OneFileDataloader(object):
    def __init__(self):
        input_file_train = './data/shakespeare_train.txt'
        input_file_test = './data/shakespeare_test.txt'
        glove_file = './data/glove.6B/glove.6B.50d.txt'
        with open(input_file_train, 'r', encoding = 'utf-8', errors='ignore') as f:
            self.file = f.read()

        self.file = re.sub(' {2,}', ' ', self.file)
        self.file = re.sub('\n{1,', '\n', self.file)

        try:
            tokens = nltk.word_tokenize(self.file)
        except LookupError:
            nltk.download('punkt')
            tokens = nltk.word_tokenize(self.file)

        self.glv_dict = glove2dict(glove_file)
        tokens = self.process_unknown_words(tokens)
        self.vocab = set(tokens)
        self.docsize = 25

        self.documents = self.split_tokens_into_documents(tokens)
        self.len = len(self.documents)
        self.doc_term_matrix = self.generate_doc_term_matrix()

    def get_matrix(self):
        return self.doc_term_matrix

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
            sentence.append(tok)
            if tok in punct:
                i += 1
            if i % self.docsize == 0:
                documents.append(sentence)
                sentence.clear()
        return documents

    def generate_doc_term_matrix(self):
        '''
        This function converts self.documents to numpy word-doc matrix
        :return: the word-document matrix
        '''
        # for i, word in enumerate(self.vocab):
        vocab = list(self.vocab)
        word_to_index = {w: i for i, w in enumerate(vocab)}
        rownames = np.array([[w] for w in vocab])

        n_docs = len(self.documents)
        colnames = np.array(['D'+ str(n) for n in range(n_docs)])
        matrix = np.zeros([len(rownames), len(colnames)])
        for doc in self.documents:
            word_count = Counter(doc)
            for word, count in word_count.items():
                matrix[word_to_index[word]] = count
        return rownames, matrix

if __name__ == '__main__':
    dataloader = OneFileDataloader()
    rownames, matrix = dataloader.get_matrix()
    pdb.set_trace()
    num_topics, num_top_words = 6, 8
