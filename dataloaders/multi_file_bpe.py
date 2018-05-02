# https://github.com/bheinzerling/bpemb/blob/master/bpe.py
# BP Subword Encodings based on the work of Sennrich Et Al (2016)

import glob, pdb, re
from torch.utils.data import Dataset
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize

from utils import is_remote


from math import log


class BPE(object):

    def __init__(self, vocab_dict):
        self.words = vocab_dict.keys()
        log_len = log(len(self.words))
        self.wordcost = {
            k: log((i+1) * log_len)
            for i, k in enumerate(self.words)}
        self.maxword = max(len(x) for x in self.words)

    def encode(self, s):
        """Uses dynamic programming to infer the location of spaces in a string
        without spaces."""

        s = s.replace(" ", "▁")

        # Find the best match for the i first characters, assuming cost has
        # been built for the i-1 first characters.
        # Returns a pair (match_cost, match_length).
        def best_match(i):
            candidates = enumerate(reversed(cost[max(0, i - self.maxword):i]))
            return min(
                (c + self.wordcost.get(s[i-k-1:i], 9e999), k+1)
                for k, c in candidates)

        # Build the cost array.
        cost = [0]
        for i in range(1, len(s) + 1):
            c, k = best_match(i)
            cost.append(c)

        # Backtrack to recover the minimal-cost string.
        out = []
        i = len(s)
        while i > 0:
            c, k = best_match(i)
            assert c == cost[i]
            out.append(s[i-k:i])

            i -= k

        return " ".join(reversed(out))

class MyDataset(Dataset):
    def __init__(self, opt, train):
        self.opt = opt
        self.train = train

        self.words = self.read_words()
        self.vocab = self.create_vocab(self.words)
        self.bpe = BPE(self.vocab)
        pdb.set_trace()
        self.encoded = self.bpe.encode(self.words)
        pdb.set_trace()

        self.len = len(self.words)

    def read_words(self):
        words = ""
        folder_path = self.opt.input_folder_path + "/"
        print('Reading directory '+folder_path+'...')
        print('This includes {}'.format(glob.glob(folder_path+'*.txt')))
        for filename in glob.glob(folder_path+'*.txt'):
            with open(filename) as f:
                words += f.read().lower()
        words = re.sub('[0-9][0-9]*', '0', words)
        words = re.sub('[\n]', ' ', words)
        return words

    def create_vocab(self, words):
        tokens = word_tokenize(words)
        return Counter(tokens)

    def bpe_words(self, vocab):
        pass
