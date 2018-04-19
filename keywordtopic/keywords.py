import random, glob, os, pdb, nltk
from gensim.summarization import keywords


class KeywordTopic(object):
    def __init__(self, sentences, fpath='', n_lines=2000, author='shakespeare'):
        if not sentences and not fpath:
            raise ValueError('Cannot initialize KeywordTopic without sentences or filepath')

        self.sample_text = None
        if sentences:
            self.sample_text = ' '.join(random.sample([' '.join(sentence) for sentence in sentences], n_lines))

        else:
            files = glob.iglob(fpath)
            files = [f for f in files if author in f]
            self.sample_text = ' '.join(self.random_lines(fpath, files, n_lines))

        assert self.sample_text

        try:
            words = nltk.corpus.stopwords.words('english')
        except LookupError:
            nltk.download('stopwords')
            words = nltk.corpus.stopwords.words('english')
        self.stopwords = set(words + ['thy', 'shall', 'thou', 'thee'])
        self.keywords = self.get_base_keywords()

    def yield_base_keywords(self):
        return self.keywords

    def get_base_keywords(self, n_words=25, single_words_only=True, scores=False):
        '''
        Gets list of n_words keywords from the supplied sample text. Note, it throws away keywords that are of length 2
        \n:param n_words: max number of words in the keyword list
        \n:return: list of strings representing keywords if scores == false, else list of (str, float) representing (keyword, score)
        '''
        words = keywords(self.sample_text, words=n_words, scores=scores, split=True)
        if isinstance(words, str):
            words = [w for w in words if w not in self.stopwords]
            if single_words_only:
                words = [w for w in words if len(w.split(' ')) == 1]
        else:
            words = [w for w in words if w[0] not in self.stopwords]
            if single_words_only:
                words = [w for w in words if len(w[0].split(' ')) == 1]
        return words

    def get_eval_keywords(self, sentence, n_words=5, scores=True):
        '''
        Gets a list of n_words keywords (and scores) from a given evaluation sentence
        \n:param sentence: input sentence, list or string
        \n:param n_words: number of keywords to identify
        \n:param scores: keyword scores
        \n:return: list of (str, float) representing (keyword, score)
        '''
        if isinstance(sentence, list):
            sentence = ' '.join(sentence)
        return keywords(sentence, words=n_words, scores=scores, split=True)

    @staticmethod
    def random_lines(fdir, files, n_lines):
        '''
        Gets one long string composed of sample_lines # of random lines from the provided filenames
        \n:param fdir: base directory for files
        \n:param files: single string filename or list of filenames
        \n:param n_lines: number of sample lines
        \n:return: string containing the line samples
        '''
        lines_per_file = n_lines/len(files)
        ret_val = []
        if isinstance(files, str):
            with open(os.path.join(fdir, files), encoding='utf-8') as f:
                ret_val = random.sample(f.readlines(), lines_per_file)
        else:
            for fname in files:
                with open(os.path.join(fdir, fname), encoding='utf-8') as f:
                    ret_val = ret_val + random.sample(f.readlines(), lines_per_file)

        return ret_val
