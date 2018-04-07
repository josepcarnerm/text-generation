import random
import pdb
from keywordtopic.keywords import KeywordTopic
from dataloaders.single_file_str_sentences import MyDataset as sentences


class MyDataset(sentences):
    def __init__(self, opt, train):
        super(MyDataset, self).__init__(opt, train)
        if train:
            self.kw_topic = KeywordTopic(self.sentences_all['train'])
        else:
            self.kw_topic = KeywordTopic(self.sentences_all['test'])
        # self.keywords = self.kw_topic.get_base_keywords(scores=False)
        # print("Got the following keywords: {}".format(self.keywords))

    def __getitem__(self, index):
        random.seed(index)
        i = random.randint(0, (self.len - 1))
        sentence = self.sentences[i]
        try:
            topic = self.kw_topic.get_eval_keywords(' '.join(sentence[0]), scores=False, n_words=1)
        except ZeroDivisionError:
            topic = 'none'
        return topic, sentence
