import string

import nltk
nltk.download('wordnet')

import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import is_remote, zeros
from nltk.corpus import wordnet as wn
from models.word_rnn_topic_provided import Model as WordRNNModelTopic


class Model(WordRNNModelTopic):

    def __init__(self, opt):
        super(Model, self).__init__(opt)

    def analyze_topics(self, batch):
        batch_size, sentence_len = len(batch[0]), len(batch) - 1

        # Analyze select_topics
        examples = []
        # Is noun, adjective, verb or adverb?
        is_nava = lambda word: len(wn.synsets(word)) != 0

        # Select "topic" as the least common noun, verb, adjective or adverb in each sentence
        print('Analyzing topic candidates......')
        for i in range(len(batch[0])):
            sentence = [batch[j][i] for j in range(len(batch))]
            words_sorted = sorted([(self.word_count[word], word) for word in set(sentence) if is_nava(word)])
            least_common_word = words_sorted[0][1] if len(words_sorted) > 0 else sentence[0]
            examples.append({'sentence': sentence, 'topic candidates': words_sorted})

        for e in examples:
            try:
                print('Sentence: {}. Topic candidates: {}.'.format(' '.join(e['sentence']), e['topic candidates']))
            except:
                print('Exception when printing')

    def get_test_topic(self):
        return self.select_topics([['happy']])

    def select_topics(self, batch):

        # batch is weirdly ordered due to Pytorch Dataset class from which we inherit in each dataloader : sizes are sentence_len x batch_size

        # Is noun, adjective, verb or adverb?
        is_nava = lambda word: len(wn.synsets(word)) != 0
        batch_size = len(batch[0])

        # Select "topic" as the least common noun, verb, adjective or adverb in each sentence
        topics = torch.LongTensor(batch_size, 1)
        topics_words = []
        for i in range(batch_size):
            sentence = [batch[j][i] for j in range(len(batch))]
            words_sorted = sorted([(self.word_count[word], word) for word in set(sentence) if is_nava(word) and word in self.word2idx])
            least_common_word = words_sorted[0][1] if len(words_sorted) > 0 else sentence[0]
            topics[i] = self.from_string_to_tensor([least_common_word])
            topics_words.append(least_common_word)

        return Variable(topics).cuda() if is_remote() else Variable(topics), topics_words
