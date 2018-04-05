import string

import nltk
nltk.download('wordnet')

import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import is_remote, zeros
from nltk.corpus import wordnet as wn
from models.word_rnn_topic_least_frequent_word import Model as WordRNNTopicLeastFrequentWordModel


class Model(WordRNNTopicLeastFrequentWordModel):

    def __init__(self, opt):
        super(Model, self).__init__(opt)

    def analyze_topics(self, batch):
        examples = []

        # Select "topic" as the closest word, in the embedded space, to the centroid of the sentence.
        for i in range(len(batch[0])):
            sentence = [batch[j][i] for j in range(len(batch))]
            sentence_var = Variable(self.from_string_to_tensor(sentence))
            sentence_var = sentence_var.cuda() if is_remote() else sentence_var
            sentence_emb = self.encoder(sentence_var)
            centroid = torch.mean(sentence_emb, 0)
            distances = torch.sum((sentence_emb - centroid) ** 2, 1)
            closest_word_to_centroid = sentence[distances.min(0)[1].data[0]]
            distances_to_centroid = {sentence[i]: distances[i].data[0] for i in range(len(sentence))}
            examples.append({'sentence': ' '.join(sentence), 'closest_word_to_centroid': closest_word_to_centroid,
                             'distances_to_centroid': distances_to_centroid})
        for e in examples:
            print('Sentence: {}. Closest word to centroid: {}. Distances to centroid: {}.'.format(
                e['sentence'], e['closest_word_to_centroid'], e['distances_to_centroid'])
            )

    def select_topics(self, batch):

        # batch is weirdly ordered due to Pytorch Dataset class from which we inherit in each dataloader : sizes are sentence_len x batch_size
        batch_size = len(batch[0])

        # Select "topic" as the closest word, in the embedded space, to the centroid of the sentence.
        topics = torch.LongTensor(batch_size, 1)
        topics_words = []
        for i in range(batch_size):
            sentence = [batch[j][i] for j in range(len(batch))]
            sentence_var = Variable(self.from_string_to_tensor(sentence))
            sentence_var = sentence_var.cuda() if is_remote() else sentence_var
            sentence_emb = self.encoder(sentence_var)
            centroid = torch.mean(sentence_emb, 0)
            distances = torch.sum((sentence_emb-centroid)**2, 1)
            closest_word_to_centroid = sentence[distances.min(0)[1].data[0]]
            topics[i] = self.from_string_to_tensor([closest_word_to_centroid])
            topics_words.append(closest_word_to_centroid)

        return Variable(topics).cuda() if is_remote() else Variable(topics), topics_words
