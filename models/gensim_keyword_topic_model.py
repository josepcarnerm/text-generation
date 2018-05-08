import string
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import is_remote, zeros
from nltk.corpus import wordnet as wn
from models.word_rnn_topic_provided import Model as WordRNNModelTopic
from gensim.summarization import keywords

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
        aggr_topics_words = []
        for i in range(batch_size):
            sentence = [batch[j][i] for j in range(len(batch))]
            # import pdb; pdb.set_trace()
            try:
                topic_words = self.get_eval_keywords(sentence)
            except:
            	continue
                # import pdb; pdb.set_trace()
            try:
            	topics[i] = self.from_string_to_tensor(topic_words)
            except:
            	continue
                # import pdb; pdb.set_trace()
            
            aggr_topics_words += topic_words

        return Variable(topics).cuda() if is_remote() else Variable(topics), aggr_topics_words

    def get_eval_keywords(self, sentence, n_words=5, scores=False):
        '''
        Gets a list of n_words keywords (and scores) from a given evaluation sentence
        \n:param sentence: input sentence, list or string
        \n:param n_words: number of keywords to identify
        \n:param scores: keyword scores
        \n:return: list of (str, float) representing (keyword, score)
        '''
        if isinstance(sentence, list):
            sentence = ' '.join(sentence)
        # print("Sentence trying to be keyworded: " + sentence)
        return keywords(sentence, words=n_words, scores=scores, split=True)

    # def evaluate(self, batch):

    #     loss_reconstruction = 0
    #     loss_topic = 0

    #     self.copy_weights_encoder()
    #     topics, topics_words = self.select_topics(batch)
    #     inp, target = self.get_input_and_target(batch)

    #     # Topic is provided as an initialization to the hidden state
    #     if self.opt.bidir:
    #         topic_enc = torch.cat([self.encoder(topics) for _ in range(self.opt.n_layers_rnn*2)], 1) \
    #             .contiguous().permute(1, 0, 2)  # N_layers x 1 x N_hidden
    #     else:
    #         topic_enc = torch.cat([self.encoder(topics) for _ in range(self.opt.n_layers_rnn)], 1) \
    #                          .contiguous().permute(1, 0, 2)  # N_layers x 1 x N_hidden

    #     hidden = topic_enc, topic_enc.clone()

    #     # Encode/Decode sentence
    #     loss_topic_total_weight = 0
    #     last_output = inp[:, 0]  # Only used if "reuse_pred" is set
    #     running_sentence = []
    #     for w in range(self.opt.sentence_len):

    #         x = last_output if self.opt.reuse_pred else inp[:, w]
    #         output, hidden = self.forward(x, hidden)

    #         # Reconstruction Loss
    #         loss_reconstruction += self.criterion(output.view(self.opt.batch_size, -1), target[:, w])

    #         _, words_i = output.max(1)
    #         last_output = words_i
    #         import pdb; pdb.set_trace()
    #         running_sentence += [self.inverted_word_dict[i.data[0]] for i in words_i]
    #         # Topic closeness loss
    #         if (w == self.opt.sentence_len - 1):

    #             loss_topic += self.closeness_to_topics(output, topics) 

    #     loss_topic = torch.mean(loss_topic/loss_topic_total_weight)

    #     self.losses_reconstruction.append(loss_reconstruction.data[0])
    #     self.losses_topic.append(loss_topic.data[0])

    #     if (self.opt.ETL):
    #         return self.opt.loss_alpha*loss_reconstruction + (1-self.opt.loss_alpha)*loss_topic
    #     else:
    #         return self.opt.loss_alpha*loss_reconstruction