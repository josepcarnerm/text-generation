import string
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import is_remote, zeros
from nltk.corpus import wordnet as wn
from models.word_rnn_topic_provided import Model as WordRNNTopic


class Model(WordRNNTopic):

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
            # centroid = torch.mean(sentence_emb, 0)
            # distances = torch.sum((sentence_emb-centroid)**2, 1)
            # import pdb; pdb.set_trace()
            # closest_word_to_centroid = sentence[distances.min(0)[1].data[0]]
            topics[i] = sentence_emb #self.from_string_to_tensor([closest_word_to_centroid])
            # topics_words.append(closest_word_to_centroid)
#        import pdb; pdb.set_trace()
        return Variable(topics).cuda() if is_remote() else Variable(topics) #, topics_words

    def evaluate(self, batch):

        loss_reconstruction = 0
        loss_topic = 0

        self.copy_weights_encoder()
        import pdb; pdb.set_trace()
        topics, topics_words = self.select_topics(batch)
        inp, target = self.get_input_and_target(batch)
        # Topic is provided as an initialization to the hidden state
        topic_enc = torch.cat([self.encoder(topics) for _ in range(self.opt.n_layers_rnn)], 1)
        # import pdb; pdb.set_trace()
        topic_enc = topic_enc.permute(1, 0, 2)
                           # N_layers x batch_size x N_hidden
        hidden = topic_enc, topic_enc.clone()

        # Encode/Decode sentence
        loss_topic_total_weight = 0
        last_output = inp[:, 0]  # Only used if "reuse_pred" is set
        for w in range(self.opt.sentence_len):

            x = last_output if self.opt.reuse_pred else inp[:, w]
            output, hidden = self.forward(x, hidden)

            # Reconstruction Loss
            loss_reconstruction += self.criterion(output, target[:, w])

            # Topic closeness loss: Weight each word contribution by the inverse of it's frequency
            _, words_i = output.max(1)
            loss_topic_weights = Variable(torch.from_numpy(numpy.array(
                [1/self.word_count[self.inverted_word_dict[i.data[0]]] for i in words_i]
            )).unsqueeze(1)).float()
            loss_topic_weights = loss_topic_weights.cuda() if is_remote() else loss_topic_weights
            loss_topic_total_weight += loss_topic_weights
            loss_topic += self.closeness_to_topics(output, topics) * loss_topic_weights

        loss_topic = torch.mean(loss_topic/loss_topic_total_weight)

        self.losses_reconstruction.append(loss_reconstruction.data[0])
        self.losses_topic.append(loss_topic.data[0])

        return self.opt.loss_alpha*loss_reconstruction + (1-self.opt.loss_alpha)*loss_topic