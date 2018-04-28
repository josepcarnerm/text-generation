import string, pdb

import nltk
nltk.download('wordnet')

import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import is_remote, zeros
from nltk.corpus import wordnet as wn
from models.word_rnn import Model as WordRNNModel


class Model(WordRNNModel):

    def __init__(self, opt):
        super(Model, self).__init__(opt)

        self.load_word_counts()
        self.encoder_topic = nn.Embedding(self.N_WORDS, self.opt.hidden_size_rnn)
        if self.opt.use_pretrained_embeddings:
            embeddings = torch.zeros((len(self.word2idx)), self.word_dict_dim).float()
            for k, v in self.word2idx.items():
                embeddings[v] = self.word_dict[k]
            self.encoder_topic.weight = nn.Parameter(embeddings)
            self.encoder_topic.weight.requires_grad = False
            del self.word_dict  # Clear the memory
        for p in self.encoder_topic.parameters(): p.requires_grad = False  # Freeze weights

        self.submodules = self.submodules + [self.encoder_topic]

        self.losses_reconstruction = []
        self.losses_topic = []

    def initialize(self, baseline_model):
        # baseline_model must be path to "checkpoint" file
        baseline = torch.load(baseline_model).get('model')
        self.baseline = baseline
        self.word2idx = baseline.word2idx
        self.encoder.load_state_dict(baseline.encoder.state_dict())
        self.rnn.load_state_dict(baseline.rnn.state_dict())
        self.decoder.load_state_dict(baseline.decoder.state_dict())
        self.encoder_topic.load_state_dict(baseline.encoder.state_dict())  # Initialize encoder_topic with encoder

    def load_word_counts(self):
        if self.opt.use_pretrained_embeddings:
            self.word_count = torch.load(self.opt.data_dir + self.opt.input_file + '.sentences.g_word_count')
        else:
            self.word_count = torch.load(self.opt.data_dir + self.opt.input_file + '.sentences.word_count')

    def closeness_to_topics(self, words_weights, topics):

        _, words = words_weights.max(1)

        dist = nn.PairwiseDistance(p=2)

        closeness = []

        for i in range(len(topics)):
            topic_str = self.inverted_word_dict[topics[i].data[0]]
            topic = topics[i]
            word = words[i]
            synonyms = [topic.data[0]]
            for syn in wn.synsets(topic_str):
                synonyms += [self.word2idx[l.name()] for l in syn.lemmas() if l.name() in self.word2idx]

            synonyms = torch.from_numpy(numpy.array(synonyms))
            synonyms = Variable(synonyms).cuda() if is_remote() else Variable(synonyms)

            closeness.append(torch.mean(torch.stack([dist(self.encoder(syn), self.encoder(word)) for syn in synonyms])))

        return torch.stack(closeness)

    def analyze_topics(self, batch):
        pass

    def analyze_closeness_to_topics(self, batch):

        batch_size, sentence_len = len(batch[0]), len(batch) - 1

        # Analyze closeness to topics
        print('Analyzing predictions......')
        examples = []

        self.copy_weights_encoder()
        topics, _ = self.select_topics(batch)
        if len(batch) ==2:
            batch = batch[1]
        inp, target = self.get_input_and_target(batch)
        # Topic is provided as an initialization to the hidden state
        if self.opt.bidirectional:
            hidden = torch.cat([self.encoder(topics) for _ in range(self.opt.n_layers_rnn*2)], 1).permute(1, 0, 2), \
                     torch.cat([self.encoder(topics) for _ in range(self.opt.n_layers_rnn*2)], 1).permute(1, 0, 2)  # N_layers x batch_size x N_hidden
        else:
            hidden = torch.cat([self.encoder(topics) for _ in range(self.opt.n_layers_rnn)], 1).permute(1, 0, 2), \
                     torch.cat([self.encoder(topics) for _ in range(self.opt.n_layers_rnn)], 1).permute(1, 0, 2)  # N_layers x batch_size x N_hidden

        for i in range(len(batch[0])):
            sentence = [batch[j][i] for j in range(len(batch))]
            examples.append(
                {'sentence': ' '.join(sentence), 'topic': self.inverted_word_dict[topics[i].data[0]], 'preds and dist': []})

        # Encode/Decode sentence
        for w in range(sentence_len):
            output, hidden = self.forward(inp[:, w], hidden)

            # Topic closeness loss: Weight each word contribution by the inverse of it's frequency
            # _, words_i = output.max(1)
            # Sample from the network as a multinomial distribution
            output_dist = output.div(0.8).exp()
            words_i = torch.multinomial(output_dist, 1)

            loss_topic_weights = Variable(torch.from_numpy(numpy.array(
                [1 / self.word_count[self.inverted_word_dict[i.data[0]]] for i in words_i]
            )).unsqueeze(1)).float()
            loss_topic_weights = loss_topic_weights.cuda() if is_remote() else loss_topic_weights
            closeness = self.closeness_to_topics(output, topics)

            for i in range(len(batch[0])):
                examples[i]['preds and dist'].append({
                    'predicted word': self.inverted_word_dict[words_i[i].data[0]],
                    'word weight in topic loss': loss_topic_weights[i].data[0],
                    'closeness to sentence topic': closeness[i].data[0]
                })
        for e in examples:
            try:
                print('Sentence: {}. Topic: {}. Predictions, weights and closeness: {}.'.format(
                    ' '.join(e['sentence']), e['topic'], '\n\t' + '\n\t'.join([str(x) for x in e['preds and dist']])
                ))
            except:
                print('Exception when printing')

    def analyze(self, batch):

        self.analyze_topics(batch)
        self.analyze_closeness_to_topics(batch)

        # Analyze losses
        print('Analyzing Losses......')
        print('Last topic loss: {}. Last reconstruction loss: {}'.format(
            self.losses_topic[-1]/self.opt.sentence_len, self.losses_reconstruction[-1]/self.opt.sentence_len
        ))

    def select_topics(self, batch):
        try:
            batch_size = len(batch[1][0])
        except:
            import pdb; pdb.set_trace()
        topics_words = batch[0]
        topics = self.from_string_to_tensor(topics_words).view(batch_size, 1)
        return Variable(topics).cuda() if is_remote() else Variable(topics), topics_words

    def copy_weights_encoder(self):
        # Copy weights in encoder to to encoder_topic so that both embedding layers are the same but the latter
        # is not influenced by the topic loss
        if not self.opt.use_pretrained_embeddings:
            self.encoder_topic.load_state_dict(self.encoder.state_dict())

    def evaluate(self, batch):

        loss_reconstruction = 0
        loss_topic = 0

        self.copy_weights_encoder()
        topics, topics_words = self.select_topics(batch)
        inp, target = self.get_input_and_target(batch)

        # Topic is provided as an initialization to the hidden state
        if self.opt.bidirectional:
            topic_enc = torch.cat([self.encoder(topics) for _ in range(self.opt.n_layers_rnn*2)], 1) \
                .contiguous().permute(1, 0, 2)  # N_layers x 1 x N_hidden
        else:
            topic_enc = torch.cat([self.encoder(topics) for _ in range(self.opt.n_layers_rnn)], 1) \
                             .contiguous().permute(1, 0, 2)  # N_layers x 1 x N_hidden

        hidden = topic_enc, topic_enc.clone()

        # Encode/Decode sentence
        loss_topic_total_weight = 0
        last_output = inp[:, 0]  # Only used if "reuse_pred" is set
        for w in range(self.opt.sentence_len):

            x = last_output if self.opt.reuse_pred else inp[:, w]
            output, hidden = self.forward(x, hidden)

            # Reconstruction Loss
            loss_reconstruction += self.criterion(output.view(self.opt.batch_size, -1), target[:, w])

            # Topic closeness loss: Weight each word contribution by the inverse of it's frequency
            _, words_i = output.max(1)
            last_output = words_i
            loss_topic_weights = Variable(torch.from_numpy(numpy.array(
                [1/self.word_count[self.inverted_word_dict[i.data[0]]] for i in words_i]
            )).unsqueeze(1)).float()
            loss_topic_weights = loss_topic_weights.cuda() if is_remote() else loss_topic_weights
            loss_topic_total_weight += loss_topic_weights
            loss_topic += self.closeness_to_topics(output, topics) * loss_topic_weights

        loss_topic = torch.mean(loss_topic / loss_topic_total_weight)

        self.losses_reconstruction.append(loss_reconstruction.data[0])
        self.losses_topic.append(loss_topic.data[0])

        ratio = float(loss_reconstruction.detach().cpu().data.numpy()[0] / loss_topic.detach().cpu().data.numpy()[0])
        loss = self.opt.loss_alpha*loss_reconstruction + (1-self.opt.loss_alpha)*loss_topic*ratio
        return loss, loss_reconstruction, loss_topic

    def get_test_topic(self):
        return self.select_topics((['love'], [['love']]))

    def test_word_rnn(self, prime_words, predict_len, temperature=0.8):

        hidden = self.baseline.init_hidden(1)
        prime_input = Variable(self.baseline.from_string_to_tensor(prime_words).unsqueeze(0))

        if is_remote():
            prime_input = prime_input.cuda()
        predicted = ' '.join(prime_words)

        # Use priming string to "build up" hidden state
        for p in range(len(prime_words) - 1):
            _, hidden = self.baseline.forward(prime_input[:, p], hidden)

        inp = prime_input[:, -1]

        for p in range(predict_len):
            output, hidden = self.baseline.forward(inp, hidden)

            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            # Add predicted character to string and use as next input
            predicted_word = self.baseline.from_predicted_index_to_string(top_i)
            predicted += ' '+predicted_word
            inp = Variable(self.baseline.from_string_to_tensor([predicted_word]).unsqueeze(0))
            if is_remote():
                inp = inp.cuda()

        return predicted

    def test(self, prime_words, predict_len, temperature=0.8):

        self.copy_weights_encoder()
        topic, _ = self.get_test_topic()
        if self.opt.bidirectional:
            topic_enc = torch.cat([self.encoder(topic) for _ in range(self.opt.n_layers_rnn*2)], 1) \
                .contiguous().permute(1, 0, 2)  # N_layers x 1 x N_hidden
        else:
            topic_enc = torch.cat([self.encoder(topic) for _ in range(self.opt.n_layers_rnn)], 1) \
                             .contiguous().permute(1, 0, 2)  # N_layers x 1 x N_hidden
        hidden = topic_enc, topic_enc.clone()
        prime_input = Variable(self.from_string_to_tensor(prime_words).unsqueeze(0))

        if is_remote():
            prime_input = prime_input.cuda()
        predicted = ' '.join(prime_words)

        # Use priming string to "build up" hidden state
        for p in range(len(prime_words) - 1):
            _, hidden = self.forward(prime_input[:, p], hidden)

        inp = prime_input[:, -1]

        for p in range(predict_len):
            output, hidden = self.forward(inp, hidden)

            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0]

            # Add predicted character to string and use as next input
            predicted_word = self.from_predicted_index_to_string(top_i)
            predicted += ' '+predicted_word
            inp = Variable(self.from_string_to_tensor([predicted_word]).unsqueeze(0))
            if is_remote():
                inp = inp.cuda()

        import pdb; pdb.set_trace()
        return predicted

    def parameters(self):
        params = (p for p in super(Model, self).parameters() if p.requires_grad)
        return params