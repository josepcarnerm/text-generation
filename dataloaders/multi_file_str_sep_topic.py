import numpy
import torch, unidecode, random, os
from torch.autograd import Variable
from collections import Counter
from dataloaders.single_file_str_sentences import MyDataset as SentenceDataset

from utils import ALL_CHARS, is_remote, glove2dict

class MyDataset(SentenceDataset):
	NON_DIGITS = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n\t\x0b\x0c"
	END_SENTENCE = "!?."

	def __init__(self, opt, train):
		super(MyDataset, self).__init__(opt, train)

		self.opt = opt
		self.topic_folder_path = opt.topic_folder_path
		self.prob_repeat_example = 0.25
		self.n_items = int(1/self.prob_repeat_example)
		self.topic_sentences = []
		self.load_topic_files()


	def load_topic_files(self):
		for filename in os.listdir(self.topic_folder_path):
			file = open(self.topic_folder_path + filename, "r",encoding='utf-8', errors='ignore').read()
			self.words = str(file)
			self.topic_sentences = []
			for c in self.NON_DIGITS:
				self.words = self.words.replace(c, ' ' + c + ' ')

			for c in self.END_SENTENCE:
				self.words = self.words.replace(c, self.END_SENTENCE[0])
			self.topic_sentences += self.words.split(self.END_SENTENCE[0])

			self.topic_sentences = [
				[word.lower() for word in sentence.split(' ') if word.strip() != '']
				for sentence in self.topic_sentences
			]

			# When using pretrained glove vectors, only pick topic_sentences whose words (all of them) have its corresponding glove vector
			if self.opt.use_pretrained_embeddings:
				self.topic_sentences = [
					sentence for sentence in self.topic_sentences if all(self.glv_dict.get(word) is not None for word in sentence)
			]

			numpy.random.shuffle(self.topic_sentences)
			n_train = int(len(self.topic_sentences)*0.75)
			self.topic_sentences_all = {'train': self.topic_sentences[:n_train], 'test': self.topic_sentences[n_train:]}
			self.topic_len = len(self.topic_sentences)

			# torch.save(self.sentences_all, sentences_file)

	def __getitem__(self, index):
		random.seed(index)
		j = random.randint(0, (self.len - 1))
		i = random.randint(0, (self.topic_len - 1))
		while len(self.topic_sentences[i]) <= self.opt.sentence_len / 2:
			i = random.randint(0, (self.topic_len - 1))

		while len(self.sentences[j]) <= self.opt.sentence_len:
			j = random.randint(0, (self.len - 1))

		return self.topic_sentences[i], self.sentences[j]

	# def __len__(self):
	# 	return self.len