# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import pickle
from tqdm import tqdm
import random
from divide_data import parse, getDF, build_vocab
import re
import numpy as np

UNK, PAD = '<UNK>', '<PAD>'
MAX_VOCAB_SIZE = 100000


def label_transform(labels, class_num):
    num = len(labels)
    levels = torch.zeros(num, class_num - 1)
    for i, label in enumerate(labels):
        level = [1] * label.item() + [0] * (class_num - 1 - label.item())
        level = torch.LongTensor(level)
        levels[i, :] = level
    return levels.cuda()

def label_transform2(labels, class_num):
    num = len(labels)
    levels = torch.zeros(num, class_num - 1)
    for i, label in enumerate(labels):
        level = [1] * label.item() + [-1] * (class_num - 1 - label.item())
        level = torch.LongTensor(level)
        levels[i, :] = level
    return levels.cuda()

def build_dataset(config, use_word):
    with open(config.train_path, 'rb') as f:
        # class_num = pickle.load(f)
        train_input = pickle.load(f)
        train_labels = pickle.load(f)
        # train_class_num = pickle.load(f)
        test_input = pickle.load(f)
        test_labels = pickle.load(f)
        # test_class_num = pickle.load(f)

    if use_word:
        tokernizer = lambda x: x.split(" ")
    else:
        tokernizer = lambda x: [y for y in x]
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.dataset, tokernizer=tokernizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    def biGramHash(sequence, t, buckets):
        t1 = sequence[t-1] if t-1 > 0 else 0
        return (t1*14918087)%buckets

    def triGramHash(sequence, t, buckets):
        t1 = sequence[t-1] if t-1>=0 else 0
        t2 = sequence[t-2] if t-2>=0 else 0
        return (t2*14918087*18408749+ t1*14918087)%buckets
    
    def load_dataset(data, labels, pad_size=32):
        contents = []
        for label, line in tqdm(zip(labels, data)):
            line = re.sub('[,.<>/?\'\":;{}()\[\]~`!@#$%^&*\-_+=]', ' ', line)
            line =line.lower()
            lin = line.strip()
            if not lin:
                continue
            content = lin
            words_line = []
            token = tokernizer(content)
            seq_len = len(token)
            if pad_size:
                if len(token) < pad_size:
                    token.extend([vocab.get(PAD)] * (pad_size - len(token)))
                else:
                    token = token[:pad_size]
                    seq_len = pad_size
            for word in token:
                words_line.append(vocab.get(word, vocab.get(UNK)))
            buckets = config.n_gram_vocab
            bigram = []
            trigram = []

            for i in range(pad_size):
                bigram.append(biGramHash(words_line, i, buckets))
                trigram.append(triGramHash(words_line, i, buckets))
            contents.append((words_line, int(label), seq_len, bigram, trigram))
        return contents
    train = load_dataset(train_input, train_labels, config.pad_size)
    test = load_dataset(test_input, test_labels, config.pad_size)
    return vocab, train, test
            
class DatasetIterater(object):
    def __init__(self, batches, batch_size, class_num):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.class_num = class_num
        self.index_list = torch.randperm(len(batches)).tolist()

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).cuda()
        y = torch.LongTensor([_[1] for _ in datas]).cuda()
        bigram = torch.LongTensor([_[3] for _ in datas]).cuda()
        trigram = torch.LongTensor([_[4] for _ in datas]).cuda()

        seq_len = torch.LongTensor([_[2] for _ in datas]).cuda()
        levels = label_transform2(y, self.class_num)
        return (x, seq_len, bigram, trigram), y, levels

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batch_idx = self.index_list[self.index * self.batch_size:len(self.batches)]
            self.index += 1
            batches = [self.batches[idx] for idx in batch_idx]
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            self.index_list = torch.randperm(len(self.batches)).tolist()
            raise StopIteration
        else:
            batch_idx = self.index_list[self.index * self.batch_size: (self.index+1)*self.batch_size]
            self.index += 1
            batches = [self.batches[idx] for idx in batch_idx]
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches+1
        else:
            return self.n_batches


class DoublyDatasetIterater(object):
    def __init__(self, datas, batch_size, class_num):
        self.batch_size = batch_size
        self.datas = datas
        self.n_batches = 0
        self.class_num = class_num

        #获取每个类别数据的index
        labels = [_[1] for _ in datas]
        positive_idx = []
        negative_idx = []
        for i in range(class_num-1):
            positive_idx.append([k for k, val in enumerate(labels) if val > i])
            negative_idx.append([k for k, val in enumerate(labels) if val <= i])
        self.positive_idx = positive_idx
        self.negative_idx = negative_idx
    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).cuda()
        y = torch.LongTensor([_[1] for _ in datas]).cuda()
        bigram = torch.LongTensor([_[3] for _ in datas]).cuda()
        trigram = torch.LongTensor([_[4] for _ in datas]).cuda()

        seq_len = torch.LongTensor([_[2] for _ in datas]).cuda()
        levels = label_transform2(y, self.class_num)
        return (x, seq_len, bigram, trigram), y, levels

    def __next__(self):
        rand_idx = []
        for i in range(self.class_num-1):
            tmps = random.sample(self.negative_idx[i], self.batch_size)
            rand_idx += tmps
            tmps = random.sample(self.positive_idx[i], self.batch_size)
            rand_idx +=tmps
        batches = [self.datas[idx] for idx in rand_idx]
        batches = self._to_tensor(batches)
        self.n_batches += 1
        return batches

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches


class RandomDatasetIterater(object):
    def __init__(self, datas, batch_size, class_num):
        self.batch_size = batch_size
        self.datas = datas
        self.n_batches = 0
        self.index = list(range(len(datas)))
        self.class_num = class_num

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).cuda()
        y = torch.LongTensor([_[1] for _ in datas]).cuda()
        bigram = torch.LongTensor([_[3] for _ in datas]).cuda()
        trigram = torch.LongTensor([_[4] for _ in datas]).cuda()

        seq_len = torch.LongTensor([_[2] for _ in datas]).cuda()
        levels = label_transform(y, self.class_num)
        return (x, seq_len, bigram, trigram), y, levels

    def __next__(self):
        random_idx = random.sample(self.index, self.batch_size)
        batches = [self.datas[idx] for idx in random_idx]
        batches = self._to_tensor(batches)
        self.n_batches += 1
        return batches
    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches


def build_iterator(dataset, config, doubly_flag):
    if doubly_flag:
        iters = DoublyDatasetIterater(dataset, config.batch_size, config.class_num)
    else:
        iters = DatasetIterater(dataset, config.batch_size, config.class_num)
    return iters

def buil_random_iterator(dataset, config):
    return RandomDatasetIterater(dataset, config.batch_size, config.class_num)

def build_test_iterator(dataset, config, batch=256):

    iters = DatasetIterater(dataset, batch, config.class_num)
    return iters

def get_time_dif(start_time):
    end_time = time.time()
    time_diff = end_time-start_time
    return timedelta(seconds=int(round(time_diff)))



class ClassDatasetIterater(object):
    def __init__(self, datas, batch_size, class_num):
        self.batch_size = batch_size
        self.datas = datas
        self.n_batches = 0
        self.class_num = class_num

        #获取每个类别数据的index
        labels = [_[1] for _ in datas]
        class_idx = []
        for i in range(class_num):
            class_idx.append([k for k, val in enumerate(labels) if val== i])
        self.class_idx = class_idx
    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).cuda()
        y = torch.LongTensor([_[1] for _ in datas]).cuda()
        bigram = torch.LongTensor([_[3] for _ in datas]).cuda()
        trigram = torch.LongTensor([_[4] for _ in datas]).cuda()

        seq_len = torch.LongTensor([_[2] for _ in datas]).cuda()
        levels = label_transform(y, self.class_num)
        return (x, seq_len, bigram, trigram), y, levels

    def __next__(self):
        rand_idx = []
        for i in range(self.class_num):
            tmps = random.sample(self.class_idx[i], self.batch_size)
            rand_idx += tmps
        batches = [self.datas[idx] for idx in rand_idx]
        batches = self._to_tensor(batches)
        self.n_batches += 1
        return batches

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches

def build_class_iterator(dataset, config):
    iters = ClassDatasetIterater(dataset, config.batch_size, config.class_num)
    return iters


