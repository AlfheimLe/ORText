# coding: UTF-8
import numpy as np
import pickle
import os
from random import shuffle
import pandas as pd
import gzip
from tqdm import tqdm
import pickle as pkl
import re
from get_args import process_args
from sklearn.model_selection import train_test_split

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

UNK, PAD = '<UNK>', '<PAD>'
MAX_VOCAB_SIZE = 100000

def build_vocab(data_name, tokernizer, max_size, min_freq):
    data_path = os.path.join('./Source/', data_name + ".gz")
    df = getDF(data_path)
    doc = df.reviewText
    vocab_dic = {}
    for idx in tqdm(doc.index):
        line = doc[idx]
        line = re.sub('[,.<>/?\'\":;{}()\[\]~`!@#$%^&*\-_+=]', ' ', line)
        line = line.lower()
        lin = line.strip()
        if not lin:
            continue
        content = lin.split('\t')[0]
        for word in tokernizer(content):
            vocab_dic[word] = vocab_dic.get(word, 0)+1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1]>=min_freq], key=lambda x:x[1], reverse=True)[:max_size]
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic)+1})
    return vocab_dic

def divide_data(data_name, ratio=0.8, class_num=5):
    data_path = os.path.join('./Source/', data_name+".gz")
    df = getDF(data_path)
    doc = df.reviewText
    label = df.overall
    character = [[] for i in range(class_num)]
    data_size = len(label)
    for idx in label.index:
        character[int(label[idx])-1].append(doc[idx])
    del doc, label
    X, y = [], []
    for i, data in enumerate(character):
        shuffle(data)
        for x in data:
            X.append(x)
            y.append(i)
    for k in range(10):
        print(k)
        train_input, test_input, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=k)

        # prior = np.array(train_class_num)/sum(train_class_num)
        divide_data_path = os.path.join('./divide/', data_name, str(k))
        if not os.path.exists(divide_data_path):
            os.makedirs(divide_data_path)
        with open(os.path.join(divide_data_path, data_name+'.pkl'), 'wb') as f:
            # pickle.dump(class_num, f)
            pickle.dump(train_input, f)
            pickle.dump(train_labels, f)
            # pickle.dump(train_class_num, f)
            pickle.dump(test_input, f)
            pickle.dump(test_labels, f)
            # pickle.dump(test_class_num, f)

if __name__ == '__main__':
    args = process_args()
    data_name = args.name
    vocab_path = os.path.join('./divide', data_name, 'vocab.pkl')
    if not os.path.exists(vocab_path):
        tokernizer = lambda x: x.split(" ")
        vocab = build_vocab(data_name, tokernizer=tokernizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        if not os.path.exists(os.path.join('./divide', data_name)):
            os.makedirs(os.path.join('./divide', data_name))
        pkl.dump(vocab, open(vocab_path, 'wb'))
    divide_data(data_name, ratio=0.8, class_num=5)

