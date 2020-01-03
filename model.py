# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os




def init_model(model, method = 'xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

class Config(object):
    def __init__(self, args, embedding, algorithm_name):
        self.dataset = args.name
        self.model_name = algorithm_name
        self.train_path = os.path.join('./divide', self.dataset, str(args.num), self.dataset+'.pkl')
        self.class_list = []
        self.vocab_path = os.path.join('./divide', self.dataset, 'vocab.pkl')
        self.save_path = os.path.join('./divide', self.dataset, str(args.num), self.model_name+'.ckpt')
        self.log_path = os.path.join('./divide', self.dataset, str(args.num), 'log', self.model_name)
        self.embedding_pretrained = torch.tensor(
            np.load(os.path.join('./divide/', self.dataset, str(args.num), embedding))["embedding"].astype("float32")) \
            if embedding != 'random' else None
        # self.device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
        self.lam = args.lam
        self.dropout = 0.5                                      #随机失活
        self.num_classes = args.class_num#len(self.class_list)               #类别说
        self.n_vocab = 0                                        #词表大小
        self.num_epochs =args.epochs                            #总的迭代次数
        self.pad_size = 128                                     #超过pad_size截取
        self.batch_size = args.batch
        self.learning_rate = args.lr
        self.class_num = args.class_num
        self.num = args.num
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300    #字向量维度
        self.hidden_size = 256                                  #隐藏层
        self.n_gram_vocab = 250499                              #ngram 词表大小

class OR(nn.Module):
    def __init__(self, config):
        super(OR, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embdding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_gram_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.embedding_ngram2 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.embedding_ngram3 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed*3, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, (config.class_num-1)*2)
        self.class_num = config.class_num
        # self.linear_1_bias = nn.Parameter(torch.zeros(config.num_classes-1))

    def forward(self, x):
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        logits = out.view(-1, (self.class_num - 1), 2)
        probas = F.softmax(logits, dim=2)[:, :, 1]
        return logits, probas


class POR(nn.Module):
    def __init__(self, config):
        super(POR, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embdding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_gram_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.embedding_ngram2 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.embedding_ngram3 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed*3, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, 1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(config.num_classes-1))

    def forward(self, x):
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)+self.linear_1_bias
        # logits = out + self.linear_1_bias
        # probas = torch.sigmoid(logits)
        return out

class CORAL(nn.Module):
    def __init__(self, config):
        super(CORAL, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embdding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_gram_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.embedding_ngram2 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.embedding_ngram3 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed*3, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, 1, bias=False)

        self.linear_1_bias = nn.Parameter(torch.zeros(config.num_classes-1))

    def forward(self, x):
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        logits = out + self.linear_1_bias
        probas = torch.sigmoid(logits)
        return logits, probas

class FastText(nn.Module):
    def __init__(self, config):
        super(FastText, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embdding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_gram_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.embedding_ngram2 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.embedding_ngram3 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed * 3, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.class_num)

    def forward(self, x):
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        probas = torch.softmax(out, dim=1)
        return out, probas

class CNNPOR(nn.Module):
    def __init__(self, config):
        super(CNNPOR, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embdding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_gram_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.embedding_ngram2 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.embedding_ngram3 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed * 3, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.class_num)
        self.fc3 = nn.Linear(config.hidden_size, 1, bias=False)
        self.linear_1_bias = nn.Parameter(torch.zeros(config.num_classes-1))

    def forward(self, x):
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        probas = self.fc2(out)
        out = self.fc3(out)
        logits = out + self.linear_1_bias
        return logits, probas