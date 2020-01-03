import time
import torch
import numpy as np
from get_args import process_args
from model import OR, Config, init_model
import os
from process_data import build_dataset, build_iterator, get_time_dif, build_test_iterator, buil_random_iterator
from train import train_coral, train_or

def main():
    args = process_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    if args.seed == -1:
        RANDOMSEED = None
    else:
        RANDOMSEED = args.seed
    torch.manual_seed(RANDOMSEED)
    torch.cuda.manual_seed(RANDOMSEED)
    IMP_WEIGHT = args.imp_weight
    if not IMP_WEIGHT:
        imp = torch.ones(args.class_num - 1, dtype=torch.float)
    elif IMP_WEIGHT == 1:
        pass
    else:
        raise ValueError('Incorrect importance weight parameter.')
    imp = imp.cuda()
    embedding = 'random'
    torch.backends.cudnn.deterministic = True

    config = Config(args, embedding, 'OR')
    start_time = time.time()
    print("Loading data...")
    vocab, train_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config, doubly_flag=False)
    # train_iter = buil_random_iterator(train_data,config)
    test_iter = build_test_iterator(test_data, config)

    print("Time used: ", get_time_dif(start_time))

    config.n_vocab = len(vocab)
    model = OR(config).cuda()
    init_model(model)
    print("start training...")
    train_or(config, model, train_iter, test_iter, imp)

if __name__ == '__main__':
    main()