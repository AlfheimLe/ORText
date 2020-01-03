import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tensorboardX import SummaryWriter
from auc import cal_auc
import os
import pickle
from sklearn import metrics
import math
from tqdm import tqdm
from torch.autograd import Variable
from scipy.optimize import minimize
import numpy as np
def encode_labels(labels, class_num):
    num = len(labels)
    preds = torch.zeros(num, class_num-1)
    for i, label in enumerate(labels):
        pred = [1]*label.item()+[0]*(class_num-1-label.item())
        pred = torch.tensor(pred, dtype=torch.float32)
        preds[i,:] = pred
    return preds

def train_FastText(config, model, train_iter, test_iter):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    model.train()
    test_auc_array = []
    train_auc_array = []
    loss_array = []
    norm_array = []
    time_array = []
    mse_array = []
    mae_array = []
    total_time = 0
    dev_best_loss = float('inf')
    last_improve = 0
    pairs = config.batch_size
    total_batch = 0
    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch+1, config.num_epochs))
        for i, (trains, labels, levels) in enumerate(train_iter):
            torch.cuda.synchronize()
            start = time.time()
            ranking, probas = model(trains)
            total_score = F.cross_entropy(ranking, labels)
            total_score.backward()
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.synchronize()
            esl_time = time.time() - start
            total_time += esl_time
            if total_batch % 100 == 0:
                # 每轮在训练集上的精度
                train_auc = 0
                _, probas = torch.max(probas, 1)
                preds_code = encode_labels(probas, config.class_num)
                train_auc = train_auc / (config.class_num - 1)
                train_auc_array.append(train_auc)

                test_auc, test_loss, mse, mae = evaluate_fasttext(config, model, test_iter)
                if test_loss < dev_best_loss:
                    dev_best_loss = test_loss
                    torch.save(model.state_dict(), config.save_path)
                    imporve = '*'
                    last_improve = total_batch
                else:
                    imporve = ''
                test_auc_array.append(test_auc)
                loss_array.append(test_loss)
                time_array.append(total_time)
                mae_array.append(mae)
                mse_array.append(mse)
                msg = 'Iter: {0:>6},  Train AUC: {1:>6.2%},  ' \
                      'Val AUC: {2:>6.2%}, Test Loss: {3:>6.2%} MSE: {4:>.6} MAE: {5:>.6}  Time: {6}, {7}'
                print(msg.format(total_batch, train_auc, test_auc, test_loss, mse, mae, total_time, imporve))
                writer.add_scalar("AUC/train", train_auc, total_batch)
                writer.add_scalar("AUC/dev", test_auc, total_batch)
                writer.add_scalar('loss/test', test_loss, total_batch)

            total_batch += 1



        # if total_batch > 25100:
        #     print('Achieve toatl iterations')
        #     break


    writer.close()
    save_folder = os.path.join('./result', config.model_name, config.dataset)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    auc_file = config.dataset + str(config.num) + '.pkl'
    # pickle.dump(auc_array, open(os.path.join(save_folder, auc_file), 'wb'))
    with open(os.path.join(save_folder, auc_file), 'wb') as f:
        pickle.dump(train_auc_array, f)
        pickle.dump(test_auc_array, f)
        pickle.dump(time_array, f)
        pickle.dump(mse_array, f)
        pickle.dump(mae_array, f)

def evaluate_fasttext(config, model, data_iter):
    model.eval()
    test_preds = []
    true_levels = []
    true_labels = []
    pred_labels = []
    loss = 0
    mse, mae, num = 0,0,0
    with torch.no_grad():
        for texts, labels, levels in data_iter:
            ranking, probas = model(texts)
            true_levels.append(levels)
            _, probas = torch.max(probas, 1)
            preds_code = encode_labels(probas, config.class_num)
            loss += F.cross_entropy(ranking, labels)
            test_preds.append(preds_code)
            true_labels.append(labels)
            pred_labels.append(probas)
            # pred_labels = torch.sum(probas>0.5, dim=1)
            num += labels.size(0)
            mse += torch.sum((probas-labels)**2)
            mae += torch.sum(torch.abs(probas-labels))
        true_levels = torch.cat(true_levels, dim=0)
        test_preds = torch.cat(test_preds, dim=0)

        true_labels = torch.cat(true_labels, dim=0)
        pred_labels = torch.cat(pred_labels, dim=0)
        for k in range(config.class_num):
            class_idx = torch.arange(true_labels.nelement())[true_labels.eq(k)]
            class_num = len(class_idx)
            acc = torch.sum(true_labels[class_idx] == pred_labels[class_idx])
            class_mse = torch.sum((true_labels[class_idx] - pred_labels[class_idx]) ** 2)
            class_mae = torch.sum(torch.abs(true_labels[class_idx] - pred_labels[class_idx]))
            print("class {0}, acc {1:>.4}, mse {2:>8}, mae {3:>8}, class num {4}".
                  format(k, acc.item() / class_num, class_mse.float(), class_mae.float(), class_num))
        auc = 0
        for k in range(config.class_num-1):
            true_label = true_levels[:, k].cpu().numpy()
            pred_label = test_preds[:, k].numpy()
            tmp = metrics.roc_auc_score(true_label, pred_label)
            auc += tmp
            print("classification {0} auc {1:>6.2%}".format(k, tmp))
        auc = auc/(config.class_num-1)
    return auc, loss.item(), mse.float()/num, mae.float()/num




"""
OR Model
"""
def or_cost_fn(logits, levels, imp):
    val = (-torch.sum((F.log_softmax(logits, dim=2)[:, :, 1] * levels +
                       F.log_softmax(logits, dim=2)[:, :, 0] * (1 - levels)) * imp, dim=1))
    return torch.mean(val)

def train_or(config, model, train_iter, test_iter,imp):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    model.train()
    test_auc_array = []
    train_auc_array = []
    loss_array = []
    norm_array = []
    time_array = []
    mae_array = []
    mse_array = []
    total_time = 0
    dev_best_loss = float('inf')
    last_improve = 0
    pairs = config.batch_size
    total_batch = 0
    flag = False
    # bar = tqdm(range(config.num_epochs))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch+1, config.num_epochs))
        for i, (trains, labels, levels) in enumerate(train_iter):
            torch.cuda.synchronize()
            start = time.time()
            ranking, probas = model(trains)
            levels = (levels+1)/2
            total_score = or_cost_fn(ranking, levels, imp)
            total_score.backward()
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.synchronize()
            esl_time = time.time() - start
            total_time += esl_time
            if total_batch % 100 == 0:
                # 每轮在训练集上的精度
                train_auc = 0

                train_auc = train_auc / (config.class_num - 1)
                train_auc_array.append(train_auc)

                test_auc, test_loss, mse, mae = evaluate_or(config, model, test_iter, imp)
                if test_loss < dev_best_loss:
                    dev_best_loss = test_loss
                    torch.save(model.state_dict(), config.save_path)
                    imporve = '*'
                    last_improve = total_batch
                else:
                    imporve = ''
                test_auc_array.append(test_auc)
                loss_array.append(test_loss)
                time_array.append(total_time)
                mse_array.append(mse)
                mae_array.append(mae)
                msg = 'Iter: {0:>6},  Train AUC: {1:>6.2%},  ' \
                      'Val AUC: {2:>6.2%}, MSE: {3:>.6} MAE: {4:>.6} ,Test Loss: {5:>6.2%} Time: {6}, {7}'
                print(msg.format(total_batch, train_auc, test_auc, mse, mae, test_loss, total_time, imporve))
                # bar.set_description(msg.format(total_batch, train_auc, test_auc, test_loss, total_time, imporve))
                writer.add_scalar("AUC/train", train_auc, total_batch)
                writer.add_scalar("AUC/dev", test_auc, total_batch)
                writer.add_scalar('loss/test', test_loss, total_batch)
                model.train()
            total_batch += 1

    # for i, (trains, labels, levels) in enumerate(train_iter):
    #     torch.cuda.synchronize()
    #     start = time.time()
    #     ranking, probas = model(trains)
    #     total_score = or_cost_fn(ranking, levels, imp)
    #     total_score.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    #
    #     torch.cuda.synchronize()
    #     esl_time = time.time() - start
    #     total_time += esl_time
    #
    #     if total_batch % 100 == 0:
    #             # 每轮在训练集上的精度
    #         train_auc = 0
    #
    #         train_auc = train_auc / (config.class_num - 1)
    #         train_auc_array.append(train_auc)
    #
    #         test_auc, test_loss = evaluate_or(config, model, test_iter, imp)
    #         if test_loss < dev_best_loss:
    #             dev_best_loss = test_loss
    #             torch.save(model.state_dict(), config.save_path)
    #             imporve = '*'
    #             last_improve = total_batch
    #         else:
    #             imporve = ''
    #         test_auc_array.append(test_auc)
    #         loss_array.append(test_loss)
    #         time_array.append(total_time)
    #
    #         msg = 'Iter: {0:>6},  Train Acc: {1:>6.2%},  ' \
    #                   'Val Acc: {2:>6.2%}, Test Loss: {3:>6.2%}  Time: {4}, {5}'
    #         print(msg.format(total_batch, train_auc, test_auc, test_loss, total_time, imporve))
    #         writer.add_scalar("acc/train", train_auc, total_batch)
    #         writer.add_scalar("acc/dev", test_auc, total_batch)
    #         writer.add_scalar('loss/test', test_loss, total_batch)
    #         model.train()
    #     total_batch += 1
    #     if total_batch > config.num_epochs:
    #         print("Achieve total iterations")
    #         break
    writer.close()
    save_folder = os.path.join('./result', config.model_name, config.dataset)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    auc_file = config.dataset + str(config.num) + '.pkl'
    # pickle.dump(auc_array, open(os.path.join(save_folder, auc_file), 'wb'))
    with open(os.path.join(save_folder, auc_file), 'wb') as f:
        pickle.dump(train_auc_array, f)
        pickle.dump(test_auc_array, f)
        pickle.dump(time_array, f)
        pickle.dump(mse_array, f)
        pickle.dump(mae_array, f)

def evaluate_or(config, model, data_iter, imp):
    model.eval()
    test_preds = []
    true_levels = []
    true_labels = []
    pred_labels = []
    loss = 0
    mse, mae, num = 0,0,0
    with torch.no_grad():
        for texts, labels, levels in data_iter:
            ranking, probas = model(texts)
            true_levels.append(levels)
            loss += or_cost_fn(ranking, levels, imp)
            test_preds.append(ranking[:,:,1])
            pred_lab = torch.sum(probas>0.5, dim=1)
            num += labels.size(0)
            mse += torch.sum((pred_lab-labels)**2)
            mae += torch.sum(torch.abs(pred_lab-labels))
            true_labels.append(labels)
            pred_labels.append(pred_lab)
        true_levels = torch.cat(true_levels, dim=0)
        test_preds = torch.cat(test_preds, dim=0)

        auc = 0

        true_labels = torch.cat(true_labels, dim=0)
        pred_labels = torch.cat(pred_labels, dim=0)
        for k in range(config.class_num):
            class_idx = torch.arange(true_labels.nelement())[true_labels.eq(k)]
            class_num = len(class_idx)
            acc = torch.sum(true_labels[class_idx] == pred_labels[class_idx])
            class_mse = torch.sum((true_labels[class_idx] - pred_labels[class_idx]) ** 2)
            class_mae = torch.sum(torch.abs(true_labels[class_idx] - pred_labels[class_idx]))
            print("class {0}, acc {1:>.4}, mse {2:>8}, mae {3:>8}, class num {4}".
                  format(k, acc.item() / class_num, class_mse.float(), class_mae.float(), class_num))

        for k in range(config.class_num-1):
            true_label = true_levels[:, k].cpu().numpy()
            pred_label = test_preds[:, k].cpu().numpy()
            tmp = metrics.roc_auc_score(true_label, pred_label)
            auc += tmp
            print("classification {0} auc {1:>6.2%}".format(k, tmp))
            # auc += cal_auc(true_levels[:, k], test_preds[:, k])
        auc = auc/(config.class_num-1)
    return auc, loss.item(), mse.float()/num, mae.float()/num



"""
CORAL Model
"""
def coral_cost_fn(logits, levels, imp):
    val = (-torch.sum((F.logsigmoid(logits)*levels +
                      (F.logsigmoid(logits)-logits)*(1-levels))*imp,
           dim=1))
    return torch.mean(val)

def train_coral(config, model, train_iter, test_iter,imp):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    model.train()
    test_auc_array = []
    train_auc_array = []
    loss_array = []
    norm_array = []
    time_array = []
    mae_array = []
    mse_array = []
    total_time = 0
    dev_best_loss = float('inf')
    last_improve = 0
    pairs = config.batch_size
    total_batch = 0
    flag = False
    loss_fn = nn.MSELoss(reduction='mean')
    # bar = tqdm(range(config.num_epochs))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch+1, config.num_epochs))
        for i, (trains, labels, levels) in enumerate(train_iter):
            torch.cuda.synchronize()
            start = time.time()
            ranking, probas = model(trains)
            levels = (levels+1)/2
            total_score = coral_cost_fn(ranking, levels, imp)

            # total_score = loss_fn(probas,levels)
            total_score.backward()
            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.synchronize()
            esl_time = time.time() - start
            total_time += esl_time
            if total_batch % 100 == 0:
                # 每轮在训练集上的精度
                train_auc = 0

                test_auc, test_loss,preds, true, MSE, MAE = evaluate_coral(config, model, test_iter, imp)
                if test_loss < dev_best_loss:
                    dev_best_loss = test_loss
                    torch.save(model.state_dict(), config.save_path)
                    imporve = '*'
                    last_improve = total_batch
                else:
                    imporve = ''
                test_auc_array.append(test_auc)
                loss_array.append(test_loss)
                time_array.append(total_time)
                mae_array.append(MAE)
                mse_array.append(MSE)
                msg = 'Iter: {0:>6},  Train AUC: {1:>6.2%},  ' \
                      'Val AUC: {2:>6.2%}, MSE: {3:>.6}, MAE: {4:>.6} Test Loss: {5:>6.2%}  Time: {6}, {7}'
                print(msg.format(total_batch, train_auc, test_auc, MSE, MAE, test_loss, total_time, imporve))

                writer.add_scalar("AUC/train", train_auc, total_batch)
                writer.add_scalar("AUC/dev", test_auc, total_batch)
                writer.add_scalar('loss/test', test_loss, total_batch)
                model.train()
            total_batch += 1
    test_auc, test_loss, preds, true , MSE, MAE = evaluate_coral(config, model, test_iter, imp)
    writer.close()
    save_folder = os.path.join('./result', config.model_name, config.dataset)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    auc_file = config.dataset + str(config.num) + '.pkl'
    # pickle.dump(auc_array, open(os.path.join(save_folder, auc_file), 'wb'))
    with open(os.path.join(save_folder, auc_file), 'wb') as f:
        pickle.dump(train_auc_array, f)
        pickle.dump(test_auc_array, f)
        pickle.dump(time_array, f)
        pickle.dump(mse_array, f)
        pickle.dump(mae_array, f)

def evaluate_coral(config, model, data_iter, imp):
    model.eval()
    test_preds = []
    true_levels = []
    test_preds_lab = []
    true_label = []
    num_example = 0
    loss = 0
    mse = 0
    mae = 0
    with torch.no_grad():
        for texts, labels, levels in data_iter:
            ranking, probas = model(texts)
            preds_labs = torch.sum(probas > 0.5, dim=1)
            true_levels.append(levels)
            loss += coral_cost_fn(ranking, levels, imp)
            test_preds.append(ranking)
            test_preds_lab.append(preds_labs)
            true_label.append(labels)
            mse += torch.sum((preds_labs-labels)**2)
            mae += torch.sum(torch.abs(preds_labs-labels))
            num_example += labels.size(0)
        true_levels = torch.cat(true_levels, dim=0)
        test_preds = torch.cat(test_preds, dim=0)
        test_preds_lab = torch.cat(test_preds_lab, dim=0)
        true_label = torch.cat(true_label, dim=0)

        for k in range(config.class_num):
            class_idx = torch.arange(true_label.nelement())[true_label.eq(k)]
            class_num = len(class_idx)
            acc = torch.sum(true_label[class_idx] == test_preds_lab[class_idx])
            class_mse = torch.sum((true_label[class_idx] - test_preds_lab[class_idx]) ** 2)
            class_mae = torch.sum(torch.abs(true_label[class_idx] - test_preds_lab[class_idx]))
            print("class {0}, acc {1:>.4}, mse {2:>8}, mae {3:>8}, class num {4}".
                  format(k, acc.item() / class_num, class_mse.float(), class_mae.float(), class_num))

        auc = 0
        for k in range(config.class_num-1):
            # auc += cal_auc(true_levels[:, k], test_preds[:, k])
            true_label = true_levels[:, k].cpu().numpy()
            pred_label = test_preds[:, k].cpu().numpy()
            tmp = metrics.roc_auc_score(true_label, pred_label)
            auc += tmp
            print("classification {0} auc {1:>6.2%}".format(k, tmp))
        auc = auc/(config.class_num-1)
    return auc, loss.item(), test_preds_lab.tolist(), true_label.tolist(), mse.float()/num_example, mae.float()/num_example



"""
Our Model
"""
def sigmod(x):
    return 1.0/(1.0+np.exp(-x))

def optimize_threshold(config, model, train_iter):
    print('optimizing threshold')
    lr = 0.0001
    bias = np.zeros(config.class_num - 1)
    # bias.requires_grad = True
    # bias = bias.cuda()
    ranking_array, levels_array = [], []
    with torch.no_grad():
        for i, (trains, labels, levels) in enumerate(train_iter):
            ranking = model(trains)
            ranking_array.append(ranking)
            levels_array.append(levels)
    ranking_array = torch.cat(ranking_array, dim=0)
    levels_array = torch.cat(levels_array, dim=0)
    # for i in range(500):
    j=0
    def loss(args):
        ranking_array, levels_array = args
        # levels_array = 0.5*(levels_array+1)
        sigma = 0.5
        beta = 1./(1-np.exp(-1./(2*sigma**2)))
        # loss = lambda bias: np.mean(
            # np.sum(np.power(1 - levels_array * (ranking_array - bias), 2), axis=1))
        loss = lambda bias: np.mean(np.sum(np.clip(1-levels_array*(ranking_array-bias), 0, a_max=1000000), axis=1))
        # loss = lambda bias: np.mean(-np.sum(0.5*(levels_array+1)*np.log(sigmod(ranking_array-bias))+(1-0.5*(levels_array+1))*np.log(1-sigmod(ranking_array-bias)), axis=1))
        return loss
    ranking_array = ranking_array.cpu().numpy()
    levels_array = levels_array.cpu().numpy()
    args = [ranking_array, levels_array]
    res = minimize(loss(args), bias, method='SLSQP')
    bias = torch.from_numpy(res.x)
    bias = bias.float()
    bias = bias.cuda()
    return bias


def train_por(config, model, train_iter, train_iter2, test_iter):

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.lam)

    writer = SummaryWriter(log_dir=config.log_path+'/'+time.strftime('%m-%d_%H.%M', time.localtime()))
    model.train()
    test_auc_array =[]
    train_auc_array = []
    loss_array = []
    norm_array = []
    mae_array = []
    mse_array = []
    time_array = []
    sigma = 0.5
    beta = 1. / (1 - math.exp(-1. / (2 * math.pow(sigma, 2))))
    total_time = 0
    dev_best_loss = float('inf')
    last_improve = 0
    pairs = config.batch_size

    for i, (trains, labels, levels) in enumerate(train_iter):
        # ranking_array, levels_array = [],[]
        torch.cuda.synchronize()
        start = time.time()
        ranking = model(trains)
        labels = labels.float()
        total_score = 0
        nlevels = 0.5*(levels+1)
        # for k in range(config.class_num-1):
        #     negative_idx = torch.arange(labels.nelement())[labels.eq(k)]
        #     negative_size = len(negative_idx)
        #     positive_idx = torch.arange(labels.nelement())[labels.gt(k)]
        #     positive_size = len(positive_idx)
        #     tmp_score = 0
        #     for idx in positive_idx:
        #         tmp_score += torch.sum(torch.clamp(1-ranking[idx, k]+ranking[negative_idx,k],0))
        #     total_score += tmp_score/(positive_size*negative_size)

            # negative_idx = torch.arange(labels.nelement())[labels.eq(k)]
            # negative_size = len(negative_idx)
            # positive_idx = torch.arange(labels.nelement())[labels.eq(k+1)]
            # positive_size = len(positive_idx)
            # tmp_score = 0
            # tmp_score += torch.sum(torch.clamp(1 - ranking[positive_idx, k] + ranking[negative_idx, k], 0))
            # total_score += tmp_score
        for k in range(config.class_num-1):
            tmp_score = torch.sum(torch.clamp(1-ranking[pairs*(k*2+1):pairs*(k*2+2),k] + ranking[pairs*k*2:pairs*(k*2+1), k], 0))
            # tmp_score = torch.sum(torch.pow(
            #     1 - ranking[pairs * (k * 2 + 1):pairs * (k * 2 + 2), k] + ranking[pairs * k * 2:pairs * (k * 2 + 1), k],
            #     2))
            # tmp_score = torch.0pv]sum(torch.pow(
            #     1 - ranking[pairs * (k * 2 + 1):pairs * (k * 2 + 2), k] + ranking[pairs * k * 2:pairs * (k * 2 + 1), k],
            #     2))
            # tmp_score = torch.sum(torch.log(1-torch.exp(-ranking[pairs*(k*2+1):pairs*(k*2+2),k] + ranking[pairs*k*2:pairs*(k*2+1), k])))
            # tmp_score = torch.sum(beta*(1-torch.exp((1-ranking[pairs*(k*2+1):pairs*(k*2+2),k] + ranking[pairs*k*2:pairs*(k*2+1), k])**2/(2*sigma**2)) ))
            total_score += tmp_score

        # total_score = total_score/(config.class_num-1)
        total_score.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        esl_time = time.time() - start
        total_time += esl_time
        # ranking_array.append(ranking)
        # levels_array.append(levels)
        if i % 100 ==0:
            #每轮在训练集上的精度
            train_auc = 0
            with torch.no_grad():
                for k in range(config.class_num - 1):
                    true_label = levels[:, k].cpu().numpy()
                    pred_label = ranking[:,k].cpu().numpy()
                    train_auc += metrics.roc_auc_score(true_label, pred_label)
            #optimize Thresholds
            # ranking_array = torch.cat(ranking_array, dim=0)
            # levels_array = torch.cat(levels_array, dim=0)
            bias = optimize_threshold(config, model, train_iter2)
            train_auc = train_auc/(config.class_num-1)
            train_auc_array.append(train_auc)
            test_auc, mse, mae = evaluate_por(config, model, test_iter, bias)
            test_auc_array.append(test_auc)
            time_array.append(total_time)
            mae_array.append(mae)
            mse_array.append(mse)
            # loss_array.append(test_loss)
            msg = 'Iter: {0:>6},  Train AUC: {1:>6.2%},  ' \
                  'Val AUC: {2:>6.2%}, MSE: {3:.6} MAE {4:.6}  Time: {5}'
            print(msg.format(i, train_auc, test_auc, mse, mae, total_time))
            writer.add_scalar("AUC/train", train_auc, i)
            writer.add_scalar("AUC/dev", test_auc, i)
            # writer.add_scalar('loss/test', test_loss, i)
            model.train()
        if i > config.num_epochs:
            print("Achieve total iterations")
            break

    writer.close()
    save_folder = os.path.join('./result', config.model_name, config.dataset)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    auc_file = config.dataset + str(config.num) + '.pkl'
    # pickle.dump(auc_array, open(os.path.join(save_folder, auc_file), 'wb'))
    with open(os.path.join(save_folder, auc_file), 'wb') as f:
        pickle.dump(train_auc_array, f)
        pickle.dump(test_auc_array, f)
        pickle.dump(time_array, f)
        pickle.dump(mse_array, f)
        pickle.dump(mae_array, f)

def evaluate_por(config, model, data_iter, bias):
    model.eval()
    test_preds = []
    true_levels = []
    true_labels = []
    pred_labels = []
    mae, mse, num_example=0,0,0
    with torch.no_grad():
        for texts, labels, levels in data_iter:
            ranking = model(texts)
            true_levels.append(levels)
            test_preds.append(ranking)
            preds = torch.sum(torch.sigmoid(ranking-bias)>0.5, dim=1)
            num_example+=labels.size(0)
            mae += torch.sum(torch.abs(preds-labels))
            mse += torch.sum((preds-labels)**2)
            true_labels.append(labels)
            pred_labels.append(preds)
        true_levels = torch.cat(true_levels, dim=0)
        test_preds = torch.cat(test_preds, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
        pred_labels = torch.cat(pred_labels, dim=0)
        for k in range(config.class_num):
            class_idx = torch.arange(true_labels.nelement())[true_labels.eq(k)]
            class_num = len(class_idx)
            acc = torch.sum(true_labels[class_idx] == pred_labels[class_idx])
            class_mse = torch.sum((true_labels[class_idx]-pred_labels[class_idx])**2)
            class_mae = torch.sum(torch.abs(true_labels[class_idx]-pred_labels[class_idx]))
            print("class {0}, acc {1:>.4}, mse {2:>8}, mae {3:>8}, class num {4}".
                  format(k, acc.item()/class_num, class_mse.float(), class_mae.float(), class_num))
        auc = 0
        for k in range(config.class_num-1):
            true_label = true_levels[:, k].cpu().numpy()
            pred_label = test_preds[:, k].cpu().numpy()
            tmp = metrics.roc_auc_score(true_label, pred_label)
            auc += tmp
            print("classification {0} auc {1:>6.2%}".format(k, tmp))
            # auc += cal_auc(true_levels[:, k], test_preds)
        auc = auc/(config.class_num-1)

    return auc, mse.float()/num_example, mae.float()/num_example

######################################
#           CNNPOR MODEL             #
######################################

def train_cnnor(config, model, train_iter, test_iter):

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    writer = SummaryWriter(log_dir=config.log_path+'/'+time.strftime('%m-%d_%H.%M', time.localtime()))
    model.train()
    test_auc_array =[]
    train_auc_array = []
    loss_array = []
    norm_array = []
    time_array = []
    total_time = 0
    dev_best_loss = float('inf')
    last_improve = 0
    pairs = config.batch_size
    for i, (trains, labels, levels) in enumerate(train_iter):
        torch.cuda.synchronize()
        start = time.time()
        ranking, probas = model(trains)
        # labels = labels.float()
        total_score = 0
        for k in range(config.class_num-1):
            negative_idx = torch.arange(labels.nelement())[labels.eq(k)]
            negative_size = len(negative_idx)
            positive_idx = torch.arange(labels.nelement())[labels.eq(k+1)]
            positive_size = len(positive_idx)
            tmp_score = 0
            for idx in positive_idx:
                tmp_score += torch.sum(torch.clamp(1-ranking[idx]+ranking[negative_idx],0))
            total_score += tmp_score
        total_score += F.cross_entropy(probas, labels)
        # total_score += F.cross_entropy(probas, labels)
        total_score.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        esl_time = time.time() - start
        total_time += esl_time

        if i % 100 ==0:
            #每轮在训练集上的精度
            train_auc = 0
            # with torch.no_grad():
            #     for k in range(config.class_num - 1):
            #         # train_auc += cal_auc(levels[:, k], ranking[:, k])
            #         true_label = levels[:, k].cpu().numpy()
            #         pred_label = ranking[:, k].cpu().numpy()
            #         train_auc += metrics.roc_auc_score(true_label, pred_label)
            # train_auc = train_auc/(config.class_num-1)
            # train_auc_array.append(train_auc)
            test_auc = evaluate_cnnor(config, model, test_iter)
            test_auc_array.append(test_auc)
            time_array.append(total_time)

            # loss_array.append(test_loss)
            msg = 'Iter: {0:>6},  Train AUC: {1:>6.2%},  ' \
                  'Val AUC: {2:>6.2%},  Time: {3}'
            print(msg.format(i, train_auc, test_auc, total_time))
            writer.add_scalar("AUC/train", train_auc, i)
            writer.add_scalar("AUC/dev", test_auc, i)
            # writer.add_scalar('loss/test', test_loss, i)
            model.train()

        if i > config.num_epochs:
            print("Achieve total iterations")
            break

    writer.close()
    save_folder = os.path.join('./result', config.model_name, config.dataset)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    auc_file = config.dataset + str(config.num) + '.pkl'
    # pickle.dump(auc_array, open(os.path.join(save_folder, auc_file), 'wb'))
    with open(os.path.join(save_folder, auc_file), 'wb') as f:
        pickle.dump(train_auc_array, f)
        pickle.dump(test_auc_array, f)
        pickle.dump(time_array, f)

def evaluate_cnnor(config, model, data_iter):
    model.eval()
    test_preds = []
    true_levels = []
    with torch.no_grad():
        for texts, labels, levels in data_iter:
            ranking, probas = model(texts)
            _, probas = torch.max(probas, 1)
            preds_code = encode_labels(probas, config.class_num)
            true_levels.append(levels)
            test_preds.append(preds_code)
        true_levels = torch.cat(true_levels, dim=0)
        test_preds = torch.cat(test_preds, dim=0)
        auc = 0
        for k in range(config.class_num-1):
            true_label = true_levels[:, k].cpu().numpy()
            pred_label = test_preds[:, k].cpu().numpy()
            auc += metrics.roc_auc_score(true_label, pred_label)
        auc = auc/(config.class_num-1)
    return auc
