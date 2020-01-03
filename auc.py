import torch

def cal_auc(true_label, prob):
    f = list(zip(prob, true_label))
    rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
    rankList = [i+1 for i in range(len(rank)) if rank[i]==1]
    posNum = 0
    negNum = 0
    for i in range(len(true_label)):
        if(true_label[i]==1):
            posNum += 1
        else:
            negNum += 1
    auc = (sum(rankList)-(posNum*(posNum+1))/2)/(posNum*negNum)
    return auc

if __name__ == '__main__':
    true = torch.tensor([1, 0, 1, 1, 0, 1])
    prob = torch.rand(6)
    auc = cal_auc(true, prob)
    print(auc)