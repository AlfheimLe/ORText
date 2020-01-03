import argparse

def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str)
    parser.add_argument('--cuda', default='1', help='Which cuda device to use')
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num', type=int, default=0)
    parser.add_argument('--lam', type=float, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sigma', type=float, default=0.5)
    parser.add_argument('--word', default=True, type=bool)
    parser.add_argument('--class_num', default=5, type=int)
    parser.add_argument('--imp_weight', type=int, default=0)
    args = parser.parse_args()
    return args
