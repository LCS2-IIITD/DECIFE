import argparse
import torch
from optim.train import train
import dataloader
import numpy as np
from dgl import random

# python3 main.py --nu 0.2 --lr 0.6 --weight-decay 0.005 --n-epochs 100
def main(args):
    if args.seed!=-1:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    checkpoints_path=f'./model_checkpoints/'

    data=dataloader.loader(args)
    train(args,data,checkpoints_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OCGNN')
    parser.add_argument("--nu", type=float, default=0.2,
            help="hyperparameter nu (must be 0 < nu <= 1)")
    parser.add_argument("--seed", type=int, default=52,
            help="random seed, -1 means dont fix seed")
    parser.add_argument("--lr", type=float, default=1e-3,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=5000,
            help="number of training epochs")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    args = parser.parse_args()

    main(args)
