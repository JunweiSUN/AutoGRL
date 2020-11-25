from .models import GAE, VGAE
from .utils import train_model
from .dataloader import DataLoader
import torch
import argparse

__all__ = ['add_and_remove_edges']

def get_args():
    parser = argparse.ArgumentParser(description='VGAE')
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--emb_size', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--gen_graphs', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--val_frac', type=float, default=0.05)
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--criterion', type=str, default='roc')
    parser.add_argument('--no_mask', action='store_true')
    parser.add_argument('--gae', action='store_true')
    # # tmp args for debuging
    parser.add_argument("--w_r", type=float, default=1)
    parser.add_argument("--w_kl", type=float, default=1)
    args = parser.parse_args()
    return args

def add_and_remove_edges(data, i, j):
    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl = DataLoader(args, data)

    model = GAE(dl.adj_norm.to(args.device), dl.features.size(1), args.hidden_size, args.emb_size)
    model.to(args.device)
    model = train_model(args, dl, model)
