from .models import GAE, VGAE
from .utils import train_model, sample_graph_det
from .dataloader import DataLoader
import torch
import argparse
import numpy as np
from torch_geometric.utils import from_scipy_sparse_matrix

__all__ = ['perturb_edges']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_size', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--val_frac', type=float, default=0.9)
    parser.add_argument('--test_frac', type=float, default=0.1)
    parser.add_argument('--criterion', type=str, default='roc')
    parser.add_argument('--no_mask', type=bool, default=True)
    args = parser.parse_args()
    return args

def perturb_edges(data, remove_pct, add_pct):
    args = get_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl = DataLoader(args, data)

    model = GAE(dl.adj_norm.to(args.device), dl.features.size(1), args.hidden_size, args.emb_size)
    model.to(args.device)
    model = train_model(args, dl, model)

    features = dl.features.to(args.device)
    with torch.no_grad():
        A_pred = model(features)
    A_pred = torch.sigmoid(A_pred).detach().cpu().numpy()
    np.fill_diagonal(A_pred, 0)

    A = sample_graph_det(dl.adj_orig, A_pred, remove_pct, add_pct)
    data.edge_index, _ = from_scipy_sparse_matrix(A)
