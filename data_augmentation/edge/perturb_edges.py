from .models import GAE, VGAE
from .utils import train_model, sample_graph_det
from .dataloader import DataLoader
import torch
import argparse
import numpy as np
from torch_geometric.utils import from_scipy_sparse_matrix, train_test_split_edges ,
from torch_geometric.nn import GCNConv, GAE
import pickle
import os.path as osp
import traceback
from copy import deepcopy

__all__ = ['perturb_edges']

CWD = osp.dirname(osp.abspath(__file__))
ROOT = CWD + '/../..'

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

# def perturb_edges(data, name, remove_pct, add_pct):
#     if remove_pct == 0 and add_pct == 0:
#         return
#     try:
#         cached = pickle.load(open(f'{ROOT}/cache/edge/{name}_{remove_pct}_{add_pct}.pt', 'rb'))
#         print(f'Use cached edge augmentation for dataset {name}')
#         data.edge_index = cached
#         return
#     except FileNotFoundError:
#         try:
#             A_pred, adj_orig = pickle.load(open(f'{ROOT}/cache/edge/{name}.pt', 'rb'))
#             A = sample_graph_det(adj_orig, A_pred, remove_pct, add_pct)
#             data.edge_index, _ = from_scipy_sparse_matrix(A)
#             pickle.dump(data.edge_index, open(f'{ROOT}/cache/edge/{name}_{remove_pct}_{add_pct}.pt', 'wb'))
#             return
#         except FileNotFoundError:
#             print(f'cache/edge/{name}_{remove_pct}_{add_pct}.pt not found! Regenerating it now')
    
#     args = get_args()
#     print(args)
#     args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     dl = DataLoader(args, data)

#     model = GAE(dl.adj_norm.to(args.device), dl.features.size(1), args.hidden_size, args.emb_size)
#     model.to(args.device)
#     model = train_model(args, dl, model)

#     features = dl.features.to(args.device)
#     with torch.no_grad():
#         A_pred = model(features)
#     A_pred = torch.sigmoid(A_pred).detach().cpu().numpy()
#     np.fill_diagonal(A_pred, 0)

#     A = sample_graph_det(dl.adj_orig, A_pred, remove_pct, add_pct)
#     data.edge_index, _ = from_scipy_sparse_matrix(A)
#     pickle.dump((A_pred, dl.adj_orig) ,open(f'{ROOT}/cache/edge/{name}.pt', 'wb'))
#     pickle.dump(data.edge_index, open(f'{ROOT}/cache/edge/{name}_{remove_pct}_{add_pct}.pt', 'wb'))

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

def perturb_edges(data, name, remove_pct, add_pct, hidden_channels=16, epochs=400):
    if remove_pct == 0 and add_pct == 0:
        return
    try:
        cached = pickle.load(open(f'{ROOT}/cache/edge/{name}_{remove_pct}_{add_pct}.pt', 'rb'))
        print(f'Use cached edge augmentation for dataset {name}')
        data.edge_index = cached
        return
    except FileNotFoundError:
        try:
            A_pred, adj_orig = pickle.load(open(f'{ROOT}/cache/edge/{name}.pt', 'rb'))
            A = sample_graph_det(adj_orig, A_pred, remove_pct, add_pct)
            data.edge_index, _ = from_scipy_sparse_matrix(A)
            pickle.dump(data.edge_index, open(f'{ROOT}/cache/edge/{name}_{remove_pct}_{add_pct}.pt', 'wb'))
            return
        except FileNotFoundError:
            print(f'cache/edge/{name}_{remove_pct}_{add_pct}.pt not found! Regenerating it now')


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = deepcopy(data.edge_index)
    data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0)
    num_features = data.ori_x.shape[1]
    model = GAE(GCNEncoder(num_features, hidden_channels))
    model = model.to(device)
    x = data.ori_x.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_val_auc = 0
    best_z = None
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(x, train_pos_edge_index)
        loss = model.recon_loss(z, train_pos_edge_index)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index)

        auc, ap = model.test(z, data.val_pos_edge_index, data.val_neg_edge_index)
        print('Val | Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
        if auc > best_val_auc:
            best_val_auc = auc
            best_z = deepcopy(z)
    
    A_pred = torch.sigmoid(torch.mm(z, z.T)).cpu().numpy()

    adj_orig = to_scipy_sparse_matrix(edge_index).asformat('csr')
    adj_pred = sample_graph_det(adj_orig, A_pred, remove_pct, add_pct)
    data.edge_index, _ = from_scipy_sparse_matrix(adj_pred)

    pickle.dump((A_pred, adj_orig) ,open(f'{ROOT}/cache/edge/{name}.pt', 'wb'))
    pickle.dump(data.edge_index, open(f'{ROOT}/cache/edge/{name}_{remove_pct}_{add_pct}.pt', 'wb'))