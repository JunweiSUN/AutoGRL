import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

import os.path as osp
import argparse
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from data_prepare import load_data
from feature_engineering import get_embedding


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cora')
args = parser.parse_args()

data = load_data(args.dataset, 0, transform=T.NormalizeFeatures())
if data.x is None:
    data.x = get_embedding(args.dataset, data, 'onehot')

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(data.x.shape[1], 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, int(max(data.y)) + 1, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    if args.dataset == 'photo':
        F.nll_loss(model(data.train_x, data.train_edge_index), data.train_y).backward()
    else:
        F.nll_loss(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


@torch.no_grad()
def test():
    model.eval()

    if args.dataset == 'photo':
        train_pred = model(data.train_x, data.train_edge_index).max(1)[1]
        train_acc = train_pred.eq(data.train_y).sum().item() / data.train_x.shape[0]
        val_pred = model(data.val_x, data.val_edge_index).max(1)[1]
        val_acc = val_pred.eq(data.val_y).sum().item() / data.val_x.shape[0]
        test_pred = model(data.test_x, data.test_edge_index).max(1)[1]
        test_acc = test_pred.eq(data.test_y).sum().item() / data.test_x.shape[0]
        accs = [train_acc, val_acc, test_acc]
    else:
        logits, accs = model(data.x, data.edge_index), []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
    return accs

all_test_accs = []
for run in range(10):
    best_val_acc = test_acc = 0
    for epoch in range(1, 500):
        train()
        train_acc, val_acc, tmp_test_acc = test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc = tmp_test_acc
        log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_acc, best_val_acc, test_acc))
    all_test_accs.append(test_acc)

print(np.mean(all_test_accs))
print(np.std(all_test_accs))
