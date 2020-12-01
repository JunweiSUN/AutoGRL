import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv
from data_augmentation.node import dropout_nodes, dropout_x
from data_augmentation.label import label_propagation

dataset = Planetoid('data', 'pubmed', transform=T.NormalizeFeatures())
data = dataset[0]
label_propagation(data)
# dropout_nodes(data, 0.5)
# dropout_x(data, 0.5)
import random
import numpy as np
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(2)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=8, concat=False,
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
    F.nll_loss(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()

def train_flag():
    y = data.y.squeeze()[data.train_mask]
    forward = lambda perturb : model(data.x+perturb, data.edge_index)[data.train_mask]
    model_forward = (model, forward)

    flag(model_forward, data.x.shape, y, F.nll_loss)

def flag(model_forward, perturb_shape, y, criterion) :
    m = 3
    step_size = 1e-3
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    perturb = torch.FloatTensor(*perturb_shape).uniform_(-step_size, step_size).to(device)
    perturb.requires_grad_()
    out = forward(perturb)
    loss = criterion(out, y)
    loss /= m

    for _ in range(m-1):
        loss.backward()
        perturb_data = perturb.detach() + step_size * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        out = forward(perturb)
        loss = criterion(out, y)
        loss /= m

    loss.backward()
    optimizer.step()

def test():
    model.eval()
    logits, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

best_val_acc = 0
best_test_acc = 0
for epoch in range(1, 201):    
    # train()
    train_flag()
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    accs = test()
    train_acc, val_acc, test_acc = accs
    # print(log.format(epoch, *accs))
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
print(best_val_acc, best_test_acc)
