import argparse
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import APPNP, GATConv, GCNConv
from evaluator import Evaluator
from trainer import EarlyStoppingTrainer, SWAEnsembleTrainer, SnapshotEnsembleTrainer, PlateauTrainer
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from utils.collections import print_statistics
import numpy as np
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(2)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--alpha', type=float, default=0.1)
args = parser.parse_args()

dataset = Planetoid(".", "cora", split='full', transform=T.NormalizeFeatures())
data = dataset[0]
# print_statistics(data)
from augmentation import label_propagation, graph_perturbation
# label_propagation(data)
# graph_perturbation(data, centrality_type='eigenvector', data_type='edge')
from torch_geometric.utils import dropout_adj
data.edge_index = dropout_adj(data.edge_index, p=0.5)[0]


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 128, cached=True)
        self.conv2 = GCNConv(128, dataset.num_classes, cached=True)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)

from augmentation import add_and_remove_edges
# add_and_remove_edges(data, 20, 20)
model = Net(dataset)
data = data.to('cuda')
model = model.to('cuda')

trainer = EarlyStoppingTrainer(args, model)
evaluator = Evaluator(metric='acc')
trainer.train(data)
logits = trainer.inference(data)
y_pred = logits[data.test_mask].max(1)[1]
y_true = data.y.squeeze()[data.test_mask]
print(evaluator(y_true, y_pred))
