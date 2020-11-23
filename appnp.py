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

dataset = Planetoid(".", "citeseer", split='public', transform=T.NormalizeFeatures())
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
        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, data):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)

model = Net(dataset)
data = data.to('cuda')
model = model.to('cuda')

from augmentation import add_and_remove_edges
add_and_remove_edges(data, 1, 1)
exit(0)

trainer = EarlyStoppingTrainer(args, model)
evaluator = Evaluator(metric='acc')
trainer.train(data)
logits = trainer.inference(data)
y_pred = logits[data.test_mask].max(1)[1]
y_true = data.y.squeeze()[data.test_mask]
print(evaluator(y_true, y_pred))

