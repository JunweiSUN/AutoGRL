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

dataset = Planetoid(".", "citeseer", transform=T.NormalizeFeatures(), split='random')
data = dataset[0]
# print_statistics(data)
from augmentation import label_propagation
# label_propagation(data)


class Net(torch.nn.Module):
    def __init__(self, dataset):
        super(Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.act = torch.nn.PReLU(num_parameters=args.hidden)

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=args.dropout, training=self.training)
        # x = self.act(self.lin1(x))
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)

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

