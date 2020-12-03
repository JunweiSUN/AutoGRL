import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon, WikiCS
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa
from nas.models import GNNModel
from data_prepare import load_data
from torch_geometric.utils import is_undirected
from search_space import pruning_search_space

data = load_data('')
ss, _ = pruning_search_space(data)
print(ss)

