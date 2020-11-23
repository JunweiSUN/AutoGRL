from networkx.algorithms.node_classification import local_and_global_consistency
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx, is_undirected
import torch_geometric.transforms as T
from utils.collections import print_statistics, remove_unique_feature
from augmentation import label_propagation, graph_perturbation
import networkx as nx
import torch
dataset = Planetoid(".", "citeseer", transform=T.NormalizeFeatures())
data = dataset[0]
print_statistics(data)

train_idx = torch.where(data.train_mask==True)[0].numpy()
train_label = data.y[data.train_mask].numpy()
test_idx = torch.where(data.test_mask==True)[0].numpy()
test_label = data.y[data.test_mask]

nx_data = to_networkx(data, to_undirected=True, node_attrs=['y'])
train_label_dict = {}
for i, node_id in enumerate(train_idx):
    train_label_dict[node_id] = train_label[i]
nx.set_node_attributes(nx_data, train_label_dict, 'label')

graph_perturbation(data, centrality_type='pagerank')
