from data_augmentation.edge import add_and_remove_edges
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

dataset = Planetoid('data', 'cora', transform=T.NormalizeFeatures())
data = dataset[0]

add_and_remove_edges(data, 1, 1)