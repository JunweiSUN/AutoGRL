from torch_geometric.utils import is_undirected
from pandas import DataFrame
import torch

__all__ = ['print_statistics']

def print_statistics(data):
    print('Original feature size:', data.x.shape[1])
    remove_unique_feature(data)
    print('Current feature size:', data.x.shape[1])
    edge_direction = 'Undirected' if is_undirected(data.edge_index) else 'Directed'
    print('{} graph'.format(edge_direction))
    print('{} graph'.format('Weighted' if is_weighted(data) else 'Unweighted'))
    print('Number of nodes: ', data.x.shape[0])
    num_edges = data.edge_index.shape[1]
    if edge_direction == 'Undirected':
        num_edges = num_edges // 2
    print('Number of edges: ', num_edges)
    print('Number of classes: {}'.format(int(max(data.y)) + 1))
    if hasattr(data, 'train_mask'):
        print('Number of training nodes:', data.train_mask.sum().item())
    if hasattr(data, 'val_mask'):
        print('Number of validation nodes:', data.val_mask.sum().item())
    if hasattr(data, 'test_mask'):
        print('Number of test nodes:', data.test_mask.sum().item())

def is_weighted(data):
    if hasattr(data, 'edge_weight'):
        if data.edge_weight.max() != data.edge_weight.min():
            return True
    return False

def remove_unique_feature(data):
    df = DataFrame(data.x.numpy())
    unique_counts = df.nunique()
    unique_counts = unique_counts[unique_counts == 1]
    df.drop(unique_counts.index, axis=1, inplace=True)
    print('Drop {} features'.format(len(unique_counts)))
    data.x = torch.tensor(df.to_numpy(), dtype=torch.float)
