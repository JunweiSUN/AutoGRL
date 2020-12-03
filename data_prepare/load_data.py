from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, PPI, Amazon, WikiCS
from torch_geometric.utils import from_networkx, subgraph
from ogb.nodeproppred import PygNodePropPredDataset
import os.path as osp
import networkx as nx
import numpy as np
import torch
import pickle
import random

__all__ = ['load_data']

def load_data(name, transform=None):
    '''
    Load data from files and return a pytorch geometric `Data` object
    '''
    ROOT = osp.dirname(osp.abspath(__file__)) + '/..'
    if name in ['cora', 'citeseer', 'pubmed']:   # datasets for transductive node classifiction
        data = Planetoid(osp.join(ROOT, 'data'), name, transform=transform)[0]
        data.task = 'semi' # simi-supervised
        data.setting = 'transductive' # transductive
        return data
    
    elif name in ['wikics']:
        dataset = WikiCS(osp.join(ROOT, 'data', 'wikics'), transform=transform)
        data = dataset[0]
        data.task = 'semi'
        data.setting = 'transductive'
        data.train_mask = data.train_mask[:,0]
        data.val_mask = data.val_mask[:, 0]
        data.stopping_mask = data.stopping_mask[:, 0]
        return data

    elif name in ['ppi']: # datasets for inductive node classification
        train_dataset = PPI(osp.join(ROOT, 'data', 'ppi'), split='train', transform=transform)
        val_dataset = PPI(osp.join(ROOT, 'data', 'ppi'), split='val', transform=transform)
        test_dataset = PPI(osp.join(ROOT, 'data', 'ppi'), split='test', transform=transform)
        return (train_dataset, val_dataset, test_dataset)
    elif name in ['usa-airports']:
        try:
            data = pickle.load(open(osp.join(ROOT, 'data', name, 'data.pkl'), 'rb'))
        except FileNotFoundError:
            print('Data not found. Re-generating...')
        nx_graph = nx.read_edgelist(osp.join(ROOT, 'data', name, 'edges.txt'))
        nx_graph = nx.convert_node_labels_to_integers(nx_graph, label_attribute='id2oid') # oid for original id
        oid2id = {int(v):k for k,v in nx.get_node_attributes(nx_graph, 'id2oid').items()}
        id2label = {}
        for line in open(osp.join(ROOT, 'data', name, 'labels.txt')):
            linesplit = line.strip().split()
            oid = int(linesplit[0])
            label = int(linesplit[1])
            id2label[oid2id[oid]] = {'y': label} # here we assume that the label id start from 0 and the labeling is consistant.
        nx.set_node_attributes(nx_graph, id2label)

        data = from_networkx(nx_graph)
        num_nodes = len(nx_graph.nodes)
        node_idxs = list(range(num_nodes))
        random.shuffle(node_idxs)
        # split data, train:val:test = 80%:10%:10%
        train_idxs = node_idxs[:int(0.8 * num_nodes)]
        val_idxs = node_idxs[int(0.8 * num_nodes):int(0.9 * num_nodes)]
        test_idxs = node_idxs[int(0.9 * num_nodes):]

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[train_idxs] = True
        data.val_mask[val_idxs] = True
        data.test_mask[test_idxs] = True
        if data.x and transform:
            data.x = transform(data.x)
        data.num_nodes = num_nodes
        data.task = 'sup' # simi-supervised
        data.setting = 'transductive' # transductive
        pickle.dump(data, open(osp.join(ROOT, 'data', name, 'data.pkl'), 'wb'))
        return data

    elif name in ['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(name, root=osp.join(ROOT, 'data'), transform=transform)
        split_idx = dataset.get_idx_split()
        data = dataset[0]
        split_idx['val'] = split_idx.pop('valid')
        for key, idx in split_idx.items():
            mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            mask[idx] = True
            data[f'{key}_mask'] = mask
        data.task = 'sup' # simi-supervised
        data.setting = 'transductive' # transductive
        return data

    elif name in ['photo']:
        dataset = Amazon('data/photo', 'photo', transform=transform)
        data = dataset[0]
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[:-1000] = True
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask[-1000: -500] = True
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask[-500:] = True

        data.train_edge_index, _ = subgraph(data.train_mask, data.edge_index, relabel_nodes=True)
        data.val_edge_index, _ = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)
        data.val_edge_index, _ = subgraph(data.val_mask, data.edge_index, relabel_nodes=True)
        data.train_x = data.x[data.train_mask]
        data.train_y = data.y[data.train_mask]
        data.val_x = data.x[data.val_mask]
        data.val_y = data.y[data.val_mask]
        data.test_x = data.x[data.test_mask]
        data.test_y = data.y[data.test_mask]

        data.num_train_nodes = data.train_x.shape[0]
        data.task = 'sup' # simi-supervised
        data.setting = 'inductive' # transductive
        return data

    else:
        raise NotImplementedError('Not supported dataset.')


load_data('wikics')