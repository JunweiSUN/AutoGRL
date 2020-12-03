from data_augmentation.node import dropout_nodes, dropout_x
from data_augmentation.edge import perturb_edges
from data_augmentation.label import label_propagation
from feature_engineering import node2vec, onehot, svd, xavier, spectral
from search_space import data_augmentation, feature_engineering
from data_prepare import load_data
from itertools import product
from torch_geometric.transforms import NormalizeFeatures
from copy import deepcopy
import torch
import pickle



for name in ['cora', 'usa-airports', 'ogbn-arxiv']:
    raw_data = load_data(name, transform=NormalizeFeatures())
    if raw_data.x is None:
        no_feature = True
    else:
        no_feature = False

    choices = []
    for k, v in data_augmentation.items():
        choices.append(v)
    for k, v in feature_engineering.items():
        choices.append(v)
    
    all_possible_samples = list(product(*choices))

    for sample in all_possible_samples:
        combine_str = '_'.join([name] + [str(e) for e in sample])
        print(combine_str)
        if no_feature and 'none' in combine_str: # datasets with no features must have an FE method
            continue
        
        data = deepcopy(raw_data)

        add_edge_pct = sample[0]
        remove_edge_pct = sample[1]
        dropout_x_pct = sample[2]
        dropout_nodes_pct = sample[3]
        use_lp = sample[4]
        fe_type = sample[5]

        if not no_feature:
            perturb_edges(data, name, remove_edge_pct, add_edge_pct)

        if fe_type == 'svd':
            embedding = svd(data, name)
        if fe_type == 'node2vec':
            embedding = node2vec(data, name)
        if fe_type == 'onehot':
            embedding = onehot(data, name)
        if fe_type == 'xavier':
            embedding = xavier(data, name)
        if fe_type == 'spectral':
            embedding = spectral(data, name)
        concat_embedding(data, embedding)

        if no_feature:
            perturb_edges(data, name, remove_edge_pct, add_edge_pct)

        

        if use_lp:
            label_propagation(data, name)

        

for name in ['ppi']:
    raw_data = load_data(name, transform=None)
    no_feature = False
    choices = []
    for k, v in data_augmentation.items():
        choices.append(v)
    for k, v in feature_engineering.items():
        choices.append(v)
    
    all_possible_samples = list(product(*choices))

    for sample in all_possible_samples:
        combine_str = '_'.join([name] + [str(e) for e in sample])
        print(combine_str)
        if no_feature and 'none' in combine_str: # datasets with no features must have an FE method
            continue

        pickle.dump(raw_data, open(f'cache/data/{combine_str}.pt', 'wb'))
