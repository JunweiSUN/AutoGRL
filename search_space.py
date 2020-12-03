from copy import deepcopy
import torch
from itertools import product
from feature_engineering import get_embedding
from data_augmentation.edge import perturb_edges
from data_augmentation.label import label_propagation
from data_augmentation.node import dropout_x, dropout_nodes
import pickle

data_aug = {
    "add_edges": [0, 10, 20, 30],
    "remove_edges": [0, 10, 20, 30],
    "dropout_x": [0, 0.1, 0.4, 0.7],
    "dropout_nodes": [0, 0.1, 0.4, 0.7],
    "label_propogation": [0, 1]
}

fe = {
    "embedding_type": ['spectral', 'node2vec', 'onehot', 'svd', 'xavier', 'none']
}

hpo = {
    "lr": [0.1, 0.005, 0.001, 0.0005],
    "epochs": [100, 200, 400, 800],
    "dropout": [0, 0.1, 0.4, 0.7],
    "norm_type": ['bn', 'ln', 'in', 'none'],
    "act_type": ['tanh', 'relu', 'leaky_relu', 'prelu', 'elu', 'identity'],
    "hidden_size": [16, 32, 64, 128, 256],
    "act_first": [0, 1]
}

nas = {
    "conv_type": [],
    "aggr_type": ['add', 'mean', 'max'],
    "layer_aggr_type": ['plain', 'res', 'jk', 'dense'],
    "num_layers": [1, 2, 3, 4, 5, 6, 7, 8]
}

def pruning_search_space(data):
    '''
    pruning the search space according to the data's EDA info
    and return a full search space
    '''
    _data_aug, _fe, _hpo, _nas = deepcopy(data_aug), deepcopy(fe), deepcopy(hpo), deepcopy(nas)
    task = data.task
    setting = data.setting
    if task == 'sup':
        _data_aug['label_propogation'].remove(1) # supervised task doesn't need LP
    if data.x is None:
        _fe['embedding_type'].remove('none') # dataset with no feature must have a hand-crafted feature
    else:
        _fe['embedding_type'].remove('onehot')
        _fe['embedding_type'].remove('xavier') # we do not use one hot and xavier feature on dataset with feature
                                               # since they do not provide any extra infomation
    if setting == 'inductive': # no extra embedding for inductive task since the graph structure is not complete
        _fe['embedding_type'] = ['none']

    ss = {} # search space
    ss.update(_data_aug)
    ss.update(_fe)
    ss.update(_hpo)
    ss.update(_nas)

    return ss, (_data_aug, _fe, _hpo, _nas)

def precompute_cache(name, raw_data, data_aug, fe):
    ss = {}
    ss.update(data_aug)
    ss.update(fe)
    for sample in product(*ss.values()):
        combine_str = '_'.join([name] + [str(e) for e in sample])
        data = deepcopy(raw_data)

        add_edge_pct = sample[0]
        remove_edge_pct = sample[1]
        dropout_x_pct = sample[2]
        dropout_nodes_pct = sample[3]
        use_lp = sample[4]
        fe_type = sample[5]

        if use_lp:  # label augmentation
            label_propagation(data, name)

        embedding = get_embedding(name, data, fe_type) # feature engineering
        if data.x is None:                # use data.ori_x to compute A_pred
            data.ori_x = deepcopy(embedding)
            data.x = deepcopy(embedding)
        else:
            data.ori_x = deepcopy(data.x)
            if embedding is not None:
                data.x = torch.cat([data.x, embedding], dim=1)

        perturb_edges(data, name, remove_edge_pct, add_edge_pct) # edge augmentation, using the original feature

        dropout_x(data, name, dropout_x_pct) # node augmentation
        dropout_nodes(data, name, dropout_nodes_pct)

        data.ori_x = None

        pickle.dump(data, open(f'cache/data/{combine_str}.pt', 'wb')) # save cache data



