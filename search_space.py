from copy import deepcopy
import torch
from itertools import product
from feature_engineering import get_embedding
from data_augmentation.edge import perturb_edges
from data_augmentation.label import label_propagation
from data_augmentation.node import dropout_x, dropout_nodes
import pickle
import shap
import pandas as pd
import numpy as np

data_aug = {
    "add-edges": [0, 10, 20, 30],
    "remove-edges": [0, 10, 20, 30],
    "dropout-x": [0, 0.1, 0.4, 0.7],
    "dropout-nodes": [0, 0.1, 0.4, 0.7],
    "label_propogation": [0, 1]
}

fe = {
    "embedding-type": ['spectral', 'node2vec', 'onehot', 'svd', 'xavier', 'none']
}

hpo = {
    "lr": [0.1, 0.005, 0.001, 0.0005],
    "epochs": [100, 200, 400, 800],
    "dropout": [0, 0.1, 0.4, 0.7],
    "norm-type": ['bn', 'ln', 'in', 'none'],
    "act-type": ['tanh', 'relu', 'leaky-relu', 'prelu', 'elu', 'identity'],
    "hidden_size": [16, 32, 64, 128, 256],
    "act-first": [0, 1]
}

nas = {
    "conv-type": ['gat-1', 'gat-2', 'gat-4', 'gat-8', 'gcn', 'sage', 'cheb', 'tag', 'arma', 'gin'],
    "aggr-type": ['add', 'mean', 'max'],
    "layer-aggr-type": ['plain', 'res', 'jk', 'dense'],
    "num-layers": [1, 2, 3, 4, 5, 6, 7, 8]
}

def pruning_search_space_by_eda(data):
    '''
    pruning the search space according to the data's EDA info
    and return a full search space
    '''
    _data_aug, _fe, _hpo, _nas = deepcopy(data_aug), deepcopy(fe), deepcopy(hpo), deepcopy(nas)
    task = data.task
    setting = data.setting
    if task == 'sup':
        _data_aug['label-propogation'].remove(1) # supervised task doesn't need LP
    if data.x is None:
        _fe['embedding_type'].remove('none') # dataset with no feature must have a hand-crafted feature
    else:
        _fe['embedding-type'].remove('onehot')
        _fe['embedding-type'].remove('xavier') # we do not use one hot and xavier feature on dataset with feature
                                               # since they do not provide any extra infomation

    ss = {} # search space
    ss.update(_data_aug)
    ss.update(_fe)
    ss.update(_hpo)
    ss.update(_nas)

    return ss, (_data_aug, _fe, _hpo, _nas)

def pruning_search_space_by_shap(archs, model, search_space, p):
    '''
    pruning the search space according to the data's shap info
    return a pruned search space
    '''

    ss = deepcopy(search_space)
    explainer = shap.TreeExplainer(model)
    X = [[str(e) for e in row] for row in archs]
    shap_values = explainer.shap_values(pd.DataFrame(X, columns=ss.keys()))

    stat = {}

    for i, row in enumerate(shap_values):
        for j, feat_name in enumerate(ss.keys()):
            if len(ss[feat_name]) <= 1:             # cannot prune this feature any more
                continue

            feat_value = feat_name + '|' + str(archs[i][j])
            shap_value = shap_values[i][j]
            if feat_value not in stat:
                stat[feat_value] = [shap_value]
            else:
                stat[feat_value].append(shap_value)
            
    for k, v in stat.items():
        stat[k] = np.mean(v).item()
    
    stat_list = sorted(stat.items(), key=lambda e: e[1], reverse=False) # get top p worse feature

    i = 0
    count = 0
    while count < p:
        if i >= len(stat_list):
            break

        record = stat_list[i]
        feat_name, feat_value = record[0].split('|')
        shap_value = record[1]
        try: # is a int value?
            feat_value = int(feat_value)
        except ValueError:
            try: # is a float value?
                feat_value = float(feat_value)
            except ValueError:
                pass # is a str value

        i += 1
        try:
            ss[feat_name].remove(feat_value)
        except ValueError:
            continue

        count += 1
        print(f'pruning feature: {feat_name} | value: {feat_value} | shap value: {shap_value}')

    return ss

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
        if data.x is None:                             # use data.ori_x to compute A_pred
            assert data.setting == 'transductive'
            data.ori_x = deepcopy(embedding)
            data.x = deepcopy(embedding)
        else:
            if data.setting == 'inductive':
                data.ori_x = deepcopy(data.train_x)
                if embedding is not None:
                    data.train_x = torch.cat([data.train_x, embedding], dim=1)
            else:
                data.ori_x = deepcopy(data.x)
                if embedding is not None:
                    data.x = torch.cat([data.x, embedding], dim=1)

        perturb_edges(data, name, remove_edge_pct, add_edge_pct) # edge augmentation, using the original feature

        dropout_x(data, name, dropout_x_pct) # node augmentation
        dropout_nodes(data, name, dropout_nodes_pct)

        data.ori_x = None

        pickle.dump(data, open(f'cache/data/{combine_str}.pt', 'wb')) # save cache data
