import torch
import pickle
import os.path as osp

CWD = osp.dirname(osp.abspath(__file__))
ROOT = CWD + '/../..'

def dropout_x(data, name, p):
    if p > 0:
        try:
            cached = pickle.load(open(f'{ROOT}/cache/node/{name}_{p}_x.pt', 'rb'))
            print(f'Use cached x augmentation with p={p} for dataset {name}')
            data.x = cached
            return
        except FileNotFoundError:
            print(f'cache/node/{name}_{p}_x.pt not found! Regenerating it now')
        data.x = torch.nn.functional.dropout(data.x, p)
        pickle.dump(data.x, open(f'{ROOT}/cache/node/{name}_{p}_x.pt', 'wb'))

def dropout_nodes(data, name, p):
    if p > 0:
        try:
            cached = pickle.load(open(f'{ROOT}/cache/node/{name}_{p}_n.pt', 'rb'))
            print(f'Use cached node augmentation with p={p} for dataset {name}')
            data.x = cached
            return
        except FileNotFoundError:
            print(f'cache/node/{name}_{p}_n.pt not found! Regenerating it now')
        num_nodes = data.x.shape[0]
        mask = torch.full((num_nodes, 1), 1-p)
        mask = torch.bernoulli(mask).expand(data.x.shape)
        data.x = data.x.mul(mask) / (1-p)
        pickle.dump(data.x, open(f'{ROOT}/cache/node/{name}_{p}_n.pt', 'wb'))
