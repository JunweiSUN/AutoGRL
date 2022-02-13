import torch
import numpy as np
import pickle
from torch_geometric.nn.models.node2vec import Node2Vec
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
from sparsesvd import sparsesvd
import os.path as osp

CWD = osp.dirname(osp.abspath(__file__))
ROOT = CWD + '/..'
__all__ = ['get_embedding']

def get_embedding(name, data, method):
    if method == 'svd':
        embedding = svd(data, name)
    elif method == 'node2vec':
        embedding = node2vec(data, name)
    elif method == 'onehot':
        embedding = onehot(data, name)
    elif method == 'xavier':
        embedding = xavier(data, name)
    elif method == 'spectral':
        embedding = spectral(data, name)
    else:
        embedding = None
    return embedding

def load_embedding(name, method):
    embedding = pickle.load(open(f'{ROOT}/cache/feature/{method}_{name}.pt', 'rb'))
    print(f'Use cached {method} feature for dataset {name}')
    return embedding

def save_embedding(embedding, name, method):
    pickle.dump(embedding, open(f'{ROOT}/cache/feature/{method}_{name}.pt', 'wb'))

def spectral(data, name, embedding_dim=128):
    try:
        result = load_embedding(name, 'spectral')
        return result
    except FileNotFoundError:
        print(f'cache/feature/spectral_{name}.pt not found! Regenerating it now')

    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include(f'{CWD}/norm_spec.jl')
    print('Setting up spectral embedding')

    if data.setting == 'inductive':
        N = data.num_train_nodes
        edge_index = to_undirected(data.train_edge_index, num_nodes=data.num_train_nodes)
    else:
        N = data.num_nodes
        edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    np_edge_index = np.array(edge_index.T)
    row, col = edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.to_scipy(layout='csr')

    result = torch.tensor(Main.main(adj, embedding_dim)).float()
    save_embedding(result, name, 'spectral')
    return result

def node2vec(data, name, embedding_dim=64, epochs=40):
    try:
        result = load_embedding(name, 'node2vec')
        return result
    except FileNotFoundError:
        print(f'cache/feature/node2vec_{name}.pt not found! Regenerating it now')

    if data.setting == 'inductive':
        model = Node2Vec(data.train_edge_index, embedding_dim, walk_length=20, context_size=10, walks_per_node=10, num_negative_samples=1, sparse=True).to('cuda')
    else:
        model = Node2Vec(data.edge_index, embedding_dim, walk_length=20, context_size=10, walks_per_node=10, num_negative_samples=1, sparse=True).to('cuda')
    loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to('cuda'), neg_rw.to('cuda'))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    for epoch in range(1, epochs + 1):
        loss = train()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    result = model().detach().cpu()
    save_embedding(result, name, 'node2vec')
    return result

def svd(data, name, embedding_dim=64):
    try:
        result = load_embedding(name, 'svd')
        return result
    except FileNotFoundError:
        print(f'cache/feature/svd_{name}.pt not found! Regenerating it now')

    if data.setting == 'inductive':
        N = data.num_train_nodes
        row, col = data.train_edge_index
    else:
        N = data.num_nodes
        row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.to_scipy(layout='csc')
    ut, s, vt = sparsesvd(adj, embedding_dim)

    result = torch.tensor(np.dot(ut.T, np.diag(s)), dtype=torch.float)
    save_embedding(result, name, 'svd')
    return result

def onehot(data, name):
    try:
        result = load_embedding(name, 'onehot')
        return result
    except FileNotFoundError:
        print(f'cache/feature/onehot_{name}.pt not found! Regenerating it now')

    assert data.x is None

    if data.setting == 'inductive':
        result = torch.eye(data.num_train_nodes)
    else:
        result = torch.eye(data.num_nodes)
    
    save_embedding(result, name, 'onehot')
    return result

def xavier(data, name, embedding_dim=64):
    try:
        result = load_embedding(name, 'xavier')
        return result
    except FileNotFoundError:
        print(f'cache/feature/xavier_{name}.pt not found! Regenerating it now')

    assert data.x is None
    if data.setting == 'inductive':
        result = torch.nn.init.xavier_uniform_(torch.zeros((data.num_train_nodes, embedding_dim)))
    else:
        result = torch.nn.init.xavier_uniform_(torch.zeros((data.num_nodes, embedding_dim)))

    save_embedding(result, name, 'xavier')
    return result
