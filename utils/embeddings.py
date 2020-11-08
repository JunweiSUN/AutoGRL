import numpy as np
import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import os.path as osp
from sparsesvd import sparsesvd

CWD = osp.dirname(__file__)

def spectral(data, post_fix, embedding_dim):
    from julia.api import Julia
    jl = Julia(compiled_modules=False)
    from julia import Main
    Main.include(f'{CWD}/norm_spec.jl')
    print('Setting up spectral embedding')
    data.edge_index = to_undirected(data.edge_index)
    np_edge_index = np.array(data.edge_index.T)

    N = data.num_nodes
    row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.to_scipy(layout='csr')

    result = torch.tensor(Main.main(adj, 128)).float()
    torch.save(result, f'{CWD}/../embeddings/spectral_{post_fix}.pt')
    return result

def node2vec(data, post_fix, embedding_dim, epochs=20):
    from torch_geometric.nn.models.node2vec import Node2Vec
    model = Node2Vec(data.edge_index, embedding_dim, walk_length=20, context_size=10, walks_per_node=10, num_negative_samples=1, sparse=True).to('cuda')
    loader = model.loader(batch_size=128, shuffle=True, num_workers=8)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)

    def train():
        model.train()
        total_loss = 0
        for pos_rw, new_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to('cuda'), new_rw.to('cuda'))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    for epoch in range(1, epochs+1):
        loss = train()
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

    result = model().detach().cpu()
    torch.save(result, f'{CWD}/../embeddings/node2vec_{post_fix}.pt')
    return result

def svd(data, post_fix, embedding_dim):
    N = data.num_nodes
    row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.to_scipy(layout='csc')
    ut, s, vt = sparsesvd(adj, embedding_dim)

    result = torch.tensor(np.dot(ut.T, np.diag(s)), dtype=torch.float)
    torch.save(result, f'{CWD}/../embeddings/svd_{post_fix}.pt')
    return result


def get_embeddings(data, embedding_type, post_fix, embedding_dim=128, use_cache=True):
    if use_cache:
        try:
            x = torch.load(f'{CWD}/../embeddings/{embedding_type}_{post_fix}.pt')
            print(f'Using cached {embedding_type}_{post_fix}.pt')
            return x
        except:
            print(f'embeddings/{embedding_type}_{post_fix}.pt not found! Regenerating it now')

    if embedding_type == 'spectral':
        return spectral(data, post_fix, embedding_dim)
    
    if embedding_type == 'node2vec':
        return node2vec(data, post_fix, embedding_dim)

    if embedding_type == 'svd':
        return svd(data, post_fix, embedding_dim)

    raise NotImplementedError('Not implemented embedding type')
