import os
import pickle
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import remove_self_loops, to_scipy_sparse_matrix
from sklearn.preprocessing import normalize
from .utils import sparse_to_tuple

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CUR_DIR)

class DataLoader():
    def __init__(self, args, data):
        self.args = args
        self.load_data(data)
        self.mask_test_edges(args.val_frac, args.test_frac, args.no_mask)
        self.normalize_adj()
        self.to_pyt_sp()

    def load_data(self, data):        
        adj = data.edge_index
        adj, _ = remove_self_loops(adj)
        adj = to_scipy_sparse_matrix(adj).asformat('csr')
        features = data.x.numpy()
        features = sp.csr_matrix(features)
        self.adj_orig = adj
        self.features_orig = features

    def load_data_binary(self, dataset):
        adj = pickle.load(open(f'{BASE_DIR}/data/citation_networks_binary/{dataset}_adj.pkl', 'rb'))
        if adj.diagonal().sum() > 0:
            adj = sp.coo_matrix(adj)
            adj.setdiag(0)
            adj.eliminate_zeros()
            adj = sp.csr_matrix(adj)
        features = pickle.load(open(f'{BASE_DIR}/data/citation_networks_binary/{dataset}_features.pkl', 'rb'))
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        features = sp.csr_matrix(features)
        self.adj_orig = adj
        if dataset == 'ppi':
            features = features.toarray()
            m = features.mean(axis=0)
            s = features.std(axis=0, ddof=0, keepdims=True) + 1e-12
            features -= m
            features /= s
            self.features_orig = sp.csr_matrix(features)
        else:
            self.features_orig = normalize(features, norm='l1', axis=1)

    def mask_test_edges(self, val_frac, test_frac, no_mask):
        adj = self.adj_orig
        assert adj.diagonal().sum() == 0

        adj_triu = sp.triu(adj)
        edges = sparse_to_tuple(adj_triu)[0]
        edges_all = sparse_to_tuple(adj)[0]
        num_test = int(np.floor(edges.shape[0] * test_frac))
        num_val = int(np.floor(edges.shape[0] * val_frac))

        all_edge_idx = list(range(edges.shape[0]))
        np.random.shuffle(all_edge_idx)
        val_edge_idx = all_edge_idx[:num_val]
        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        test_edges = edges[test_edge_idx]
        val_edges = edges[val_edge_idx]
        if no_mask:
            train_edges = edges
        else:
            train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

        def ismember(a, b, tol=5):
            rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
            return np.any(rows_close)

        test_edges_false = []
        while len(test_edges_false) < len(test_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if ismember([idx_j, idx_i], np.array(test_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(test_edges_false)):
                    continue
            test_edges_false.append([idx_i, idx_j])

        val_edges_false = []
        while len(val_edges_false) < len(val_edges):
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            if ismember([idx_i, idx_j], train_edges):
                continue
            if ismember([idx_j, idx_i], train_edges):
                continue
            if ismember([idx_i, idx_j], val_edges):
                continue
            if ismember([idx_j, idx_i], val_edges):
                continue
            if val_edges_false:
                if ismember([idx_j, idx_i], np.array(val_edges_false)):
                    continue
                if ismember([idx_i, idx_j], np.array(val_edges_false)):
                    continue
            val_edges_false.append([idx_i, idx_j])

        # Re-build adj matrix
        adj_train = sp.csr_matrix((np.ones(train_edges.shape[0]), (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
        self.adj_train = adj_train + adj_train.T
        self.adj_label = adj_train + sp.eye(adj_train.shape[0])
        # NOTE: these edge lists only contain single direction of edge!
        self.val_edges = val_edges
        self.val_edges_false = np.asarray(val_edges_false)
        self.test_edges = test_edges
        self.test_edges_false = np.asarray(test_edges_false)

    def normalize_adj(self):
        adj_ = sp.coo_matrix(self.adj_train)
        adj_.setdiag(1)
        rowsum = np.array(adj_.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        self.adj_norm = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()

    def to_pyt_sp(self):
        adj_norm_tuple = sparse_to_tuple(self.adj_norm)
        adj_label_tuple = sparse_to_tuple(self.adj_label)
        features_tuple = sparse_to_tuple(self.features_orig)
        self.adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm_tuple[0].T),
                                                torch.FloatTensor(adj_norm_tuple[1]),
                                                torch.Size(adj_norm_tuple[2]))
        self.adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label_tuple[0].T),
                                                torch.FloatTensor(adj_label_tuple[1]),
                                                torch.Size(adj_label_tuple[2]))
        self.features = torch.sparse.FloatTensor(torch.LongTensor(features_tuple[0].T),
                                                torch.FloatTensor(features_tuple[1]),
                                                torch.Size(features_tuple[2]))

