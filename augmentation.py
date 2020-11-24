from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx, is_undirected, from_networkx, to_scipy_sparse_matrix, from_scipy_sparse_matrix
import numpy as np
import scipy
import torch
import random
from copy import deepcopy
from scipy import sparse
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GAE
from torch_geometric.utils import train_test_split_edges
from networkx import degree_centrality, in_degree_centrality, eigenvector_centrality, pagerank

def label_propagation(data, alpha=0.99, max_iter=10):
    '''
    Label propagation algorithm, modified from the NetworkX implementation.
    Label some nodes then add them to the training set.
    Only support undirected graphs.
    '''
    if hasattr(data, 'adj_t'):
        X = data.adj_t.to_scipy(layout='csr').astype('long')
    else:
        X = to_scipy_sparse_matrix(data.edge_index).asformat('csr').astype('long')
    n_samples = X.shape[0]
    n_classes = int(max(data.y)) + 1
    F = np.zeros((n_samples, n_classes))

    degrees = X.sum(axis=0).A[0]
    degrees[degrees == 0] = 1  # Avoid division by 0
    D2 = np.sqrt(sparse.diags((1.0 / degrees), offsets=0))
    P = alpha * D2.dot(X).dot(D2)

    train_idxs = torch.where(data.train_mask==True)[0].numpy()
    y = data.y.numpy().squeeze()
    labels = []
    label_to_id = {}
    lid = 0

    for node_id in train_idxs:
        label = y[node_id]
        if label not in label_to_id:
            label_to_id[label] = lid
            lid += 1
        labels.append([node_id, label_to_id[label]])
    labels = np.array(labels)
    label_dict = np.array(
        [label for label, _ in sorted(label_to_id.items(), key=lambda x: x[1])]
    )
    B = np.zeros((n_samples, n_classes))
    B[labels[:, 0], labels[:, 1]] = 1 - alpha

    remaining_iter = max_iter
    while remaining_iter > 0:
        F = P.dot(F) + B
        remaining_iter -= 1

    predicted_label_ids = np.argmax(F, axis=1)
    predicted = label_dict[predicted_label_ids].tolist()

    all_labeled_mask = data.train_mask + data.val_mask + data.test_mask
    count = 0
    for node_id, row in enumerate(F):
        if row.max() >= (row.sum() - row.max()) * 2 and not all_labeled_mask[node_id]:
            data.train_mask[node_id] = True
            data.y[node_id] = predicted[node_id]
            count += 1
    print('Label propagation: Label additional {} nodes in the training set.'.format(count))

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=True)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

def sample_graph_det(adj_orig, A_pred, remove_pct, add_pct):
    if remove_pct == 0 and add_pct == 0:
        return copy.deepcopy(adj_orig)
    orig_upper = scipy.sparse.triu(adj_orig, 1)
    n_edges = orig_upper.nnz
    edges = np.asarray(orig_upper.nonzero()).T
    if remove_pct:
        n_remove = int(n_edges * remove_pct / 100)
        pos_probs = A_pred[edges.T[0], edges.T[1]]
        e_index_2b_remove = np.argpartition(pos_probs, n_remove)[:n_remove]
        mask = np.ones(len(edges), dtype=bool)
        mask[e_index_2b_remove] = False
        edges_pred = edges[mask]
    else:
        edges_pred = edges

    if add_pct:
        n_add = int(n_edges * add_pct / 100)
        # deep copy to avoid modifying A_pred
        A_probs = np.array(A_pred)
        # make the probabilities of the lower half to be zero (including diagonal)
        A_probs[np.tril_indices(A_probs.shape[0])] = 0
        # make the probabilities of existing edges to be zero
        A_probs[edges.T[0], edges.T[1]] = 0
        all_probs = A_probs.reshape(-1)
        e_index_2b_add = np.argpartition(all_probs, -n_add)[-n_add:]
        new_edges = []
        for index in e_index_2b_add:
            i = int(index / A_probs.shape[0])
            j = index % A_probs.shape[0]
            new_edges.append([i, j])
        edges_pred = np.concatenate((edges_pred, new_edges), axis=0)
    adj_pred = scipy.sparse.csr_matrix((np.ones(len(edges_pred)), edges_pred.T), shape=adj_orig.shape)
    adj_pred = adj_pred + adj_pred.T
    return adj_pred

def add_and_remove_edges(data, remove_pct, add_pct, hidden_channels=16, epochs=400):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    edge_index = deepcopy(data.edge_index)
    # data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0)
    # num_features = data.x.shape[1]
    # model = GAE(GCNEncoder(num_features, hidden_channels))
    # model = model.to(device)
    # x = data.x.to(device)
    # train_pos_edge_index = data.train_pos_edge_index.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # best_val_auc = 0
    # best_z = None
    # for epoch in range(1, epochs + 1):
    #     model.train()
    #     optimizer.zero_grad()
    #     z = model.encode(x, train_pos_edge_index)
    #     loss = model.recon_loss(z, train_pos_edge_index)
    #     loss.backward()
    #     optimizer.step()

    #     model.eval()
    #     with torch.no_grad():
    #         z = model.encode(x, train_pos_edge_index)

    #     auc, ap = model.test(z, data.val_pos_edge_index, data.val_neg_edge_index)
    #     print('Val | Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))
    #     if auc > best_val_auc:
    #         best_val_auc = auc
    #         best_z = deepcopy(z)
    
    # A_pred = torch.sigmoid(torch.mm(z, z.T)).cpu().numpy()
    import pickle
    A_pred = pickle.load(open('citeseer_graph_2_logits.pkl', 'rb'))
    adj_orig = to_scipy_sparse_matrix(edge_index).asformat('csr')
    adj_pred = sample_graph_det(adj_orig, A_pred, remove_pct, add_pct)
    data.edge_index, _ = from_scipy_sparse_matrix(adj_pred)

def graph_perturbation(data, centrality_type='degree', data_type='edge'):
    '''
    Data augmentation on nodes or edges
    data: torch_geometric.Data
    centrality_type: str, one of ['degree', 'eigenvector', 'pagerank']
    data_type: str, node or edge
    '''
    undirected_flag = is_undirected(data.edge_index)
    G = to_networkx(data, to_undirected=undirected_flag)
    if centrality_type == 'degree':
        if undirected_flag:
            node_c = degree_centrality(G) # node_c for node_centrality
        else:
            node_c = in_degree_centrality(G)
    elif centrality_type == 'eigenvector':
        node_c = eigenvector_centrality(G)
    elif centrality_type == 'pagerank':
        node_c = pagerank(G)
    else:
        raise NotImplementedError('Not implemented graph perturbation method.')

    if data_type == 'node':
        pass
    
    elif data_type == 'edge':
        edge_c = {} # # edge_c for edge_centrality
        for edge in G.edges:
            u, v = edge[0], edge[1]
            if undirected_flag:
                edge_c[edge] = np.log((node_c[u] + node_c[v]) / 2)
            else:
                edge_c[edge] = np.log(node_c[v])

        s_max = max(edge_c.values())
        s_mean = sum(edge_c.values()) / len(edge_c)
        p_e = 0.5
        p_tao = 0.5
        for edge, s in edge_c.items():
            edge_c[edge] = min((s_max - s) / (s_max - s_mean) * p_e, p_tao)

        removed_edges = []
        probs = np.random.uniform(0, 1, (len(edge_c)))
        for i, (edge, p) in enumerate(edge_c.items()):
            if probs[i] < p:
                removed_edges.append(edge)
        
        G.remove_edges_from(removed_edges)
        print('Remove {} edges'.format(len(removed_edges)))
        data.edge_index = from_networkx(G).edge_index
    else:
        raise NotImplementedError('Not supported data type')



