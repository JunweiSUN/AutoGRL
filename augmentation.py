from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np
import torch
from copy import deepcopy
from scipy import sparse

def label_propagation(data, alpha=0.99, threshold=0.01, max_iter=10):
    '''
    Label propagation algorithm, modified from the NetworkX implementation.
    Label some nodes then add them to the training set.
    Only support undirected graphs.
    '''
    X = to_scipy_sparse_matrix(data.edge_index).asformat('csr').astype('long')
    n_samples = X.shape[0]
    n_classes = int(max(data.y)) + 1
    F = np.zeros((n_samples, n_classes))

    degrees = X.sum(axis=0).A[0]
    degrees[degrees == 0] = 1  # Avoid division by 0
    D2 = np.sqrt(sparse.diags((1.0 / degrees), offsets=0))
    P = alpha * D2.dot(X).dot(D2)

    train_idxs = torch.where(data.train_mask==True)[0].numpy()
    y = data.y.numpy()
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

