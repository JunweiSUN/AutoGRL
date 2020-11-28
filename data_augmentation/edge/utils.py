import time
import copy
import pickle
import warnings
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def get_scores(edges_pos, edges_neg, A_pred, adj_label):
    # get logists and labels
    preds = A_pred[edges_pos.T]
    preds_neg = A_pred[edges_neg.T]
    logists = np.hstack([preds, preds_neg])
    labels = np.hstack([np.ones(preds.size(0)), np.zeros(preds_neg.size(0))])
    # logists = A_pred.view(-1)
    # labels = adj_label.to_dense().view(-1)
    # calc scores
    roc_auc = roc_auc_score(labels, logists)
    ap_score = average_precision_score(labels, logists)
    precisions, recalls, thresholds = precision_recall_curve(labels, logists)
    pr_auc = auc(recalls, precisions)
    warnings.simplefilter('ignore', RuntimeWarning)
    f1s = np.nan_to_num(2*precisions*recalls/(precisions+recalls))
    best_comb = np.argmax(f1s)
    f1 = f1s[best_comb]
    pre = precisions[best_comb]
    rec = recalls[best_comb]
    thresh = thresholds[best_comb]
    # calc reconstracted adj_mat and accuracy with the threshold for best f1
    adj_rec = copy.deepcopy(A_pred)
    adj_rec[adj_rec < thresh] = 0
    adj_rec[adj_rec >= thresh] = 1
    labels_all = adj_label.to_dense().view(-1).long()
    preds_all = adj_rec.view(-1).long()
    recon_acc = (preds_all == labels_all).sum().float() / labels_all.size(0)
    results = {'roc': roc_auc,
               'pr': pr_auc,
               'ap': ap_score,
               'pre': pre,
               'rec': rec,
               'f1': f1,
               'acc': recon_acc,
               'adj_recon': adj_rec}
    return results

def train_model(args, dl, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # weights for log_lik loss
    adj_t = dl.adj_train
    norm_w = adj_t.shape[0]**2 / float((adj_t.shape[0]**2 - adj_t.sum()) * 2)
    pos_weight = torch.FloatTensor([float(adj_t.shape[0]**2 - adj_t.sum()) / adj_t.sum()]).to(args.device)
    # move input data and label to gpu if needed
    features = dl.features.to(args.device)
    adj_label = dl.adj_label.to_dense().to(args.device)

    best_vali_criterion = 0.0
    best_state_dict = None
    model.train()
    for epoch in range(args.epochs):
        t = time.time()
        A_pred = model(features)
        optimizer.zero_grad()
        loss = norm_w * F.binary_cross_entropy_with_logits(A_pred, adj_label, pos_weight=pos_weight)

        A_pred = torch.sigmoid(A_pred).detach().cpu()
        r = get_scores(dl.val_edges, dl.val_edges_false, A_pred, dl.adj_label)
        # print('Epoch{:3}: train_loss: {:.4f} recon_acc: {:.4f} val_roc: {:.4f} val_ap: {:.4f} f1: {:.4f} time: {:.4f}'.format(
        #     epoch+1, loss.item(), r['acc'], r['roc'], r['ap'], r['f1'], time.time()-t))
        if r[args.criterion] > best_vali_criterion:
            best_vali_criterion = r[args.criterion]
            best_state_dict = copy.deepcopy(model.state_dict())
            # r_test = get_scores(dl.test_edges, dl.test_edges_false, A_pred, dl.adj_label)
            # r_test = r
            # print("          test_roc: {:.4f} test_ap: {:.4f} test_f1: {:.4f} test_recon_acc: {:.4f}".format(
            #         r_test['roc'], r_test['ap'], r_test['f1'], r_test['acc']))
        loss.backward()
        optimizer.step()

    # print("Done! final results: test_roc: {:.4f} test_ap: {:.4f} test_f1: {:.4f} test_recon_acc: {:.4f}".format(
    #         r_test['roc'], r_test['ap'], r_test['f1'], r_test['acc']))

    model.load_state_dict(best_state_dict)
    return model

def sample_graph_det(adj_orig, A_pred, remove_pct, add_pct):
    if remove_pct == 0 and add_pct == 0:
        return copy.deepcopy(adj_orig)
    orig_upper = sp.triu(adj_orig, 1)
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
    adj_pred = sp.csr_matrix((np.ones(len(edges_pred)), edges_pred.T), shape=adj_orig.shape)
    adj_pred = adj_pred + adj_pred.T
    return adj_pred
