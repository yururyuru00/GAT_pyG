import itertools
import random
import numpy as np
import scipy.sparse as sp

import torch

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot

def negative_sample(labels, adj, sample_pos, sample_neg):
    n_nodes, n_class = labels.size()[0], torch.max(labels) + 1

    v = set(range(n_nodes))
    cluster  = [set(torch.where(labels == c_idx)[0].tolist()) for c_idx in range(n_class)]
    neighbor = [set(torch.where(adj[v_idx] > 0.)[0].tolist()) for v_idx in range(n_nodes)]

    p = [neighbor[v_idx] | cluster[c_idx] for v_idx, c_idx in enumerate(labels)]
    n = [v - p[v_idx] for v_idx in range(n_nodes)]

    p = [random.sample(p[v_idx], k=sample_pos) for v_idx in range(n_nodes)]
    n = [random.sample(n[v_idx], k=sample_neg) for v_idx in range(n_nodes)]
    
    p = list(itertools.chain.from_iterable(p))
    n = list(itertools.chain.from_iterable(n))
    i  = [i for i in range(n_nodes) for _ in range(sample_pos)]
    i_ = [i for i in range(n_nodes) for _ in range(sample_neg)]

    return (p, i), (n, i_)

def upgrade_adj(label, adj, mode='hard'):

    def calc_edge_rate(adj):
        n_nodes = adj.size()[0]
        return (int(len(torch.where(adj>0.)[0].tolist())) / (n_nodes**2)) * 100.

    adj_ = adj.clone()
    print('update adj: edge rate {:.2f}% -> '.format(calc_edge_rate(adj)), end='')

    if(mode=='fuzzy'):
        prob_labels = torch.exp(label)
        pred_labels = prob_labels.max(1)[1]
        n_class, n_nodes = torch.max(pred_labels) + 1, pred_labels.size()[0]

        for vi, label_of_vi in enumerate(pred_labels):
            neighbors_of_vi = torch.multinomial(prob_labels[:, label_of_vi], num_samples=10, replacement=False)
            for neighbor_of_vi in neighbors_of_vi:
                adj_[vi][neighbor_of_vi] = 1.
        
    else: # if(mode=='hard')
        n_class, n_nodes = torch.max(label) + 1, label.size()[0]
        cluster = [torch.where(label == c)[0] for c in range(n_class)]
        neighbors_of_all_nodes = [cluster[vi_label] for vi_label in label.tolist()]

        for vi, neighbors_of_vi in enumerate(neighbors_of_all_nodes):
            for neighbor_of_vi in neighbors_of_vi:
                adj_[vi][neighbor_of_vi] = 1.
    
    print('{:.2f}%'.format(calc_edge_rate(adj_)))
    return adj_


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

