import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv


class GATNet(nn.Module):
    def __init__(self, dataset, nfeat, nhid, nclass, dropout, nheads):
        """Dense version of GAT."""
        super(GATNet, self).__init__()
        self.dropout = dropout

        self.attention = GATConv(nfeat, nhid, heads=nheads, dropout=dropout)

        if(dataset in ['Cora', 'CiteSeer']):
            self.out_att = GATConv(nhid * nheads, nclass, heads=1, dropout=dropout)
        if(dataset == 'PubMed'):
            self.out_att = GATConv(nhid * nheads, nclass, heads=nheads, dropout=dropout, concat=False)

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.attention(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.out_att(x, edge_index)
        x = F.elu(x)
        return F.log_softmax(x, dim=1)