import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn.conv import GATConv

from models import GATNet
from utils import accuracy

def train(epoch, data, model, optimizer):
    # train
    model.train()
    optimizer.zero_grad()

    prob_labels = model(data.x, data.edge_index)
    loss_train  = F.nll_loss(prob_labels[data.train_mask], data.y[data.train_mask])
    acc_train = accuracy(prob_labels[data.train_mask], data.y[data.train_mask])

    loss_train.backward()
    optimizer.step()

    # validation
    model.eval()
    prob_labels_val = model(data.x, data.edge_index)

    loss_val = F.nll_loss(prob_labels_val[data.val_mask], data.y[data.val_mask])
    acc_val = accuracy(prob_labels_val[data.val_mask], data.y[data.val_mask])
    if(epoch%10 ==0):
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.data.item()),
              'acc_train: {:.4f}'.format(acc_train.data.item()),
              'loss_val: {:.4f}'.format(loss_val.data.item()),
              'acc_val: {:.4f}'.format(acc_val.data.item()))


def test(data, model):
    model.eval()
    prob_labels_test = model(data.x, data.edge_index)
    loss_test = F.nll_loss(prob_labels_test[data.test_mask], data.y[data.test_mask])
    acc_test = accuracy(prob_labels_test[data.test_mask], data.y[data.test_mask])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))


def main():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora', help='name of dataset of {Cora, CiteSeer, PubMed}')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--pos_sample', type=int, default=1, help='number of positive sample')
    parser.add_argument('--neg_sample', type=int, default=5, help='number of negative sample')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid('../data', args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    print(data)

    n_features, n_class = data.x.size()[1], torch.max(data.y).data.item() + 1
    model = GATNet(args.dataset, n_features, args.hidden, n_class, args.dropout, args.n_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, 501):
        train(epoch, data, model, optimizer)
    test(data, model)


if __name__ == "__main__":
    main()