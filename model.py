from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, dropout):
        super(MLP, self).__init__()
        if num_layers == 1:
            hidden_size = out_size

        self.pipeline = nn.Sequential(OrderedDict([
            ('layer_0', nn.Linear(in_size, hidden_size, bias=(num_layers != 1))),
            ('dropout_0', nn.Dropout(dropout)),
            ('relu_0', nn.ReLU())
        ]))

        for i in range(1, num_layers):
            if i == num_layers - 1:
                self.pipeline.add_module('layer_{}'.format(i), nn.Linear(hidden_size, out_size, bias=True))
            else:
                self.pipeline.add_module('layer_{}'.format(i), nn.Linear(hidden_size, hidden_size, bias=True))
                self.pipeline.add_module('dropout_{}'.format(i), nn.Dropout(dropout))
                self.pipeline.add_module('relu_{}'.format(i), nn.ReLU())

        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, feature):
        return F.softmax(self.pipeline(feature), dim=1)


class GraphConv(nn.Module):
    def __init__(self, in_size, out_size, bias=True):
        super(GraphConv, self).__init__()
        self.W = nn.Linear(in_size, out_size, bias)

    def forward(self, g, feature):
        h = torch.mm(g, feature)
        return self.W(h)


class GCN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, dropout):
        super(GCN, self).__init__()
        if num_layers == 1:
            hidden_size = out_size

        self.num_layers = num_layers
        if dropout > 0.:
            self.feat_drop = nn.Dropout(dropout)
        else:
            self.feat_drop = lambda x: x

        self.W = nn.ModuleList([nn.Linear(in_size, hidden_size, bias=True)])
        self.gnn_layers = nn.ModuleList([GraphConv(in_size, hidden_size)])
        for i in range(1, num_layers):
            if i == num_layers - 1:
                self.W.append(nn.Linear(hidden_size, out_size, bias=True))
                self.gnn_layers.append(GraphConv(hidden_size, out_size, bias=True))
            else:
                self.W.append(nn.Linear(hidden_size, hidden_size, bias=True))
                self.gnn_layers.append(GraphConv(hidden_size, hidden_size))

        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, g, feature, feat_balance):
        h = feature
        for i, layer in enumerate(self.gnn_layers):
            if i == self.num_layers - 1:
                h = feat_balance * layer(g, h) + (1 - feat_balance) * self.W[i](h)

            else:
                h = self.feat_drop(h)
                h = feat_balance * layer(g, h) + (1 - feat_balance) * self.W[i](h)
                h = F.relu(h)
        return h


class LSM_GCN(nn.Module):
    def __init__(self, mlp_module, gcn_module, loss_weight, feat_balance, device):
        super(LSM_GCN, self).__init__()

        self.loss_weight = loss_weight
        self.device = device
        self.mlp = mlp_module
        self.gcn = gcn_module
        self.feat_balance = feat_balance

    def normalize(self, adj):
        rowsum = torch.sum(adj, dim=1)
        r_inv = torch.diag(rowsum ** (-1))
        r_inv[torch.isinf(r_inv)] = 0.
        r_inv[torch.isnan(r_inv)] = 0.
        norm_adj = torch.mm(r_inv, adj)
        return norm_adj

    def forward(self, feature, adj, idx, label, labels_oneHot, train_idx):
        B = self.mlp(feature)

        S = self.get_S_matrix(feature, labels_oneHot, B.clone(), train_idx)

        score = torch.mm(torch.mm(B, S), B.t()) * adj

        zero_vec = -9e15 * torch.ones_like(score)
        g = torch.where(adj > 0, score, zero_vec)

        g = F.softmax(g, dim=1)

        output = self.gcn(g, feature, self.feat_balance)
        logits = F.softmax(output, dim=1)

        gcn_loss = F.nll_loss(torch.log(logits[idx]), label)
        mlp_loss = F.nll_loss(torch.log(B[idx]), label)
        final_loss = self.loss_weight[0] * gcn_loss + self.loss_weight[1] * mlp_loss

        return logits, final_loss, S.detach(), S.detach(), output.detach()

    def get_S_matrix(self, features, y, soft_y=None, mask=None):
        soft_y[mask] = y[mask]
        n = features.shape[0]
        A = torch.mm(features, features.t())
        B = torch.mm(features.sum(1).reshape(n, 1), features.sum(1).reshape(1, n))
        sim_adj = (A / B).to(self.device)
        sim_adj[torch.isnan(sim_adj)] = 0.
        sum_matric = torch.ones((1, soft_y.shape[0])).to(self.device)
        B = torch.mm(sum_matric, soft_y).to(self.device)
        B = torch.mm(B.t(), B)
        S = (torch.mm(torch.mm(soft_y.t(), sim_adj), soft_y) / B).to(self.device)
        S = S + S * torch.eye(S.shape[0]).to(self.device)
        S= self.normalize(S).to(self.device)
        return S
