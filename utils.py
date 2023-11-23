import os
import random

import torch
import numpy as np
import scipy.sparse as sp
from tensorflow.keras.utils import to_categorical


def setup_seed(seed, cuda):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cuda is True:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_sparse(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def load_data(dataset, repeat, device, self_loop):
    path = './new_splits/{}/'.format(dataset)

    f = np.loadtxt(path + '{}.feature'.format(dataset), dtype=float)
    l = np.loadtxt(path + '{}.label'.format(dataset), dtype=int)
    test = np.loadtxt(path + '{}test.txt'.format(repeat), dtype=int)
    train = np.loadtxt(path + '{}train.txt'.format(repeat), dtype=int)
    val = np.loadtxt(path + '{}val.txt'.format(repeat), dtype=int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense())).to(device)

    idx_test = test.tolist()
    idx_train = train.tolist()
    idx_val = val.tolist()

    idx_train = torch.LongTensor(idx_train).to(device)
    idx_test = torch.LongTensor(idx_test).to(device)
    idx_val = torch.LongTensor(idx_val).to(device)
    label = torch.LongTensor(np.array(l)).to(device)

    label_oneHot = torch.FloatTensor(to_categorical(l)).to(device)

    struct_edges = np.genfromtxt(path + '{}.edge'.format(dataset), dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
                         shape=(features.shape[0], features.shape[0]), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    sadj = sadj + self_loop * sp.eye(sadj.shape[0])
    nsadj = torch.FloatTensor(sadj.todense()).to(device)

    return nsadj, features, label, label_oneHot, idx_train, idx_val, idx_test
# 将npz转成txt
# root = os.path.split(__file__)[0]
# dataset = 'wisconsin'
# for i in range(10):
#     split_path = root + '/splits/' + dataset + '_split_0.6_0.2_' + str(i) + '.npz'
#     with np.load(split_path) as splits_file:
#         train_mask = splits_file['train_mask']
#         val_mask = splits_file['val_mask']
#         test_mask = splits_file['test_mask']
#     train = np.argwhere(train_mask == 1).flatten()
#     val = np.argwhere(val_mask == 1).flatten()
#     test = np.argwhere(test_mask == 1).flatten()
#     file_train = open(root + '/new_splits/' + dataset + '/' + str(i) + 'train.txt', 'w', encoding='UTF-8')
#     file_val = open(root + '/new_splits/'  + dataset + '/' + str(i) + 'val.txt', 'w', encoding='UTF-8')
#     file_test = open(root + '/new_splits/'  + dataset + '/' + str(i) + 'test.txt', 'w', encoding='UTF-8')
#
#     for i in range(len(train)):
#         file_train.write(str(train[i]) + '\n')
#     file_train.close()
#
#     for i in range(len(val)):
#         file_val.write(str(val[i]) + '\n')
#     file_val.close()
#
#     for i in range(len(test)):
#         file_test.write(str(test[i]) + '\n')
#     file_test.close()