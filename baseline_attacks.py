import numpy as np
import argparse
import os.path as osp
import pickle as pkl
from scipy.sparse import csr_matrix
from process_isolated_nodes import process_isolated_nodes, restore_isolated_ndoes

import torch
from torch_geometric.utils import contains_isolated_nodes, subgraph, remove_isolated_nodes
#from torch_geometric.transforms import *

import networkx as nx
from torch_geometric.utils.convert import to_networkx

from deeprobust.graph.data import Pyg2Dpr
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack, DICE, Random, MinMax, PGDAttack, NodeEmbeddingAttack
from deeprobust.graph.utils import preprocess

from pGRACE.dataset import get_dataset

# from random import choice
import random

def attack_model(name, adj, features, labels, device):
    if args.rate < 1:
        n_perturbation = int(args.rate * dataset.data.num_edges / 2)
    else:
        n_perturbation = int(args.rate)
    if name == 'metattack':
        model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                          attack_structure=True, attack_features=False, device=device, lambda_=0).to(device)
        model.attack(features, adj, labels, idx_train, idx_unlabeled,
                     n_perturbations=n_perturbation, ll_constraint=False)
    elif name == 'dice':
        model = DICE()
        model.attack(adj, labels, n_perturbations=n_perturbation)
    elif name == 'random':
        model = Random()
        model.attack(adj, n_perturbations=n_perturbation)
    elif name == 'minmax':
        adj, features, labels = preprocess(adj, csr_matrix(features), labels, preprocess_adj=False)  # conver to tensor
        model = MinMax(surrogate, nnodes=adj.shape[0], loss_type='CE', device=device).to(device)
        model.attack(features, adj, labels, idx_train, n_perturbations=n_perturbation)
    elif name == 'pgd':
        adj, features, labels = preprocess(adj, csr_matrix(features), labels, preprocess_adj=False) # conver to tensor
        model = PGDAttack(surrogate, nnodes=adj.shape[0], loss_type='CE', device=device).to(device)
        model.attack(features, adj, labels, idx_train, n_perturbations=n_perturbation)
    elif name == 'nodeembeddingattack':
        model = NodeEmbeddingAttack()
        model.attack(adj, attack_type='remove', n_perturbations=n_perturbation)
    else:
        raise ValueError('Invalid name of the attack method!')
    return model


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--rate', type=float, default=0.10)
parser.add_argument('--method', type=str)
parser.add_argument('--device', default='cpu')

args = parser.parse_args()
device = args.device
path = osp.expanduser('dataset')
path = osp.join(path, args.dataset)
dataset = get_dataset(path, args.dataset)
mapping = None

data = dataset[0]
G = to_networkx(data)
# G_un = G.to_undirected()

# if nx.is_connected(G_un) == False:
#     num_con_com = nx.number_connected_components(G_un)
# print(num_con_com)

# left_nodes = list(range(2708))
# while G_un.number_of_nodes() > 1000:
#     cur = choice(left_nodes) #待删除节点
#     left_nodes.remove(cur)
#     G_tmp = G_un.subgraph(left_nodes)
#     if nx.number_connected_components(G_tmp) == num_con_com:
#         G_un = G_tmp
#     else:
#         left_nodes.append(cur)
# print(G_un)

# print(list(nx.isolates(G)))
# left_nodes = list(range(2708))
# while G.number_of_nodes() > 1000:
#     cur = choice(left_nodes) #待删除节点
#     left_nodes.remove(cur)
#     G_tmp = G.subgraph(left_nodes)
#     if list(nx.isolates(G_tmp)):
#         left_nodes.append(cur)
#     else:
#         G = G_tmp
# print(G)

G_un = G.to_undirected()
generate_subgraph_nodes = random.sample(range(2708), 1000)
# print(generate_subgraph_nodes)
G_sub = G_un.subgraph(generate_subgraph_nodes)
largest_cc = max(nx.connected_components(G_sub), key = len)
G_sub_largest_cc = G_sub.subgraph(largest_cc)
print(G_sub_largest_cc)

if args.method == 'nodeembeddingattack' and contains_isolated_nodes(dataset.data.edge_index):
    new_edge_index, mapping, mask = process_isolated_nodes(dataset.data.edge_index)
    new_num_nodes = int(new_edge_index.max() + 1)
    edge_sp_adj = torch.sparse.FloatTensor(new_edge_index.to(device),
                                           torch.ones(new_edge_index.shape[1]).to(device),
                                           [new_num_nodes, new_num_nodes])
    edge_adj = edge_sp_adj.to_dense().cpu().numpy()
    adj = csr_matrix(edge_adj)
    model = attack_model(args.method, adj, None, None, device)
    modified_adj = torch.Tensor(model.modified_adj.todense())
    edge_index = modified_adj.nonzero(as_tuple=False).T
    restored_edge_index = restore_isolated_ndoes(edge_index, mapping)
    edge_sp_adj = torch.sparse.FloatTensor(restored_edge_index.to(device),
                                           torch.ones(restored_edge_index.shape[1]).to(device),
                                           [dataset.data.num_nodes, dataset.data.num_nodes])
    modified_adj = edge_sp_adj.to_dense().to(device)
    pkl.dump(modified_adj, open(
        'poisoned_adj/%s_%s_%f_adj.pkl' % (args.dataset, args.method, args.rate),
        'wb'))
    exit()
# else:
#     adj = csr_matrix((np.ones(dataset.data.edge_index.shape[1]),
#                               (dataset.data.edge_index[0], dataset.data.edge_index[1])), shape=(dataset.data.num_nodes, dataset.data.num_nodes))
#     features = dataset.data.x.numpy()
#     labels = dataset.data.y.numpy()

data = Pyg2Dpr(dataset)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)
if args.method in ['metattack', 'minmax', 'pgd']:
    # Setup Surrogate model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
    surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
# Setup Attack Model
model = attack_model(args.method, adj, features, labels, device)
if args.method in ['random', 'dice', 'nodeembeddingattack', 'randomremove', 'randomflip']:
    modified_adj = torch.Tensor(model.modified_adj.todense())
else:
    modified_adj = model.modified_adj  # modified_adj is a torch.tensor
pkl.dump(modified_adj, open('poisoned_adj/%s_%s_%f_adj.pkl' % (args.dataset, args.method, args.rate), 'wb'))
