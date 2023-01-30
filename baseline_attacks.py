import numpy as np
import argparse
import os.path as osp
import pickle as pkl
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from process_isolated_nodes import process_isolated_nodes, restore_isolated_ndoes

import torch
torch.cuda.empty_cache()
from torch_geometric.utils import contains_isolated_nodes, subgraph, from_networkx
from torch_geometric.data import Data

import networkx as nx
from torch_geometric.utils.convert import to_networkx

from deeprobust.graph.data import Pyg2Dpr
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack, DICE, Random, MinMax, PGDAttack, NodeEmbeddingAttack
from deeprobust.graph.utils import preprocess

from pGRACE.dataset import get_dataset

import random

def attack_model(name, adj, features, labels, device):
    if args.rate < 1:
        n_perturbation = int(args.rate * sub_dataset.data.num_edges / 2)
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
parser.add_argument('--top', type=int)
parser.add_argument('--hop', type=int)
parser.add_argument('--device', default='cpu')

args = parser.parse_args()
device = args.device
path = osp.expanduser('dataset')
path = osp.join(path, args.dataset)
dataset = get_dataset(path, args.dataset)
mapping = None

data_cora = dataset[0]
print(data_cora.y.size())
# PyG to networkx
G = to_networkx(data_cora)
G_un = G.to_undirected()
pr = nx.pagerank(G)
sorted_pr = sorted(pr, key = pr.__getitem__, reverse = True)
top_pr = sorted_pr[:args.top]
print(top_pr)

def get_neigbors(g, node, depth = 1):
    output = {}
    layers = dict(nx.bfs_successors(g, source = node, depth_limit = depth))
    nodes = [node]
    for i in range(1, depth + 1):
        output[i] = []
        for x in nodes:
            output[i].extend(layers.get(x, []))
        nodes = output[i]
    return output

graph_indicator = 1
existing_node_num = 0 #每个图之前的全部图，包含的点个数
for i in top_pr:
    node_cluster = get_neigbors(G_un, i, args.hop)
    subgraph_nodes = [i]
    for _, value in node_cluster.items():
        for v in value:
            subgraph_nodes.append(v)
    G_sub = G_un.subgraph(subgraph_nodes)

    node_id = [] #节点id重制
    file = open('SubCora_A.txt', 'a')
    edges = G_sub.edges
    for i, j in edges:
        if not node_id.count(i) and node_id.count(j):
            node_id.append(i)
        elif node_id.count(i) and not node_id.count(j):
            node_id.append(j)
        elif not node_id.count(i) and not node_id.count(j):
            node_id.append(i)
            node_id.append(j)
        file.write(str(node_id.index(i) + 1 + existing_node_num) + ", " + str(node_id.index(j) + 1 + existing_node_num) + "\n")
        file.write(str(node_id.index(j) + 1 + existing_node_num) + ", " + str(node_id.index(i) + 1 + existing_node_num) + "\n")
    file.close()

    file = open('SubCora_graph_indicator.txt', 'a')
    for i in range(G_sub.number_of_nodes()):
        file.write(str(graph_indicator) + "\n")
    file.close()

    file = open('SubCora_graph_labels.txt', 'a')
    file.write("1\n")
    file.close()

    node_y = []
    file = open('SubCora_node_labels.txt', 'a')
    edges = G_sub.edges
    for i, j in edges:
        if not node_y[node_id.index(i)]:
            node_y.insert(node_id.index(i), data_cora.y[i])
        if not node_y[node_id.index(j)]:
            node_y.insert(node_id.index(j), data_cora.y[j])
    for i in G_sub.number_of_nodes():
        file.write(str(node_y[i]) + "\n")
    file.close()

    existing_node_num += G_sub.number_of_nodes()
    graph_indicator += 1

u = []
v = []
for i, j in edges: #在pyg中，无向图要存成双向图
    u.append(i)
    u.append(j)
    v.append(j)
    v.append(i)
u = [u]
v = [v]
u = torch.IntTensor(u)
v = torch.IntTensor(v)
edge_index = torch.cat((u, v), dim = 0)
data_sub = Data(x = data_cora.x, edge_index = edge_index, y = data_cora.y, train_mask = data_cora.train_mask, val_mask = data_cora.val_mask, test_mask = data_cora.test_mask)

sub_dataset = MyOwnDataset(str(args.dataset) + "_subgraph_2000_" + str(args.times))
print(sub_dataset.data)

if args.method == 'nodeemsub_dataseteddingattack' and contains_isolated_nodes(sub_dataset.data.edge_index):
    new_edge_index, mapping, mask = process_isolated_nodes(sub_dataset.data.edge_index)
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
                                           [sub_dataset.data.num_nodes, sub_dataset.data.num_nodes])
    modified_adj = edge_sp_adj.to_dense().to(device)
    pkl.dump(modified_adj, open(
        'poisoned_adj/%s_%s_%f_adj.pkl' % (args.dataset, args.method, args.rate),
        'wb'))
    exit()

data = Pyg2Dpr(sub_dataset)
adj, features, labels = data.adj, data.features, data.labels
# print(adj)
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
    # print(modified_adj)
else:
    modified_adj = model.modified_adj  # modified_adj is a torch.tensor
    # print(modified_adj)
pkl.dump(modified_adj, open('poisoned_adj/%s_subgraph_2000_%d_%s_%f_adj.pkl' % (args.dataset, args.times, args.method, args.rate), 'wb'))
