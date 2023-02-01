import numpy as np
import argparse
import os
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
from torch_geometric.data import InMemoryDataset, download_url

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

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    #返回数据集源文件名
    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]
    #返回process方法所需的保存文件名。你之后保存的数据集名字和列表里的一致
    @property
    def processed_file_names(self):
        return ['data.pt']
    def process(self):
        # Read data into huge `Data` list.
        data_list = [data_sub]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
if args.method == 'nodembeddingattack' and contains_isolated_nodes(sub_dataset.data.edge_index):
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

path = 'SubCora_' + str(args.rate) + "/"
os.mkdir(path)

data_cora = dataset[0]
data_cora_x = np.array(data_cora.x)
data_cora_y = np.array(data_cora.y)
data_cora_train_mask = np.array(data_cora.train_mask)
data_cora_val_mask = np.array(data_cora.val_mask)
data_cora_test_mask = np.array(data_cora.test_mask)
# PyG to networkx
G = to_networkx(data_cora)
G_un = G.to_undirected()
pr = nx.pagerank(G)
top_pr = sorted(pr, key=pr.__getitem__, reverse=True)
top_pr = top_pr[10:args.top]
top_pr = top_pr[::-1]
print(top_pr)

def get_neigbors(g, node, depth=1):
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
for cur in top_pr:
    # torch.cuda.empty_cache()
    node_cluster = get_neigbors(G_un, cur, args.hop)
    subgraph_nodes = [cur]
    for _, value in node_cluster.items():
        for v in value:
            subgraph_nodes.append(v)
    G_sub = G_un.subgraph(subgraph_nodes)
    print("G_sub=", G_sub)
    edges = G_sub.edges

    node_id = [] #节点id重制
    u = []
    v = []
    file = open(path + 'SubCora_A.txt', 'a')
    for i, j in edges:
        if not node_id.count(i):
            node_id.append(i)
        if not node_id.count(j):
            node_id.append(j)
        u.append(node_id.index(i))
        v.append(node_id.index(j))
        u.append(node_id.index(j))
        v.append(node_id.index(i))
        file.write(str(node_id.index(i) + 1 + existing_node_num) + ", " + str(node_id.index(j) + 1 + existing_node_num) + "\n")
        file.write(str(node_id.index(j) + 1 + existing_node_num) + ", " + str(node_id.index(i) + 1 + existing_node_num) + "\n")
    file.close()
    u = [u]
    v = [v]
    u = torch.IntTensor(u)
    v = torch.IntTensor(v)
    edge_index = torch.cat((u, v), dim = 0)

    file = open(path + 'SubCora_graph_indicator.txt', 'a')
    for i in range(G_sub.number_of_nodes()):
        file.write(str(graph_indicator) + "\n")
    file.close()

    file = open(path + 'SubCora_graph_labels.txt', 'a')
    file.write("1\n")
    file.close()
    
    node_x = []
    node_y = []
    node_train_mask = []
    node_val_mask = []
    node_test_mask = []
    flag = [True for i in range(G_sub.number_of_nodes())]
    file = open(path + 'SubCora_node_labels.txt', 'a')
    for i, j in edges:
        if flag[node_id.index(i)]:
            node_x.insert(node_id.index(i), data_cora_x[i])
            node_y.insert(node_id.index(i), data_cora_y[i])
            node_train_mask.insert(node_id.index(i), data_cora_train_mask[i])
            node_val_mask.insert(node_id.index(i), data_cora_val_mask[i])
            node_test_mask.insert(node_id.index(i), data_cora_test_mask[i])
            flag[node_id.index(i)] = False
        if flag[node_id.index(j)]:
            node_x.insert(node_id.index(j), data_cora_x[j])
            node_y.insert(node_id.index(j), data_cora_y[j])
            node_train_mask.insert(node_id.index(j), data_cora_train_mask[j])
            node_val_mask.insert(node_id.index(j), data_cora_val_mask[j])
            node_test_mask.insert(node_id.index(j), data_cora_test_mask[j])
            flag[node_id.index(j)] = False
    for i in range(G_sub.number_of_nodes()):
        file.write(str(node_y[i]) + "\n")
    file.close()

    node_x = torch.tensor(node_x)
    node_y = torch.tensor(node_y)
    node_train_mask = torch.tensor(node_train_mask)
    node_val_mask = torch.tensor(node_val_mask)
    node_test_mask = torch.tensor(node_test_mask)

    data_sub = Data(x=node_x, edge_index=edge_index, y=node_y, train_mask=node_train_mask, val_mask=node_val_mask, test_mask=node_test_mask)
    sub_dataset = MyOwnDataset(str(args.dataset) + "_subgraph_pr_" + str(args.top) + "_" + str(args.hop) + "_" + str(cur))
    print(sub_dataset.data)

    data = Pyg2Dpr(sub_dataset)
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
        # print(modified_adj.size())
    else:
        modified_adj = model.modified_adj  # modified_adj is a torch.tensor
        # print(modified_adj)
    
    existing_node_num += G_sub.number_of_nodes()
    graph_indicator += 1

    modified_adj = np.array(modified_adj.cpu())
    rc = list(modified_adj.shape) #row & column
    node_id = [] #节点id重制
    file = open(path + 'SubCora_A.txt', 'a')
    for i in range(rc[0]):
        for j in range(rc[1]):
            if modified_adj[i][j]:
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
    
    file = open(path + 'SubCora_graph_indicator.txt', 'a')
    for i in range(G_sub.number_of_nodes()):
        file.write(str(graph_indicator) + "\n")
    file.close()

    file = open(path + 'SubCora_graph_labels.txt', 'a')
    file.write("-1\n")
    file.close()

    node_y = []
    file = open(path + 'SubCora_node_labels.txt', 'a')
    for i in range(rc[0]):
        for j in range(rc[1]):
            if modified_adj[i][j]:
                node_y.insert(node_id.index(i), data_cora_y[i])
                node_y.insert(node_id.index(j), data_cora_y[j])
    for i in range(G_sub.number_of_nodes()):
        file.write(str(node_y[i]) + "\n")
    file.close()

    # pkl.dump(modified_adj, open('poisoned_adj/%s_subgraph_%s_%f_%d_%d_%d_adj.pkl' % (args.dataset, args.method, args.rate, args.top, args.hop, cur), 'wb'))

    existing_node_num += G_sub.number_of_nodes()
    graph_indicator += 1