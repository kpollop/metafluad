import random
import networkx as nx
import numpy as np
import pandas as pd
from torch_geometric.data import Data
import torch
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils.convert import from_networkx


# from torch_geometric.utils import to_undirected


def create(x, edge_index, num_edge):
    features = []
    for i in range(num_edge):
        # print(edge_index)
        m = edge_index[0][i]
        n = edge_index[1][i]
        # print(m, n)
        # print(x[m], x[n])
        feature = torch.flatten(torch.cat((x[m], x[n])))
        # feature = torch.flatten((x[m] + x[n]))
        # print(feature)
        features.append(feature)
        # if i == 0:
        #     a = feature
        #     continue
        # a = torch.cat(a, feature.unsqueeze(0), dim=0)

    return torch.stack(features), features

def load_vec(HA1_path):
    data = np.genfromtxt(HA1_path)
    name = data[:, 0:1]
    vec = data[:, 1:]
    # print(vec.shape)
    num = vec.shape[0]
    # vec = vec.reshape(num, 100, 327)
    vec = vec.reshape(num, 327, 100)
    # print(vec.shape)
    # print(vec[0])
    # print(vec[0].shape)
    vec = torch.FloatTensor(vec)
    #
    inputs = vec.unsqueeze(1)  #  (253,1,327,100)
    return inputs


def load_data(path_edge, path_node):
    # path_edge = "../targetdata/graph/combine_Pall.csv"
    df = pd.read_csv(path_edge)

    src = torch.Tensor(np.array(df["strainName1"].astype(int).values))
    dst = torch.Tensor(np.array(df["strainName2"].astype(int).values))

    # 边缘
    edge_weight = torch.Tensor(df["distance"].values)

    edge_index = torch.stack([src, dst]).to(torch.int64)
    edge_index = edge_index.transpose(0, 1)

    # print("打印边")
    # print(edge_index.shape)
    # print(edge_index)

    # 节点信息
    if path_node == "":
        path_node = "../targetdata/tensor/tensor_data_100.npy"
        x = torch.Tensor(np.load(path_node))
        print("单独卷积")
    print("节点信息来自{}文件,默认来自../targetdata/tensor/tensor_data_100.npy".format(path_node))
    x = load_vec(path_node)
    # print(x.shape)
    # print(x[:10])

    # 将节点特征、边和权重组成一个Data对象
    # data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    # print(data.edge_index)

    # 随机选择一部分和节点边作为测试集，剩余边和节点作为训练集
    # num_edge_index = len(src)  # 一共有多少条边
    # print(num_edge_index)
    # test_size = int(num_edge_index * 0.2)  # 测试集占比20%  6375
    # 不能将所有的边全部使用，计划使用10%的边训练。
    # test_size = int(num_edge_index // 20 * 1)  # 测试集占比95%, 训练集5%
    # print(test_size)

    # rand_idx = torch.randperm(test_size * 5)
    # rand_idx = torch.randperm(test_size * 20)

    # select_idx = rand_idx[:test_size]

    # test_edge = edge_index[select_idx]
    # test_weight = edge_weight[select_idx]
    #
    # remaining_indices = torch.tensor(list(set(range(edge_index.size(0))) - set(select_idx.tolist())))

    # train_edge = edge_index[remaining_indices]
    # train_weight = edge_weight[remaining_indices]

    # 构造测试集的边索引和节点特征
    # test_x = x
    # test_edge_index = test_edge

    # 构造训练集的边索引和节点特征
    # train_x = x
    # train_edge_index = train_edge

    # 构造测试集和训练集的Data对象
    # test_data = Data(x=test_x, edge_index=test_edge_index.transpose(0, 1), edge_weight=test_weight)
    # train_data = Data(x=train_x, edge_index=train_edge_index.transpose(0, 1), edge_weight=train_weight)
    data = Data(x=x, edge_index=edge_index.transpose(0, 1), edge_weight=edge_weight)

    # test_loader = DataLoader(test_data, batch_size=1)
    # train_loader = DataLoader(train_data, batch_size=1)
    # return  test_loader, train_loader
    # print(data.x.shape)
    # print(data.edge_index.shape)
    # print(data.edge_weight.shape)
    # return data, test_data, train_data
    return data

def keep_edges(graph, num_keep):
    all_edges = list(graph.edges())
    select_edges = random.sample(all_edges, num_keep)

    new_graph = nx.Graph()
    new_graph.add_edges_from(select_edges)
    return new_graph

def subgraph(data, path_adj, min, max):

    G = to_networkx(data)
    random.seed(42)

    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        # print(neighbors)
        num_edges_to_keep = random.randint(min, max)
        # print(num_edges_to_keep)
        if len(neighbors) > num_edges_to_keep:
            excess_edges = random.sample(neighbors, len(neighbors) - num_edges_to_keep)
            for neighbor in excess_edges:
                G.remove_edge(node, neighbor)


    subgraph = from_networkx(G)



    if path_adj == "":
        path_adj = "../targetdata/graph/adj.npy"
    adj_matrix = np.load(path_adj, allow_pickle=True)
    edge_weight = torch.Tensor(adj_matrix[subgraph.edge_index[0, :], subgraph.edge_index[1, :]]).unsqueeze(1)


    subgraph = Data(x=data.x, edge_index=subgraph.edge_index, edge_weight=edge_weight)

    return subgraph


def sub_subgraph(source_graph, sub, path_adj, min, max):


    x = source_graph.x
    data = to_networkx(source_graph)
    sub = to_networkx(sub)
    residualG = nx.difference(data, sub)


    residualG = from_networkx(residualG)

    print(type(residualG))

    adj_matrix = np.load(path_adj)


    edge_weight = torch.Tensor(adj_matrix[residualG.edge_index[0, :], residualG.edge_index[1, :]]).unsqueeze(1)

    residualG = Data(x=residualG.x, edge_index=residualG.edge_index, edge_weight=edge_weight)


    res_sub = subgraph(residualG, path_adj, min, max)

    return residualG, res_sub




def create_subgraph(num, path_adj, save_path, data, sub, min, max):
    for i in range(num):
        residualG, res_sub = sub_subgraph(data, sub, path_adj, min, max)
        torch.save(res_sub, save_path.format(i))
        data = residualG
        sub = res_sub



# def createsubgraph(path_node, path_edge, path_adj, min, max):
#
#     data, train_data, test_data = load_data(path_edge, path_node)
#
#     sub = subgraph(data, path_adj, min, max)#
#     return sub, data


"""path_edge = "../data/result/combine_Pall.csv"
path_node = "../targetdata/tensor/tensor_data_100.npy"
path_node_new = "../targetdata/allHA/HA_vec1.csv"
path_adj = "../targetdata/graph/adj.npy"
source, tr, te = load_data(path_edge, path_node_new)
sub = subgraph(source, path_adj, 1, 5)

# re, re_sub = sub_subgraph(source, sub, path_adj, 2, 5)
# re1, re1_sub = sub_subgraph(re, re_sub, path_adj, 2, 6)
# 
# print(source)
# print(sub)
# print(re)
# print(re_sub)
# print(re1)
# print(re1_sub)
# print("00000")
graph = []
for i in range(10):
   
    re, sub_res = sub_subgraph(source, sub, path_adj, 2, 6)
    graph.append(sub_res)
    source = re
    sub = sub_res

print(len(graph))
print(graph)
"""
# sub, data = createsubgraph(path_node_new, path_edge, path_adj, 2, 5)
# print(type(sub))
# print(sub.edge_index.shape)
# print(data.x.shape)
# print(sub.x.shape)
# print(sub.edge_weight.shape)
# print(sub.edge_weight[0]
# subgraph = from_networkx(G)
# G = to_networkx(data)

# print("---------------")
# print(residualG)
#
# print(residualG.x.shape)
#
# print(residualG.edge_index.shape)

# print(residualG.edge_weight[:3])
# print(residualG.x)



"""import numpy as np

data = np.genfromtxt('data.csv', delimiter=',')

np.save('data.npy', data)
"""


# -------------------------------
def all_subgraph():
    min = 3
    max = 5
    num = 5

    path_edge = "../data/AH3N2/AH3N2_combine.csv"
    # path_node = "../targetdata/tensor/tensor_data_100.npy"
    path_node_new = "../data/AH3N2/HA_vec_P.csv"
    path_adj = "../data/AH3N2/adj.npy"


    data = load_data(path_edge, path_node_new)
    # data = data.to(device)
    sub = subgraph(data, path_adj, 0, 1)
    graph = []
    for i in range(num):
        path_adj = "../data/AH3N2/adj.npy"
        # save_path = '../data/HA_data/Smith/subgraph/'
        residualG, res_sub = sub_subgraph(data, sub, path_adj, min, max)
        print(res_sub)
        torch.save(res_sub, '../data/meta_task/H3N2subgraph_P{}.csv'.format(i))
        graph.append(res_sub)
        data = residualG
        sub = res_sub


# all_subgraph()

# for i in range(70):
#     path = '../data/HA_data/Smith/subgraph/subgraph{}'.format(i)
#     data = torch.load(path)
#     print(data)
#     sum += data.edge_weight.shape[0]
#     print(data.edge_weight[:3])
#     print(sum)


# path_edge = "../data/HA_data/Bedford/Bedford_combine.csv"
# # path_node_new = "./data/HA_pre/HA_vec.csv"
# path_node_new = "../data/HA_data/Bedford/Bedford_HA_vec.csv"
# Bdata, s, t = load_data(path_edge, path_node_new)
# print(Bdata)
#
# if Bdata.is_directed:
#     print("This is a directed graph.")
# else:
#     print("This is an undirected graph.")
# from torch_geometric.utils import to_undirected
# from torch_geometric.transforms import to_undirected
#

# undirected_data = to_undirected(Bdata)
#
# if undirected_data.is_directed:
#     print("This is a directed graph.")
# else:
#     print("This is an undirected graph.")
#
# print(undirected_data)
# # undirected_nx_graph = to_undirected(directed_nx_graph)

# ----------------
#
# path = "../data/HA_data/Smith/subgraph/subgraph69"
# data = torch.load(path)
# print(data)


def directed_undirected(data):
    edge_index = torch.cat([data.edge_index, data.edge_index.flip([0])], dim=1)
    edge_weight = torch.cat([data.edge_weight, data.edge_weight])
    un_data = Data(x=data.x, edge_index=edge_index, edge_weight=edge_weight)

    # print(un_data)
    return un_data

def createmetatrain(path_edge, path_node_new):
    data = load_data(path_edge, path_node_new)
    print(data)
    torch.save(data, '../data/meta_traindata/NEW_H3N2_graph.csv')
    # data = data.to(device)

# path_edge = "../data/AH3N2/AH3N2_combine.csv"
# path_node_new = "../data/AH3N2/HA_vec_P.csv"

# createmetatrain(path_edge, path_node_new)