import csv
import math
from datetime import datetime
from torch_geometric.nn import GATv2Conv
import pandas as pd
from torch_geometric.utils import to_networkx
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import GATConv
from utils.subGraph import create, load_data, subgraph, directed_undirected, sub_subgraph

from torch_geometric.nn import global_mean_pool, GATConv, GATv2Conv, \
    global_max_pool, global_add_pool, GINConv, SAGEConv

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef


class SEBlock1(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x).view(x.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        # print("-----out.shape------")
        # print(out.shape)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = x * out
        return out


class CNN1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNN1, self).__init__()
        self.in_channel = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.pool(out)

        return out


class SE_CNN(nn.Module):
    def __init__(self, in_channels, out_dim, reduction_ratio=16):
        super(SE_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.se_block1 = SEBlock1(16, reduction_ratio)
        self.se_block2 = SEBlock1(32, reduction_ratio)

        self.relu = self.relu = nn.ReLU(inplace=True)
        self.cnn1 = CNN1(16, 16)
        self.cnn2 = CNN1(16, 32)
        self.cnn3 = CNN1(32, 32)
        self.cnn4 = CNN1(32, 32)
        self.bn1 = nn.BatchNorm2d(16)
        self.fc = nn.Linear(32 * 22 * 8, out_dim)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)

        out = self.se_block1(x)
        out = self.cnn1(out)
        out = self.cnn2(out)

        out = self.se_block2(out)
        out = self.cnn3(out)
        out = self.cnn4(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class MultiGAT(torch.nn.Module):


    def __init__(self, in_channels,
                 hidden_channels,
                 out_channels,
                 num_heads,
                 num_layers,
                 concat='False'):
        super(MultiGAT, self).__init__()
        self.num_layers = num_layers

        # Define the input layer.
        self.conv1 = GATConv(in_channels,
                             hidden_channels,
                             concat=True,
                             heads=num_heads,
                             dropout=0.5,
                             bias=True)
        # Define the hidden layers.
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads,
                                      hidden_channels,
                                      concat=True,
                                      dropout=0.5,
                                      heads=num_heads))

        # Define the output layer.
        self.convN = GATConv(hidden_channels * num_heads,
                             out_channels,
                             concat=True,
                             dropout=0.3,
                             heads=num_heads,
                             )
        self.layerNorm = nn.LayerNorm(out_channels)



    def forward(self, x, edge_index):
        # print("--------------")
        input = x
        x = F.relu(self.conv1(x, edge_index))
        x = x + input
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        # print(x.shape)

        x = self.convN(x, edge_index)
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(x + input)
        # x = self.layerNorm(x + input)
        return x

class RegressionModel1(nn.Module):
    def __init__(self, input_dim, reg_hidden_dim, output_dim):
        super(RegressionModel1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = reg_hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, reg_hidden_dim, bias=True)
        # self.fc2 = nn.Linear(reg_hidden_dim, reg_hidden_dim, bias=False)
        # self.fc3 = nn.Linear(reg_hidden_dim, reg_hidden_dim, bias=True)
        self.fc4 = nn.Linear(reg_hidden_dim, output_dim, bias=False)
        # self.attention = Attention(256, 8, 32, 0.5)
        self.dropout1 = nn.Dropout(p=0.5)
        # self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        # x = F.leaky_relu(self.fc2(x))
        # x = self.dropout2(x)
        # # x = F.relu(self.fc1(x))

        # x = F.leaky_relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.dropout2(x)
        x = self.fc4(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_layers, num_heads, output_dim):
        super().__init__()
        #
        #
        """
        编码器   nn.TransformerEncoder
        encoder_layer：用于构造编码器层的类，默认为 nn.TransformerEncoderLayer。
        num_layers：编码器层的数量。默认值为 6。
        norm：归一化模块的类，用于在每个编码器层之间进行归一化，默认为 nn.LayerNorm。
        batch_first：输入张量是否以 batch 维度为第一维。默认为 False。
        dropout：每个编码器层输出之前的 dropout 概率。默认值为 0.1

        编码器层  nn.TransformerEncoderLayer
        d_model：输入特征的维度和输出特征的维度。默认值为 512。
        nhead：多头注意力的头数。默认值为 8。
        dim_feedforward：前馈神经网络的隐藏层大小。默认值为 2048。
        dropout：每个子层输出之前的 dropout 概率。默认值为 0.1。
        activation：前馈神经网络中使用的激活函数类型。默认值为 'relu'。
        normalize_before：是否在每个子层之前进行层归一化。默认值为 False。
        """
        self.transformer_encoder_layer = \
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads,
                                       dim_feedforward=hidden_dim,
                                       dropout=0.2, activation='relu',
                                       # normalize_before=True
                                       )
        self.transformer = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=num_layers,
            # dropout=0.2
        )
        # self.fc = nn.Linear(input_dim, output_dim * num_heads)
        # self.fc = nn.Linear(input_dim, output_dim)
        # self.dropout1 = nn.Dropout(p=0.5)
        # self.norm = nn.LayerNorm(output_dim)
        self.fc1 = nn.Linear(input_dim, num_heads * output_dim)

    def Resforward(self, x):
        se = x
        x = self.transformer(x)
        x = self.norm(x)
        x = F.relu(x + se)
        x = self.transformer(x)
        # x = x.squeeze(1)  # 去除序列长度为1的维度
        # x = self.fc(x + se)  # 将Transformer的输出转换为256维向量
        x = self.dropout1(x)
        x = self.fc(x)
        # x = F.relu(x+se)
        # x = self.fc(x)
        return x

    def forward(self, x):
        # input = x
        input = self.fc1(x)
        # x: (batch_size, sequence_length, input_size)
        x = x.transpose(0, 1)  # 将序列长度放到第一维，变成 (sequence_length, batch_size, input_size)
        x = self.transformer(x)  # Transformer 编码器
        x = x.transpose(0, 1)  # 将序列长度放回到第二维，变成 (batch_size, sequence_length, input_size)
        # x = self.norm(x)
        # x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(x + input)
        # x = x + input
        # x = self.norm(x)
        return x


class TransformerModel1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TransformerModel1, self).__init__()
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=input_size,
                                                                    nhead=5, dim_feedforward=hidden_size)
        self.transformer = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size)
        # self.linear = nn.Linear(hidden_size, output_size)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x is a 3D tensor of shape (batch_size, seq_len, input_size)
        # we will transpose the tensor to have shape (seq_len, batch_size, input_size)
        # print(x.shape)
        x = x.transpose(0, 1)
        # print(x.shape)
        # apply the Transformer to the input tensor
        x = self.transformer(x)
        # print("transformer")
        # print(x.shape)
        # transpose the tensor back to its original shape
        x = x.transpose(0, 1)
        #
        # print(x.shape)
        # # # apply global average pooling on the output tensor
        # print("reshape")
        x = x.reshape(x.shape[0], -1)
        # print(x.shape)
        x = self.avgpool(x)
        # print("avg:")
        # print(x.permute(1, 2, 0).shape)
        # x = self.pooling(x.permute(1, 2, 0))# shape: 253 x 256
        # print("pooling")
        # print(x.shape)
        # x = x.squeeze(-1)  # shape: 253 x 256
        return x
        # return x


# 带挤压和激活的CNN块
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        se = self.squeeze(x).view(batch_size, channels)
        se = self.excitation(se).view(batch_size, channels, 1, 1)
        return x * se


# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=False, batch_first=True, dropout=0.4)

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.res = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        res = self.res(x)
        # x = x.unsqueeze(1)
        # x的维度: (batch_size, sequence_length, input_size)
        # 初始化隐藏状态
        # hx = torch.randn(2, 153, 256)  # 第一个维度是 num_layers*num_directions，第二个维度是 batch_size
        # hx = hx.to(device)

        # 前向传播
        # output, (hn, cn) = self.lstm(x, (hx, hx))
        output, (hn, cn) = self.lstm(x)
        # output的形状为 (1, 153, 256)，通过去掉第一个维度可以得到 (153, 256)
        # output = output.squeeze(0)
        # return output+x
        # return output
        return F.relu(output + res)


class comparsion_model(nn.Module):
    def __init__(self):
        super(comparsion_model, self).__init__()
        # self.SEB = SEBasicBlock(inplanes=1, planes=16)
        # self.convnet = ConvNet_T(output_dim=512)
        self.SE_CNN = SE_CNN(1, 256)
        # self.transformer = mytransformer(256, 256)
        # self.transformer = MultiHeadGAT(256, 256, 64, 3, 4)
        # self.transformer1 = TransformerModel(256,  # T_input_dim,
        #                                     512,  # T_hidden_dim,
        #                                     3,  # T_num_layers,
        #                                     4,  # T_num_heads,
        #                                     64  # T_output_dim
        #                                     )

        self.lstm_model = LSTMModel(256, 256, 2)

        # self.model = DeepWalk(dimensions=256, walk_number=4, walk_length=4)
        #
        # self.node2vec = Node2Vec(self.data.edge_index, embedding_dim=256, walk_length=4, p=1, q=0.8,
        #                          context_size=4, walks_per_node=8, num_negative_samples=5, sparse=True).to(device)
        # self.regression1 = RegressionModel1(1024, 512, 1)  # 回归层
        self.regression1 = RegressionModel1(512, 128, 1)  # 回归层

    def forward(self, data):

        x = self.SE_CNN(data.x)

        # 提取图结构特征
        x_r = self.lstm_model(x)

        m = x
        x, n = create(x_r, data.edge_index, data.edge_index.shape[1])
        # print(x.shape)
        ypre = self.regression1(x)  # 预测结果
        return ypre, m
        # return x


def keep_edges(graph, keep_edges):
    # 从graph里随机保留 keep_edges条边

    num = graph.edge_index.shape[1]
    print(num)
    selected_indices = torch.randperm(num)[:keep_edges]  # 随机选择m个列的索引
    new_tensor = graph.edge_index[:, selected_indices]  # 根据索引选择对应的列
    new_weight = graph.edge_weight[selected_indices]
    # new_edges = graph.edge_index[:, num - keep_edges]
    data = Data(x=graph.x, edge_index=new_tensor, edge_weight=new_weight)
    return data


def split_data(select_data):
    if select_data == "S":
        # path_node_new = "data/SMITH/Data/S_HA_vec_only.csv"
        # path_node_new = "data/SMITH/Data/HA_4_vec.csv"
        path_node_new = "data/SMITH/Data/HA_vec.csv"
        path_edge = "data/SMITH/Data/S_combine.csv"
        path_adj = "data/SMITH/Data/S_adj.npy"
    elif select_data == "I":
        print("IAV")
        path_node_new = "data/IAV_CNN/Data/HA_4_vec.csv"
        path_node_new = "data/IAV_CNN/Data/HA_vec_P.csv"
        path_edge = "data/IAV_CNN/Data/IAV_combine.csv"
        path_adj = "data/IAV_CNN/Data/IAV_adj.npy"

    elif select_data == "W700Log":
        print("W700Log")
        # path_node_new = "data/IAV_CNN/Data/HA_4_vec.csv"
        path_node_new = "data/W/logData700/HA_vec_P.csv"
        path_edge = "data/W/logData700/combine_Pall.csv"
        path_adj = "data/W/logData700/log_700.npy"

    elif select_data == "CW":
        print("CW")
        # path_node_new = "data/IAV_CNN/Data/HA_4_vec.csv"
        path_node_new = "data/W_CFES/HA_vec_P.csv"
        path_edge = "data/W_CFES/combine_P.csv"
        path_adj = "data/W_CFES/adj.npy"

    elif select_data == "Log":
        print("W_Log")
        path_node_new = "data/W/logData/HA_vec_P.csv"
        path_edge = "data/W/logData/log_combine.csv"
        path_adj = "data/W/logData/log_W.npy"

    elif select_data == "CO":
        print("combine")
        path_node_new = "data/C/HA_vec.csv"
        path_node_new = "data/C/Data/HA_vec_P.csv"
        path_edge = "data/C/Data/combine.csv"
        path_adj = "data/C/Data/c_adj.npy"
    elif select_data == "W":
        # path_node_new = "data/W/Data/HA_4_vec.csv"
        path_node_new = "data/W/Data/HA_vec_P.csv"
        # path_edge = "data/W/Data/W_combine.csv"
        path_edge = "data/W/Data/W_combine_log2.csv"
        path_adj = "data/W/Data/W_adj.npy"
    elif select_data == "C":
        path_node_new = "./data/CFES/Data/CFES_HA_vec.csv"
        path_node_new = "./data/CFES/Data/HA_vec_P.csv"
        path_edge = "./data/CFES/CFES_combine.csv"
        path_adj = "./data/CFES/CFES_adj.npy"
    elif select_data == "B":
        path_node_new = "./data/B/Data/HA_4_vec.csv"
        path_node_new = "./data/B/Data/HA_4_vec.csv"
        path_edge = "./data/B/Data/B_combine.csv"
        path_adj = "./data/B/Data/B_adj.npy"
    elif select_data == "4838":
        path_node_new = "./data/4838/HA_vec_P.csv"
        path_node_new = "./data/4838/HA_vec_P.csv"
        path_edge = "./data/4838/combine.csv"
        path_adj = "./data/4838/adj.npy"
    elif select_data == "2005":
        path_node_new = "./data/4838/HA_vec_P.csv"
        path_node_new = "./data/2005/HA_vec_P.csv"
        path_edge = "data/2005/combine_id.csv"
        path_adj = "data/2005/adj.npy"
    elif select_data == "QIU":
        path_node_new = "./data/4838/HA_vec_P.csv"
        path_node_new = "./data/最新/value/HA_vec_P.csv"
        path_edge = "data/最新/value/combine_id.csv"
        path_adj = "data/最新/value/adj.npy"
    elif select_data == "MYH3N2":
        path_node_new = "./data/4838/HA_vec_P.csv"
        path_node_new = "./data/MYDATAH3N2/value/HA_vec_P.csv"
        path_edge = "data/MYDATAH3N2/value/H3N2_combine.csv"
        path_adj = "data/MYDATAH3N2/value/adj.npy"
    else:
        path_node_new = "./data/HA_pre/HA_vec.csv"
        path_edge = "./data/HA_data/Smith/Smithcombine.csv"
        path_adj = "./data/HA_data/adj.npy"
    # -------------------
    # path_node_new = "./data/HA_data/Bedford/Bedford_HA_vec.csv"
    # path_edge = "./data/HA_data/Bedford/Bedford_combine.csv"
    # path_adj = "./data/HA_data/Bedford/Bedford_adj.npy"
    # arr = np.load(path_adj)
    # print(arr)
    # print(arr[1, :])
    # -------------------

    # 加载数据, data 是整体数据
    data = load_data(path_edge, path_node_new)
    # 使用其他方法获取训练数据和测试数据
    print(data)

    save_path = "./data/graph/train_data/{}_P.csv".format(select_data)
    torch.save(data, save_path)  # 保存到指定文件
    # data, te, tr = load_data(path_edge, path_node)
    # data.x = data.x.reshape(253, -1).unsqueeze(1)
    # data.x = data.x.reshape(253, -1).unsqueeze(1)
    print(data.x.shape)
    data.edge_weight = data.edge_weight.unsqueeze(1)
    # data = directed_undirected(data).to(device)
    # drawgraph(data)
    # 获取训练数据
    data = data.to(device)
    data_train = subgraph(data, path_adj, 6, 10).to(device)
    print("用于训练的数据:{}".format(data_train))

    re, re_sub = sub_subgraph(data, data_train, path_adj, 0, 0)
    re = re.to(device)
    data_evaluate = re.to(device)
    # data_evaluate = re.to(device)
    print("用于验证的数据:{}".format(data_evaluate))

    re_all, re_sub1 = sub_subgraph(re, data_train, path_adj, 0, 0)

    # data_train = directed_undirected(data_train)
    # data_evaluate = directed_undirected(data_evaluate).to(device)
    data_test = re.to(device)
    # data_test = directed_undirected(data_test).to(device)
    print("用于测试的数据:{}".format(data_test))
    return data, data_train, data_evaluate, data_test
    # return data


def load_data(path_edge, path_node):
    # path_edge = "../targetdata/graph/combine_Pall.csv"
    df = pd.read_csv(path_edge)
    # 节点对id
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
    print(x.shape)
    data = Data(x=x, edge_index=edge_index.transpose(0, 1), edge_weight=edge_weight)
    return data


def load_vec(HA1_path):
    data = np.genfromtxt(HA1_path)
    # print(data)
    name = data[:, 0:1]  # 毒株名
    vec = data[:, 1:]  # 毒株嵌入，没有卷积过
    print(vec.shape)
    num = vec.shape[0]
    # vec = vec.reshape(num, 100, 327)  # 253 个 100*327 矩阵
    vec = vec.reshape(num, 327, 100)  # 253 个 327*100 矩阵
    # vec = vec.reshape(num, 327, 100)  # 253 个 327*100 矩阵
    # print(vec.shape)
    # print(vec[0])
    # print(vec[0].shape)
    vec = torch.FloatTensor(vec)  # 转换为tensor
    #
    inputs = vec.unsqueeze(1)  # 添加一维(253,1,100,327)  (253,1,327,100)
    return inputs


def printparm(model):
    # 打印每一层的参数量
    total_params = 0
    for name, param in model.named_parameters():
        param_count = param.numel()
        print(f"Layer '{name}' parameter count: {param_count}")
        total_params += param_count

    print("Total number of parameters: ", total_params)


def train_one_epoch(model, optimizer, lr_scheduler, i, epochs, criterion, data_train):
    lambd1 = 0.004
    lambda_l2 = 0.00001  # L2正则化系数
    param_norm = 0.0
    model.train()
    optimizer.zero_grad()
    output, x = model(data_train)
    # regularization_loss = 0
    # for param in model.parameters():
    #     regularization_loss += torch.norm(param, p=2)  # 参数的L2范数

    loss = criterion(output, data_train.edge_weight)
    # loss += regularization_loss*lambda_l2
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    if (i % 100 == 0):
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {i}, Learning rate: {lr}, Loss: {loss.item()}")
    if i == epochs - 1:
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        with open(f'./data/result/CSV/节点嵌入{now}.csv', 'w', newline='') as file:
            print(x.shape)
            print("写入文件./data/result/CSV/节点嵌入.csv")
            writer = csv.writer(file)
            writer.writerows(x.tolist())
        file.close()
    return loss.item(), output  # 返回损失值

def evaluate(model, data_evaluate):
    # print(data_evaluate)
    with torch.no_grad():
        # best_loss = float('inf')
        # patience = 20  # 没有下降的最大轮数
        # counter = 0  # 计数器，记录没有下降的轮数

        model.eval()  # 设置模型为评估模式  需要冻结模型的前面层
        threshold = torch.tensor(2.0).to(device)
        output, x = model(data_evaluate)
        output = output.to(device)

        evalue_loss_mse = F.mse_loss(output, data_evaluate.edge_weight)

        # # 是否提前停止
        # if evalue_loss_mse < best_loss:
        #     best_loss = evalue_loss_mse
        #     counter = 0  # 重置计数器
        # else:
        #     counter += 1
        #     # 判断是否需要提前停止训练
        # if counter >= patience:
        #     print("Training stopped early due to no improvement in loss.")
        #     # torch.save(model.state_dict(), model_save)
        #     exit()

        # 将小于阈值的值设为4，大于等于阈值的值设为0
        output_binary_tensor = torch.where(output <= threshold, torch.tensor(1).to(device), torch.tensor(0).to(device))
        data_evaluate_binary_tensor = torch.where(data_evaluate.edge_weight.to(device) <= threshold,
                                                  torch.tensor(1).to(device), torch.tensor(0).to(device))
        predictions = output_binary_tensor.cpu().numpy().flatten()
        targets = data_evaluate_binary_tensor.cpu().numpy().flatten()
        # 计算准确率
        accuracy = round(accuracy_score(targets, predictions), 5)
        # 计算精确度
        precision = round(precision_score(targets, predictions), 5)
        # 计算召回率
        recall = round(recall_score(targets, predictions), 5)
        # 计算F1值
        f1 = round(f1_score(targets, predictions), 5)
        # 打印结果
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 score:", f1)
        mcc_score = matthews_corrcoef(targets, predictions)
        print("(MCC) Score:", mcc_score)
        # print("MCC:", f1)

        # print(output.shape)
        # print(data_evaluate.edge_weight.shape)
        decimal_places = 5

        evalue_loss_mae = F.l1_loss(output, data_evaluate.edge_weight)
        rmse = torch.sqrt(evalue_loss_mse)
        std = torch.std(data_evaluate.edge_weight - output)

        formatted_rmse = round(rmse.item(), decimal_places)
        formatted_std = round(std.item(), decimal_places)

        edge_weight_cpu = data_evaluate.edge_weight.cpu().numpy()
        output_cpu = (output.cpu().numpy())
        pcc = np.corrcoef(output.cpu().flatten(), data_evaluate.edge_weight.cpu().flatten())[0, 1]
        r2 = r2_score(edge_weight_cpu, output_cpu)

    return round(evalue_loss_mse.item(), 5), round(evalue_loss_mae.item(), 5), output, formatted_rmse, round(pcc,
                                                                                                             5), formatted_std, round(
        r2, 5)
    # return round(evalue_loss_mse.item(), 5), round(evalue_loss_mae.item(), 5), output, formatted_rmse, round(pcc,
    #                                                                                                          5), formatted_std


def train(model, criterion, optimizer, lr_scheduler, epochs, data_train, data_evaluate):
    best_loss = float('inf')
    patience = 60  # 没有下降的最大轮数
    counter = 10  # 计数器，记录没有下降的轮数
    for i in range(epochs):
        train_loss, output_train = train_one_epoch(model, optimizer, lr_scheduler, i, epochs, criterion, data_train)
        # evalue_loss_mse, evalue_loss_mae, output_evalue, rmse, pcc, std, r2 = evaluate(model, data_evaluate)

        if train_loss < best_loss:
            best_loss = train_loss
            counter = 0  # 重置计数器
        else:
            counter += 1
            # 判断是否需要提前停止训练
        if counter >= patience:
            print("Training stopped early due to no improvement in loss.")
            torch.save(model.state_dict(), model_save)
            break
        if (i % 10 == 0):
            # train_losses.append(re)
            print("第{}轮".format(i))
            evalue_loss_mse, evalue_loss_mae, output_evalue, rmse, pcc, std, r2 = evaluate(model, data_evaluate)
            if evalue_loss_mse < best_loss:
                best_loss = evalue_loss_mse
                counter = 0  # 重置计数器
            else:
                counter += 1
                # 判断是否需要提前停止训练
            if counter >= patience:
                print("Training stopped early due to no improvement in loss.")
                torch.save(model.state_dict(), model_save)
                break
            print("PCC： " + str(pcc))
            print("STD： " + str(std))
            print("R2： " + str(r2))
            print("训练偏差： " + str(train_loss))
            print("MSE验证偏差： " + str(evalue_loss_mse))
            print("MAE验证偏差： " + str(evalue_loss_mae))
            print("RMSE验证偏差： " + str(rmse))
            data = [evalue_loss_mse, evalue_loss_mae, output_evalue, rmse, pcc, std, r2]
            # 打开文件并创建CSV写入器
            # with open(filename, 'a', newline='') as file:
            #     writer = csv.writer(file)
            #     # 将字符串列表写入CSV文件的一行
            #     writer.writerow(data)
            # print("CSV file saved successfully.")

            # print("PCC： " + str(pcc))
            # print("P： " + str(p))
            # print("打印模型返回值： " + str(output_train[:10].reshape(1, -1)))
            # print("-----------------")
            # print("打印真实值：    " + str(data_train.edge_weight[:10].reshape(1, -1)))
        if (i % 100 == 0):
            print("打印模型返回值： " + str(output_train[:10].reshape(1, -1)))
            print("-----------------")
            print("打印真实值：    " + str(data_train.edge_weight[:10].reshape(1, -1)))
        if (i == epochs - 1):
            torch.save(model.state_dict(), model_save)

    # print("打印参数变化")
    # print_submodule_params(model)
    # model_weight = "../model_weight/train100_weight.pth"
    # torch.save(model.state_dict(), model_weight)
    # model_all = "../model_weight/train_model.pth"
    # torch.save(model, model_all)
    # plot_all(train_losses, evaluate_losses_mae, evaluate_losses_mse)
    # model_save = './data/result/model_new.pth'
    # torch.save(model.state_dict(), model_save)
    # writer.close()


def test(model_save, data_test, model):
    model_test = model.to(device)
    model_test.load_state_dict(torch.load(model_save))
    print(data_test)

    with torch.no_grad():
        model_test.eval()  # 设置模型为评估模式  需要冻结模型的前面层
        output, x = model_test(data_test)
        output = output.to(device)
        threshold = torch.tensor(2.0).to(device)
        # 将小于等于阈值的值设为4，大于等于阈值的值设为0
        output_binary_tensor = torch.where(output <= threshold, torch.tensor(1).to(device), torch.tensor(0).to(device))
        data_test_binary_tensor = torch.where(data_test.edge_weight.to(device) <= threshold,
                                              torch.tensor(1).to(device), torch.tensor(0).to(device))

        predictions = output_binary_tensor.cpu().numpy().flatten()
        targets = data_test_binary_tensor.cpu().numpy().flatten()

        # 计算准确率
        accuracy = round(accuracy_score(targets, predictions), 5)
        # 计算精确度
        precision = round(precision_score(targets, predictions), 5)
        # 计算召回率
        recall = round(recall_score(targets, predictions), 5)
        # 计算F1值
        f1 = round(f1_score(targets, predictions), 5)

        mcc_score = round(matthews_corrcoef(targets, predictions), 5)

        test_loss_mae = F.l1_loss(output, data_test.edge_weight)
        test_loss_mse = F.mse_loss(output, data_test.edge_weight)
        rmse = torch.sqrt(test_loss_mse)
        edge_weight_cpu = data_test.edge_weight.cpu().numpy()
        output_cpu = output.cpu().numpy()

        # Calculate r2 score
        r2 = r2_score(edge_weight_cpu, output_cpu)
        pcc = np.corrcoef(output.cpu().flatten(), data_test.edge_weight.cpu().flatten())[0, 1]
        # 保留两位小数
        decimal_places = 5
        formatted_mse = round(test_loss_mse.item(), decimal_places)
        formatted_mae = round(test_loss_mae.item(), decimal_places)
        formatted_rmse = round(rmse.item(), decimal_places)
        formatted_r2 = round(r2.item(), decimal_places)
        pcc = round(pcc, 5)

        # 打印结果
        print("===test result==")
        print("测试数据在模型上表现情况")
        print("test==={}".format(output[:20].reshape(1, -1)))
        print('real==={}'.format(data_test.edge_weight[:20].reshape(1, -1)))
        print("===定性结果===")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 score:", f1)

        print("Matthews Correlation Coefficient (MCC) Score:", mcc_score)
        print("===定量结果===")
        std = round(np.std(edge_weight_cpu - output_cpu), 5)
        print("测试数据上的残差标准差是{}".format(std))
        print("测试数据上的PCC{}:".format(pcc))
        print("测试数据上的MAE是{}".format(formatted_mae))
        print("测试数据上的MSE是{}".format(formatted_mse))
        print("测试数据上的RMSE是{}".format(formatted_rmse))
        print("测试数据上的R2是{}".format(formatted_r2))
        # PCC值的取值范围在-1到1之间，其值越接近于1表示两个变量之间的线性相关性越强，
        # 而其值越接近于-1则表示两个变量之间的线性相关性越弱。
        # 同时，P值表示PCC值的显著性水平，其值越小表示相关性越显著。
        # print("测试数据上的PCC是{} ,p是{}".format(pcc, p_value))
        print("======")

        return test_loss_mse, test_loss_mae, r2


def subgraph_s(data):
    data = data.to(device)
    x = data.x.to(device)
    print(x.shape)
    # edge_index = torch.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 1]], dtype=torch.long)
    edge_index = data.edge_index.to(device)
    print(edge_index.shape)
    edge_weight = data.edge_weight.to(device)
    # 定义每个节点的邻居数量
    num_neighbors = 500

    # 采样子图
    sampled_graphs = []
    num_sub = 3
    for node_idx in range(num_sub):
        # node_idx = node_idx.to(device)
        neighbors = edge_index[1][edge_index[0] == node_idx]  # 获取当前节点的邻居节点索引
        num_neighbors_to_sample = min(num_neighbors, neighbors.size(0))  # 限制邻居节点的数量
        sampled_neighbors = neighbors[torch.randperm(neighbors.size(0))[:num_neighbors_to_sample]]  # 随机采样邻居节点
        sampled_neighbors = sampled_neighbors.to(device)

        # 构建子图的节点索引和边索引
        sampled_nodes = torch.cat([torch.tensor([node_idx]).to(device), sampled_neighbors])
        sampled_edge_mask = (torch.isin(edge_index[0], sampled_nodes) &
                             torch.isin(edge_index[1], sampled_nodes))
        sampled_edge_index = edge_index[:, sampled_edge_mask]  # 更新边索引为相对于子图的节点索引
        sampled_edge_weight = edge_weight[sampled_edge_mask]

        # 提取子图的节点特征
        sampled_x = x[sampled_nodes]

        # 创建子图的 Data 对象
        sampled_data = Data(x=sampled_x, edge_index=sampled_edge_index, edge_weight=sampled_edge_weight)
        print("=============")
        print(sampled_data)
        sampled_graphs.append(sampled_data)

    # 打印子图数据
    for i, sampled_data in enumerate(sampled_graphs):
        print(f"Sampled Graph {i + 1}:")
        print("Nodes:", sampled_data.num_nodes)
        print("Edges:", sampled_data.num_edges)
        print("Node features:", sampled_data.x.shape)
        print("Edge indices:", sampled_data.edge_index)
        print()
    return sampled_graphs


import torch_geometric.utils as pyg_utils


def sub_evaluate(data):
    data = data.to(device)
    x = data.x.to(device)
    print(x.shape)
    # edge_index = torch.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 1]], dtype=torch.long)
    edge_index = data.edge_index.to(device)
    print(edge_index.shape)
    edge_weight = data.edge_weight.to(device)


def is_graph_connected(data):

    data = pyg_utils.to_networkx(data).to_directed()

    # 使用NetworkX中的is_strongly_connected函数判断有向图是否是强连通
    is_strongly_connected = nx.is_strongly_connected(data)
    is_strongly_connected = nx.is_weakly_connected(data)
    print("是否是来连通的图：")
    print(is_strongly_connected)

def main(keep_num_edge):

    initial_lr = 0.0001
    # 初始学习率
    decay_factor = 0.9  # 学习率衰减因子
    decay_steps = 100  # 学习率衰减步数
    # epochs = 2001
    epochs = 1601

    model = comparsion_model()
    model.to(device)

    printparm(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0.0005)
    lr_scheduler = StepLR(optimizer, step_size=decay_steps, gamma=decay_factor)


    select_data = "H3N2"

    data, data_train_W, data_evaluate_W, data_test_W = split_data(select_data)


    data_train_W = keep_edges(data_train_W, keep_num_edge)
    print(data_train_W)

    data_train = data_train_W
    is_graph_connected(data_train)
    is_graph_connected(data)
    data_evaluate = data_evaluate_W
    data_test = data_test_W

    data_train = directed_undirected(data_train)
    data_evaluate = directed_undirected(data_test)
    data_test = directed_undirected(data_test)
    print("--------")
    # is_graph_connected(data_train)
    print(data_train)
    print(data_evaluate)
    print(data_test)

    criterion = nn.MSELoss()
    train(model, criterion, optimizer, lr_scheduler, epochs, data_train, data_evaluate)
    test_loss_mse, test_loss_mae, r2 = test(model_save, data_test, model)
    print("=================================================================")
    print("所有测试平均mse: {}".format(test_loss_mse))
    print("所有测试平均mae: {}".format(test_loss_mae))
    print("=================================================================")


if __name__ == '__main__':
    seed = 22
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # 作用是确保在使用CuDNN时的计算结果是确定性的，即相同的输入和参数会产生相同的输出。
    torch.backends.cudnn.deterministic = True
    # 作用是禁用CuDNN的自动调优功能。
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 有没有GPU
    name = "1"
    print(device)
    model_save = 'data/result/model_parma/model_{}.pth'.format(name)
    main(1000)
