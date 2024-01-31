from torch_geometric.nn import GATv2Conv
import pandas as pd
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, GAE, GATv2Conv, GraphSAGE, GENConv, GMMConv, \
    GravNetConv, MessagePassing, global_max_pool, global_add_pool, GAT, GINConv, GINEConv, GraphNorm, SAGEConv, RGATConv


class GCN(torch.nn.Module):
    def __init__(self, in_channel, hidden_channels, out_channel):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channel, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channel)
        self.fc = nn.Linear(in_channel, out_channel)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x, edge_index):
        res = self.fc(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.relu(x + res)


class AGNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(AGNNModel, self).__init__()
        self.conv1 = AGNNConv(requires_grad=True)

        self.convs = nn.ModuleList([
            AGNNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 2)
        ])

        self.conv_out = AGNNConv(hidden_channels, out_channels)
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        res = self.lin(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.conv_out(x, edge_index)
        x = F.relu(x + res)
        return x


class RESGATv2(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers, num_heads, dropout=0.3):
        super(RESGATv2, self).__init__()
        self.dropout = dropout

        self.gat_layers = nn.ModuleList()
        # 第一层GAT
        self.gat_layers.append(GATv2Conv(in_features, hidden_features, heads=num_heads[0]))
        # 中间层GAT
        for l in range(1, num_layers - 1):
            self.gat_layers.append(GATv2Conv(hidden_features * num_heads[l - 1], hidden_features, heads=num_heads[l]))
        # 输出层GAT
        self.gat_layers.append(GATv2Conv(hidden_features * num_heads[-2], out_features, heads=num_heads[-1]))

        self.residual_layer = nn.Linear(in_features, out_features * num_heads[-1])

    def forward(self, x, edge_index):
        # x = data.x
        x_res = self.residual_layer(x)
        for layer in self.gat_layers[:-1]:
            x = layer(x, edge_index).flatten(1)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # 最后一层GAT
        x = self.gat_layers[-1](x, edge_index).flatten(1)
        x = F.elu(x)
        # x = F.dropout(x, p=self.dropout, training=self.training)
        # print(x.shape)
        # 残差连接
        # x_res = self.residual_layer(x)
        # print(x_res.shape)
        x = F.relu(x + x_res)
        return x


class MultiGAT(torch.nn.Module):
    """ GATConv
    in_channels：输入特征的维度，即节点的特征维度；
    out_channels：输出特征的维度，即经过卷积后每个节点的特征维度；
    heads：注意力机制中注意力头的数目，默认为 1；
    concat：是否将每个头的输出拼接起来，默认为 True；
    negative_slope：LeakyReLU 中负斜率的值，默认为 0.2；
    dropout：在输出特征上应用 dropout 的概率，默认为 0；
    bias：是否添加偏置，默认为 True；
    **kwargs：其他参数，如指定用于计算注意力权重的函数等。
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, num_layers, concat='True'):
        super(MultiGAT, self).__init__()
        self.num_layers = num_layers

        # Define the input layer.
        self.conv1 = GATConv(in_channels,
                             hidden_channels,
                             concat=concat,
                             heads=num_heads,
                             dropout=0.2,
                             bias=True)

        # Define the output layer.
        self.convN = GATConv(
            hidden_channels * num_heads,
            out_channels,
            concat=concat,
            dropout=0.2,
            heads=num_heads)

        self.fc = nn.Linear(in_channels, num_heads * out_channels)

    def forward(self, x, edge_index):
        res = self.fc(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.convN(x, edge_index)
        x = F.relu(x + res)
        return x


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, aggr='mean'):
        super(GraphSAGE, self).__init__()
        self.num_layers = num_layers
        self.aggr = aggr
        self.convF = SAGEConv(in_channels, hidden_channels, self.aggr)
        self.convs = torch.nn.ModuleList()

        # for i in range(num_layers - 2):
        #     self.convs.append(SAGEConv(hidden_channels, hidden_channels, self.aggr))

        self.convN = SAGEConv(hidden_channels, out_channels, self.aggr)
        self.fc = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x, edge_index):
        input = self.fc(x)
        x = F.relu(self.convF(x, edge_index))
        x = self.dropout(x)
        # x = x + input
        # for conv in self.convs:
        #     x = F.relu(conv(x, edge_index))

        x = self.convN(x, edge_index)
        x = F.relu(x + input)
        return x

