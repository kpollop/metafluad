import csv
import math
from datetime import datetime
from torch_geometric.nn import GATv2Conv
import pandas as pd
from torch_geometric.utils import to_networkx, bipartite_subgraph
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error
from torch.optim.lr_scheduler import StepLR
from torch_geometric.nn import GATConv
from utils.subGraph import create, load_data, subgraph, directed_undirected, sub_subgraph

from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, GAE, GATv2Conv, GraphSAGE, GENConv, GMMConv, \
    GravNetConv, MessagePassing, global_max_pool, global_add_pool, GAT, GINConv, GINEConv, GraphNorm, SAGEConv, RGATConv

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from sklearn.metrics import mean_squared_error, r2_score
from torch_geometric.utils.convert import from_networkx
import random
from torch.nn.functional import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef


# ----------

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
        # print("----池化前{}------".format(out.shape))
        out = self.pool(out)
        # print("-----池化后{}-----".format(out.shape))
        return out


class CNN2(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=(3, 3), stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        # self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        #
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(64)
        # self.fc = nn.Linear(32 * 8 * 8, out_dim)
        # self.fc = nn.Linear(32 * 42 * 14, out_dim)
        self.fc = nn.Linear(32 * 11 * 16, out_dim)
        # self.fc = nn.Linear(3456, out_dim)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # print("----池化前{}------".format(out.shape))
        out = self.pool(out)
        # print("-----池化后{}-----".format(out.shape))

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        # print("----池化前{}------".format(out.shape))
        out = self.pool(out)
        # print("-----池化后{}-----".format(out.shape))

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        #
        # out = self.conv3(out)
        # out = self.bn3(out)
        # out = self.relu(out)
        # out = self.conv3(out)
        # out = self.bn3(out)
        # out = self.relu(out)

        # print(out.shape)
        # print("----池化前{}------".format(out.shape))
        out = self.pool(out)

        # out = self.conv4(out)
        # out = self.bn4(out)
        # out = self.relu(out)
        # #
        # # out = self.conv3(out)
        # # out = self.bn3(out)
        # # out = self.relu(out)
        # # out = self.conv3(out)
        # # out = self.bn3(out)
        # # out = self.relu(out)
        #
        # # print(out.shape)
        # # print("----池化前{}------".format(out.shape))
        # out = self.pool(out)

        # print(out.shape)
        # print("-----池化后{}-----".format(out.shape))

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class SECNN(nn.Module):
    def __init__(self, in_channels, out_dim, reduction_ratio=16):
        super(SECNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.se_block = SEBlock1(32, reduction_ratio)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = self.relu = nn.ReLU(inplace=True)
        self.cnn = CNN2(32, out_dim)

    def forward(self, x):
        # print("-------se_block-------")
        # print(x.shape)
        # normalized_shape = (x.size(-1),)
        # normalized_shape = ((x.size(-1),))
        # 归一化 后两维
        # normalized_shape = (327, 100)
        # layer_norm = torch.nn.LayerNorm(normalized_shape)
        # x = layer_norm(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out = self.se_block(x)
        out = self.cnn(out)
        return out


class SE_CNN(nn.Module):
    def __init__(self, in_channels, out_dim, reduction_ratio=16):
        super(SE_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.se_block1 = SEBlock1(16, reduction_ratio)
        self.se_block2 = SEBlock1(32, reduction_ratio)
        # self.se_block3 = SEBlock1(32, reduction_ratio)

        self.relu = self.relu = nn.ReLU(inplace=True)
        self.cnn1 = CNN1(16, 16)
        self.cnn2 = CNN1(16, 32)
        self.cnn3 = CNN1(32, 32)
        self.cnn4 = CNN1(32, 32)
        self.bn1 = nn.BatchNorm2d(16)
        # self.fc = nn.Linear(32 * 11 * 16, out_dim)
        self.fc = nn.Linear(32 * 22 * 8, out_dim)

    def forward(self, x):
        # print("-------se_block-------")
        # print(x.shape)
        # normalized_shape = (x.size(-1),)
        # normalized_shape = ((x.size(-1),))
        # 归一化 后两维
        # normalized_shape = (327, 100)
        # layer_norm = torch.nn.LayerNorm(normalized_shape)
        # x = layer_norm(x)
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)

        out = self.se_block1(x)
        out = self.cnn1(out)
        out = self.cnn2(out)

        out = self.se_block2(out)
        out = self.cnn3(out)
        out = self.cnn4(out)
        # print("+++++++++++")
        # print(out.shape)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        # out = F.relu(out)
        return out



# 输出维度将会是num_heads*out_channel，多头的拼接方式默认是concat,设置False就是取平均值
# 在PyTorch Geometric (PyG)库中，GATConv的concatenation和average两种方式输出的维度是相同的，
# 这是因为在PyG实现中，无论是哪种方式，
# 都是通过将每个注意力头得到的特征向量按照通道方向进行拼接或求平均，
# 然后再进行线性变换和LeakyReLU激函数处理得到输出特征向量
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


class RegressionModel1(nn.Module):
    def __init__(self, input_dim, reg_hidden_dim, output_dim):
        super(RegressionModel1, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = reg_hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, reg_hidden_dim, bias=True)
        # self.fc2 = nn.Linear(reg_hidden_dim, reg_hidden_dim, bias=False)
        # self.fc3 = nn.Linear(reg_hidden_dim, reg_hidden_dim, bias=True)
        self.fc4 = nn.Linear(reg_hidden_dim, output_dim)
        # self.attention = Attention(256, 8, 32, 0.5)
        self.dropout1 = nn.Dropout(p=0.5)
        # self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        # x = F.leaky_relu(self.fc2(x))
        # x = self.dropout2(x)
        # # x = F.relu(self.fc1(x))

        # x = F.leaky_relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.dropout2(x)
        x = self.fc4(x)
        return x


# 定义1D Transformer模型,在 Transformer 模型中，一般不需要在最后加上全连接层。
# 这是因为，Transformer 本身已经包含了一个全连接层
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
        self.fc1 = nn.Linear(input_dim, num_heads * output_dim)
        # self.dropout1 = nn.Dropout(p=0.3)
        # self.norm = nn.LayerNorm(output_dim)

    def Resforward(self, x):
        se = x
        x = self.transformer(x)
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
        input = self.fc1(x)
        # x: (batch_size, sequence_length, input_size)
        x = x.transpose(0, 1)  # 将序列长度放到第一维，变成 (sequence_length, batch_size, input_size)
        x = self.transformer(x)  # Transformer 编码器

        x = x.transpose(0, 1)  # 将序列长度放回到第二维，变成 (batch_size, sequence_length, input_size)
        x = F.relu(x + input)
        # x = x + input
        # x = self.norm(x)
        # x = self.fc(x)
        return x


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


class MetaFluAD(nn.Module):
    def __init__(self, cnn_outdim=256):
        super(ConvNet_transformer, self).__init__()
        # self.Con1D = ConvNet_1D()

        # self.CNN = CNNWithResidual(1, 256)
        # self.SEB = SEBasicBlock(inplanes=1, planes=16)
        # self.convnet = ConvNet_T(output_dim=512)
        self.SE_CNN = SE_CNN(1, cnn_outdim)
        # self.SE_CNN = SECNN(1, 128)
        # self.transformer = mytransformer(256, 256)
        # self.transformer = Attention(256, 8, 32, 0.5)
        self.transformer = TransformerModel(cnn_outdim,  # T_input_dim,
                                            512,  # T_hidden_dim,
                                            2,  # T_num_layers,
                                            4,  # T_num_heads,
                                            64  # T_output_dim
                                            )
        self.ResGAT1 = MultiGAT(cnn_outdim, 128 // 2, 128 // 2, 4, 2, concat='True')

        self.regression1 = RegressionModel1(1024, 512, 1)

    def forward(self, data):
        x = self.SE_CNN(data.x)

        x_t = self.transformer(x.unsqueeze(1)).squeeze(1)

        data_x = x.squeeze(1)

        x_r = self.ResGAT1(data_x, data.edge_index)

        x = torch.cat((x_t.squeeze(1), x_r), dim=1)

        feature = x
        x, n = create(x, data.edge_index, data.edge_index.shape[1])

        ypre = self.regression1(x)  # 预测结果

        return ypre, feature



def keep_edges(graph, keep_edges):
    num = graph.edge_index.shape[1]
    print(num)
    selected_indices = torch.randperm(num)[:keep_edges]
    new_tensor = graph.edge_index[:, selected_indices]
    new_weight = graph.edge_weight[selected_indices]
    data = Data(x=graph.x, edge_index=new_tensor, edge_weight=new_weight)
    return data


def split_data(select_data):
    if select_data == "B-vic":
        path_node_new = "data/Spilt_B/vic/HA_vec_P.csv"
        path_edge = "data/Spilt_B/vic/combine_vic.csv"
        path_adj = "data/Spilt_B/vic/adj.npy"
    elif select_data == "B-yam":
        path_node_new = "data/Spilt_B/yam/HA_vec_P.csv"
        path_edge = "data/Spilt_B/yam/combine_vic.csv"
        path_adj = "data/Spilt_B/yam/adj.npy"
    elif select_data == "H3N2":
        path_node_new = "D:\\Desktop\\metafluad\\metaFluad\\data\\H3N2\\HA_vec.csv"
        path_edge = "D:\\Desktop\\metafluad\\metaFluad\\data\\H3N2\\AH3N2_distance.csv"
        path_adj = "D:\\Desktop\\metafluad\\metaFluad\\data\\H3N2\\H3N2_adj.npy"
    elif select_data == "AH1N1":
        path_node_new = "./data/meta_traindata/AH1N1/HA_vec_P.csv"
        path_edge = "data/meta_traindata/AH1N1/AH1N1_combine.csv"
        path_adj = "data/meta_traindata/AH1N1/adj.npy"
    elif select_data == "H5N1":
        path_node_new = "./data/meta_traindata/CF_AH5N1/HA_vec_P.csv"
        path_edge = "data/meta_traindata/CF_AH5N1/CF_H5N1_combine.csv"
        path_adj = "data/meta_traindata/CF_AH5N1/adj.npy"
    else:
        path_node_new = "./data/HA_pre/HA_vec.csv"
        path_edge = "./data/HA_data/Smith/Smithcombine.csv"
        path_adj = "./data/HA_data/adj.npy"

    data = load_data(path_edge, path_node_new)
    save_path = "./data/graph/train_data/{}_P.csv".format(select_data)
    torch.save(data, save_path)  # 保存到指定文件
    data.edge_weight = data.edge_weight.unsqueeze(1)
    # 获取训练数据
    data = data.to(device)
    data_train = subgraph(data, path_adj, 6, 8).to(device)
    re, re_sub = sub_subgraph(data, data_train, path_adj, 5, 6)
    re = re.to(device)
    data_evaluate = re_sub.to(device)

    re_all, re_sub1 = sub_subgraph(re, re_sub, path_adj, 5, 8)

    data_test = re_all.to(device)

    return data, data_train, data_evaluate, data_test


def load_data(path_edge, path_node):
    df = pd.read_csv(path_edge)
    # 节点对id
    src = torch.Tensor(np.array(df["strainName1"].astype(int).values))
    dst = torch.Tensor(np.array(df["strainName2"].astype(int).values))

    # 边缘
    edge_weight = torch.Tensor(df["distance"].values)

    edge_index = torch.stack([src, dst]).to(torch.int64)
    edge_index = edge_index.transpose(0, 1)

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
    data = np.genfromtxt(HA1_path, delimiter=' ')
    name = data[:, 0:1]  # 毒株名
    vec = data[:, 1:]  # 毒株嵌入，没有卷积过
    print(vec.shape)
    num = vec.shape[0]
    vec = vec.reshape(num, 327, 100)  # 253 个 327*100 矩阵
    vec = torch.FloatTensor(vec)  # 转换为tensor
    inputs = vec.unsqueeze(1)  #
    return inputs

def train_one_epoch(model, optimizer, lr_scheduler, i, epochs, criterion, data_train):
    model.train()
    optimizer.zero_grad()
    output, x = model(data_train)

    # # 计算L2正则化项
    # regularization_loss = 0
    # for param in model.parameters():
    #     regularization_loss += torch.norm(param, p=2)  # 参数的L2范数

    # loss1, loss_t, loss_g = cosloss(x, data_train.edge_weight)
    # loss = F.mse_loss(output, data_train.edge_weight) * lambd1 + lambd2 * loss_t + lambd3 * loss_g

    # loss_1, loss_2 = myloss1(x, data_train.edge_weight)
    # loss = criterion(output, data_train.edge_weight) + lambd *norm
    # loss = criterion(output, data_train.edge_weight) + loss_2 * r2 + loss_1 * r1
    loss = criterion(output, data_train.edge_weight)
    # loss += regularization_loss*lambda_l2
    loss.backward()
    optimizer.step()

    lr_scheduler.step()
    if (i % 100 == 0):
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {i}, Learning rate: {lr}, Loss: {loss.item()}")
    if i == epochs - 1:
        model_save = './data/result/metalearn/CW_PYG1_transformer.pth'
        torch.save(model.state_dict(), model_save)
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        with open(f'./data/result/CSV/节点嵌入{now}.csv', 'w', newline='') as file:
            writer.writerows(x.tolist())
        file.close()
        # pass
    return loss.item(), output


def evaluate(model, data_evaluate):
    with torch.no_grad():

        model.eval()  
        threshold = torch.tensor(2.0).to(device)
        output, x = model(data_evaluate)
        output = output.to(device)

        evalue_loss_mse = F.mse_loss(output, data_evaluate.edge_weight)


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
        # r2 = 0

    return round(evalue_loss_mse.item(), 5), round(evalue_loss_mae.item(), 5), output, formatted_rmse, round(pcc,
                                                                                                             5), formatted_std, round(
        r2, 5)
    # return round(evalue_loss_mse.item(), 5), round(evalue_loss_mae.item(), 5), output, formatted_rmse, round(pcc,
    #                                                                                                          5), formatted_stdF


def train(model, criterion, optimizer, lr_scheduler, epochs, data_train, data_evaluate):
    model.train()
    best_loss = float('inf')
    patience = 80  
    counter = 40  # 计数器，记录没有下降的轮数
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
            print("R2： " + str(r2))
            print("训练偏差： " + str(train_loss))
            print("MSE验证偏差： " + str(evalue_loss_mse))
            print("MAE验证偏差： " + str(evalue_loss_mae))
            data = [evalue_loss_mse, evalue_loss_mae, output_evalue, rmse, pcc, std, r2]
        if (i % 200 == 0):
            print("打印模型返回值： " + str(output_train[:10].reshape(1, -1)))
            print("-----------------")
            print("打印真实值：    " + str(data_train.edge_weight[:10].reshape(1, -1)))
        if (i == epochs - 1):
            torch.save(model.state_dict(), model_save)


def test(model_save, data_test, model):
    model.eval()
    model_test = model.to(device)
    model_test.load_state_dict(torch.load(model_save))
     # 设置模型为评估模式  需要冻结模型的前面层

    with torch.no_grad():
        print(len(data_test.edge_weight))
        print("data_test")
        for item in data_test.edge_weight:
            print(item.item(), end=',')
        print("==")

        output, x = model_test(data_test)
        ###########
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        with open(f'./data/result/CSV/AH3N2节点嵌入_no_dropout{now}.csv', 'w', newline='') as file:
            print(x.shape)
            print("写入文件./data/result/CSV/节点嵌入.csv")
            writer = csv.writer(file)
            writer.writerows(x.tolist())
        file.close()
        ###########
        print(len(output))
        print("output")
        for item in output:
            print(item.item(), end = ',')


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


def sub_evaluate(data):
    data = data.to(device)
    x = data.x.to(device)
    print(x.shape)
    edge_index = data.edge_index.to(device)
    print(edge_index.shape)
    edge_weight = data.edge_weight.to(device)

def selectnodes(data_train):
    ################################
    # 指定要删除的节点列表（这里假设要删除的节点为节点0到节点9）

    number = 25
    nodes_to_remove = random.sample(range(253), number)
    # nodes_to_remove = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # 将PyG Data对象转换为NetworkX图对象
    nx_graph = to_networkx(data_train)

    # 删除指定节点及相关边
    nx_graph.remove_nodes_from(nodes_to_remove)

    # 重新计算剩余节点的映射关系
    node_mapping = {node: idx for idx, node in enumerate(nx_graph.nodes)}

    # 创建子图的边索引和边权重
    subgraph_edge_index = []
    subgraph_edge_weight = []
    subgraph_x = []

    # 遍历原始图的边
    for u, v, w in zip(data_train.edge_index[0], data_train.edge_index[1], data_train.edge_weight):
        u, v = u.item(), v.item()

        # 如果边的起始节点都在剩余节点中，则将边添加到子图中
        if u not in nodes_to_remove and v not in nodes_to_remove:
            u_mapped = node_mapping[u]
            v_mapped = node_mapping[v]
            subgraph_edge_index.append([u_mapped, v_mapped])
            subgraph_edge_weight.append(w.item())

    # 遍历剩余节点的特征
    for node, idx in node_mapping.items():
        if node not in nodes_to_remove:
            subgraph_x.append(data_train.x[node])

    # 创建子图的Data对象
    subgraph_data = Data(
        x=torch.stack(subgraph_x),
        edge_index=torch.tensor(subgraph_edge_index).t().contiguous(),
        edge_weight=torch.tensor(subgraph_edge_weight),
        num_nodes=nx_graph.number_of_nodes()
    ).to(device)

    # 输出子图的节点数量和边的数量
    print("子图节点数量：", subgraph_data.num_nodes)
    print("子图边的数量：", subgraph_data.num_edges)
    print("子图边权重：", subgraph_data.edge_weight)
    ####################3##########3
    subgraph_data.edge_weight = subgraph_data.edge_weight.unsqueeze(1)

    return subgraph_data


def main(keep_num_edge):
    initial_lr = 0.0001
    decay_factor = 0.9
    decay_steps = 100
    epochs = 1601
    model = MetaFluAD()
    # 加载预训练模型的参数
    # path = 'data/result/metalearn/AH5N1.pth'
    # path = 'data/result/metalearn/AH3N2-2.pth'
    # path = 'data/result/metalearn/AH5N1-new.pth'
    # path = 'data/result/metalearn/BH1N1-1.pth'
    # path = 'data/result/metalearn/AH3N2.pth'
    # path = 'data/result/metalearn/AH1N1.pth'
    # path = 'data/result/metalearn/12_5_1.pth'
    # path = 'data/result/metalearn/11_30.pth'
    # # path = 'data/result/metalearn/6_26.pth'
    # path = 'data/result/model_parma/model_1.pth'
    # # # #
    # pretrained_dict = torch.load(path)
    # model_dict = model.state_dict()  # 获取当前模型的state_dict
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    model.to(device)

    printparm(model)
    # optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, weight_decay=0.02)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0.03)
    lr_scheduler = StepLR(optimizer, step_size=decay_steps, gamma=decay_factor)

    select_data = "H3N2"
    # select_data = "H5N1"
    # select_data = "AH1N1"
    # select_data = "BH1N1"
    # select_data = "B-vic"
    # select_data = "B-yam"


    data, data_train_W, data_evaluate_W, data_test_W = split_data(select_data)


    data_train_W = keep_edges(data_train_W, keep_num_edge)
    print(data_train_W)

    data_train = data_train_W
    data_evaluate = data_evaluate_W
    data_test = data_test_W

    data_train = directed_undirected(data_train)
    data_evaluate = directed_undirected(data_evaluate)
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
    print("所有测试平均mae: {}".format(test_loss_mae))
    print("=================================================================")

if __name__ == '__main__':
    seed = 222

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # 作用是确保在使用CuDNN时的计算结果是确定性的，即相同的输入和参数会产生相同的输出。
    torch.backends.cudnn.deterministic = True
    # 作用是禁用CuDNN的自动调优功能。
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 有没有GPU
    name = "H3N2"
    print(device)
    model_save = 'data/result/model_parma/model_{}.pth'.format(name)
    main(1500)
