import csv
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from torch_geometric.nn import SAGEConv

from torch_geometric.nn import GATConv

from utils.subGraph import directed_undirected, create
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 有没有GPU
from torch_geometric.data import Data


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 num_layers, num_heads, output_dim):
        super().__init__()
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
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout1 = nn.Dropout(p=0.3)
        self.norm = nn.LayerNorm(output_dim)
    def forward(self, x):
        input = x
        # x: (batch_size, sequence_length, input_size)
        x = x.transpose(0, 1)  #(sequence_length, batch_size, input_size)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # (batch_size, sequence_length, input_size)
        x = F.relu(x + input)
        return x

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

        self.conv4 = nn.Conv2d(16, 16, kernel_size=(3, 5), stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(16)

        self.fc = nn.Linear(32 * 11 * 16, out_dim)
        # self.fc = nn.Linear(3456, out_dim)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)


        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.pool(out)


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
                             dropout=0.3,
                             bias=True)
        # Define the hidden layers.
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads,
                                      hidden_channels,
                                      concat=True,
                                      dropout=0.3,
                                      heads=num_heads))

        # Define the output layer.
        self.convN = GATConv(hidden_channels * num_heads,
                             out_channels,
                             concat=True,
                             dropout=0.3,
                             heads=num_heads)
        self.layerNorm = nn.LayerNorm(out_channels)


    def forward(self, x, edge_index):
        input = x
        x = F.relu(self.conv1(x, edge_index))
        # x = x + input
        for conv in self.convs:
            x = F.relu(conv(x + input, edge_index))
        x = self.convN(x + input, edge_index)
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

        self.fc4 = nn.Linear(reg_hidden_dim, output_dim)
        self.dropout1 = nn.Dropout(p=0.5)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc4(x)
        return x


class ConvNet_transformer(nn.Module):
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
                                            3,  # T_num_layers,
                                            4,  # T_num_heads,
                                            64  # T_output_dim
                                            )
        # self.transformer = TransformerModel(128,  # T_input_dim,
        #                                     256,  # T_hidden_dim,
        #                                     3,  # T_num_layers,
        #                                     4,  # T_num_heads,
        #                                     32  # T_output_dim
        #                                     )
        self.ResGAT = MultiGAT(cnn_outdim, 128 // 2, 128 // 2, 4, 2, concat='True')
        # self.ResGAT1 = GraphSAGE1(256, 256, 256, 2, aggr='mean')

        self.regression = RegressionModel1(1024, 512, 1)  #


    def forward(self, data):
        x = self.SE_CNN(data.x)

        x_t = self.transformer(x.unsqueeze(1)).squeeze(1)

        data_x = x.squeeze(1)

        x_r = self.ResGAT1(data_x, data.edge_index)


        x = torch.cat((x_t.squeeze(1), x_r), dim=1)

        m = x
        x, n = create(x, data.edge_index, data.edge_index.shape[1])

        ypre = self.regression(x)
        # print(data.x.shape)
        return ypre, x



def to_tensor(x, l=False):
    """
    Convert a numpy array to torch tensor.
    """
    t = torch.Tensor(x)
    if l:
        t = torch.LongTensor(x)
    if torch.cuda.is_available():
        return torch.autograd.Variable(t).cuda()
    return torch.autograd.Variable(t)


def inner_train_step(model, criterion, optimizer, train, inner_iters):
    lambd1 = 0.004
    """
    Inner training step procedure.
    """
    for i in range(int(inner_iters)):
        optimizer.zero_grad()
        model.train()
        ypred, x = model(train)

        loss = criterion(ypred, train.edge_weight)
        loss.backward()
        optimizer.step()


def meta_train_step(train_set, model, criterion, optimizer,
                    inner_iters,  # Number of iterations in the inner loop
                    meta_step_size,  # Learning rate for meta update
                    meta_batch_size):  # Batch size for the outer loop
    """
    Meta training step procedure.
    """
    # Copy parameters for parameter update
    weights_original = deepcopy(model.state_dict())
    new_weights = []
    for _ in range(len(train_set)):

        sampled_tasks = np.random.choice(len(train_set), meta_batch_size, replace=True)
        for task in sampled_tasks:
            train_task = train_set[task].to(device)
            inner_train_step(model, criterion, optimizer, train_task, inner_iters)

        new_weights.append(deepcopy(model.state_dict()))
        model.load_state_dict({name: weights_original[name] for name in weights_original})

    ws = len(new_weights)
    fweights = {name: new_weights[0][name] / float(ws) for name in new_weights[0]}
    for i in range(1, ws):
        # cur_weights = deepcopy(model.state_dict())
        for name in new_weights[i]:
            fweights[name] += new_weights[i][name] / float(ws)

    model.load_state_dict({name:
                               weights_original[name] +
                               ((fweights[name] - weights_original[name])
                                * meta_step_size) for
                           name in weights_original})


def evaluate(val_set, model, criterion):
    mse_loss = 0
    mae_loss = 0
    with torch.no_grad():
        model.eval()
        for i in range(len(val_set)):
            evaluate = val_set[i].to(device)
            ypred, _ = model(evaluate)
            # ypred = ypred.squeeze(1)
            mse_loss += criterion(ypred, evaluate.edge_weight)
            mae_loss += F.l1_loss(ypred, evaluate.edge_weight)
        mse_loss = mse_loss / len(val_set)
        mae_loss = mae_loss / len(val_set)
        rmse = torch.sqrt(mse_loss)
        return mse_loss, mae_loss, rmse


def train(train_set, val_set, test_data, model, criterion, optimizer, inner_iters, meta_iters,
          meta_step_size, meta_batch_size):
    """
    Meta training.
    """
    best = 1000
    # now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # model_save = f'./data/result/metalearn/metalearn_model{now}.pth'
    # model_save = './data/result/metalearn/6_26.pth'
    model_save = './data/result/metalearn/AH5N1-new.pth'
    # Reptile training loop
    for i in range(meta_iters):
        # print("meta_iters={}".format(i))
        frac_done = float(i) / meta_iters
        current_step_size = meta_step_size * (1. - frac_done)
        meta_train_step(train_set, model, criterion, optimizer,
                        inner_iters, current_step_size, meta_batch_size)
        # print("meta_iters={}".format(i))
        if i % 50 == 0:
            print("nmeta_iters={}验证结果如下".format(i))
            mse, mae, r_mse = evaluate(val_set, model, criterion)
            # mse, mae, r_mse = evaluate(test_data, model, criterion)
            if r_mse < best:
                best = r_mse
            # else:
            # now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # model_save = f'./data/result/metalearn/metalearn_model_{now}.pth'
            # torch.save(model.state_dict(), model_save)
            print("平均mse={}".format(mse))
            print("平均r_mse={}".format(r_mse))
            print("平均mae={}".format(mae))
            print("current_step_size={}".format(current_step_size))

        if i % 100 == 0:
            print("=========================")
            evaluate_task(test_data, model, criterion, optimizer)

            # model_save1 = './data/result/metalearn/metalearn_model{}.pth'.format(i)
            # torch.save(model.state_dict(), model_save1)
        if i == meta_iters - 1:
            # now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # model_save = './data/result/metalearn/metalearn_model.pth'
            torch.save(model.state_dict(), model_save)

        # Periodically evaluate
        # if i % (eval_interval * 2) == 0:
        #     val_acc = test(val_set, model, criterion, optimizer, 200)
        #     print("accuracy on val_set: {}".format(val_acc))
        #     torch.save(model.state_dict(), './models/model_r_{}.pth'.format(i))
        #     if val_acc <= best:
        #         torch.save(model.state_dict(), './models/model_best_{}.pth'.format(i))
        #         best = val_acc


def test(test_data):
    model = ConvNet_transformer().to(device)
    path = './data/result/metalearn/metalearn_model_new1.pth'
    path = './data/result/metalearn/BH1N1.pth'
    # torch.load(model, path)
    model.load_state_dict(torch.load(path))
    print("Evaluate mode")
    with torch.no_grad():
        model.eval()
        pred, m = model(test_data)
        print(pred[:10].reshape(1, -1))
        print(test_data.edge_weight[:10].reshape(1, -1))
    mse = F.mse_loss(pred, test_data.edge_weight)
    mae = F.l1_loss(pred, test_data.edge_weight)
    rmse = torch.sqrt(mse)
    return mse, mae, rmse


def main():
    seed = 222
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # This ensures that when using CuDNN, the computational results are deterministic,
    # meaning that the same inputs and parameters will produce the same outputs.
    torch.backends.cudnn.deterministic = True
    # This disables the automatic tuning feature of CuDNN.
    torch.backends.cudnn.benchmark = False

    initial_lr = 0.0001  # Initial learning rate

    decay_factor = 0.9  # Learning rate decay factor
    decay_steps = 100  # Steps for learning rate decay
    model = ConvNet_transformer(cnn_outdim=256).to(device)
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0.0002)
    # lr = StepLR(optimizer, step_size=decay_steps, gamma=decay_factor)

    inner_iters = 4
    meta_iters = 1001
    meta_step_size = 0.001
    meta_batch_size = 1
    # eval_interval = 40

    data_list = []

    # path_I = "data/meta_traindata/AH1N1_graph.csv"
    # data_I = torch.load(path_I)
    # data_I.edge_weight = data_I.edge_weight.unsqueeze(1)
    # data_list.append(data_I)

    # path_I = "data/meta_traindata/WandMy_graph.csv"
    # data_I = torch.load(path_I)
    # data_I.edge_weight = data_I.edge_weight.unsqueeze(1)
    # data_list.append(data_I)
    #
    path_I = "data/meta_traindata/AH1N1_graph.csv"
    data_I = torch.load(path_I)
    data_I.edge_weight = data_I.edge_weight.unsqueeze(1)
    data_list.append(data_I)

   
    # # path_I = "data/meta_traindata/AH5N1-2_graph.csv"
    # # data_I = torch.load(path_I)
    # # data_I.edge_weight = data_I.edge_weight.unsqueeze(1)
    # # data_list.append(data_I)

    # # #
    # path_I = "data/meta_traindata/AH1N1_graph.csv"
    # data_I = torch.load(path_I)
    # data_I.edge_weight = data_I.edge_weight.unsqueeze(1)
    # data_list.append(data_I)
    # # # #
    path_I = "data/meta_traindata/CF_AH5N1_graph.csv"
    data_I = torch.load(path_I)
    data_I.edge_weight = data_I.edge_weight.unsqueeze(1)
    data_list.append(data_I)
   

   
    train_data = data_list
    
    ###########
    print("+++++++++")
    print(train_data)
    print("+++++++++")
    print("test=============================test")
    # path = 'data/meta_traindata/AH1N1_graph.csv'
    # path = 'data/meta_traindata/AH5N1-2_graph.csv'
    path = 'data/meta_traindata/CF_AH5N1_graph.csv'
    # path = 'data/meta_traindata/WandMy_graph.csv'
    test_data = torch.load(path).to(device)
    test_data.edge_weight = test_data.edge_weight.unsqueeze(1)
    test_data = directed_undirected(test_data)

    print(test_data)

    val_data = []
    val_data.append(test_data)

    train(train_data,
          val_data,
          test_data,
          model,
          criterion,  # Loss function
          optimizer,  # Optimizer
          inner_iters,  # Number of inner updates
          meta_iters,  # Number of iterations
          meta_step_size,
          meta_batch_size
          )

    mse, mae, rmse = test(test_data)
    print("mse={}".format(mse))
    print("rmse={}".format(rmse))
    print("mae={}".format(mae))
    print("end=============================end")


def evaluate_task(test_data, model, criterion, optimizer):
    model_evaluate = ConvNet_transformer(cnn_outdim=256).to(device)
    model_evaluate.load_state_dict(model.state_dict())
    criterion = criterion
    initial_lr = 0.0005
    optimizer = torch.optim.Adam(model_evaluate.parameters(), lr=initial_lr)
    # optimizer = optimizer
    epoches = 201
    for i in range(epoches):
        model_evaluate.train()
        optimizer.zero_grad()
        pred, m = model_evaluate(test_data)
        loss = criterion(pred, test_data.edge_weight)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("test_data ==> loss = {}".format(loss))
            print("test_data ==> rmse = {}".format(torch.sqrt(loss)))
            print(pred[:10].reshape(1, -1))
            print(test_data.edge_weight[:10].reshape(1, -1))
    print("!!!!!!!!!!!!!!!evaluate_task end!!!!!!!!!!!!!!!")


if __name__ == '__main__':
    main()
