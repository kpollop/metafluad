import csv
import os
from random import random

import networkx as nx
import torch
from gensim.models import Word2Vec
import pandas as pd
import pandas as pd
import math
import numpy as np
import scipy.sparse as sp
# def cc1():
#     # 读取原始文件
#     df = pd.read_csv('./data/W/h3n2_HI_mini.csv')
#
#     # 存储新的数据行
#     new_rows = []
#
#     # 遍历每一行
#     for i, row in df.iterrows():
#         # 取出当前行的virus和reference
#         virus, reference = row['virus'], row['reference']
#         # 在剩余行中找到符合条件的行
#         sub_df = df[(df['virus'] == virus) & (df['reference'] == reference) & (df.index != i)]
#         if not sub_df.empty:
#             # 如果有符合条件的行，则将当前行与找到的行进行合并，并计算平均值
#             new_row = pd.concat([row, sub_df], axis=0)
#             new_row['titre'] = new_row['titre'].astype(float).mean()
#             # 仅保留合并后的第一行
#             new_row = new_row.iloc[0]
#             new_rows.append(new_row)
#         else:
#             # 如果没有符合条件的行，则直接将当前行加入新的数据行
#             new_rows.append(row)
#
#     # 将所有新行拼接为一个新的DataFrame
#     new_df = pd.concat(new_rows, axis=1).T
#     # 保存新文件
#     new_df.to_csv("data/W/average_not_file.csv", index=False)
#
#
# cc1()

import pandas as pd

from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx

def create_3gram(sourcefile, gramfile):
    HA1 = []
    with open(sourcefile, encoding='utf-8') as f:
        reader = csv.reader(f)
        context = [line for line in reader]
        # del context[0]
    f.close()

    with open(gramfile, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for line in (context):
            line = ','.join(line)
            line = line.split(",")
            # print(len(line[2]) - 2)
            for i in range(len(line[1]) - 2):
                HA1.append(line[1][i:i + 3])
                row = [line[0], HA1]
            HA1 = []
            writer.writerow(row)
    f.close()

sourcefile = '../data/Spilt_B/yam/HA_Sequence_yamaga.csv'
gramfile = '../data/Spilt_B/yam/Sequence_vic_3.csv'
# create_3gram(sourcefile, gramfile)

def _3gramtovec(gramfile, _gram_vec, id_HA1, root_path):
    # 将 gram与向量的对应转化为字典
    df = pd.read_csv(_gram_vec)
    gram_vec = df[["word", "vec"]].set_index("word").to_dict(orient='dict')["vec"]
    # print(gram_vec.get("TGM"))
    print("长度{}".format(len(gram_vec)))
    df = pd.read_csv(id_HA1)
    namelist = df["HA"].tolist()
    HA_vec = []
    h_v = {}
    with open(gramfile, encoding="utf-8") as f:
        reader = csv.reader(f)
        context = [line for line in reader]
        for j in range(len(context)):
            test = ','.join(context[j])
            test = test.split(",")
            count = 0
            print(test)
            for i in range(len(test) - 1):
                if gram_vec.get(test[i + 1]) == None:
                    print(gram_vec.get('<unk>'))
                    HA_vec.append(gram_vec.get('<unk>'))
                    print("找不到")
                    # print(test[i + 1])
                    # break
                    print(test[i + 1])
                    print(j)
                else:
                    HA_vec.append(gram_vec.get(test[i + 1]))
                count = count + 1
            h_v[str(namelist[j])] = HA_vec
            HA1name = os.path.join(root_path, "HA_{}.csv".format(j))
            test = pd.DataFrame(data=h_v)  #
            test.to_csv(HA1name)

            h_v = {}
            HA_vec = []
            if count != 327:
                print("错误")
                break
                print(j)
    f.close()


HA_3gram_3 = "../data/Spilt_B/yam/Sequence_vic_3.csv"  # 由 create_3gram产生的文件处理后
gram_vec = "../data/protVec_100d_3grams_[].csv"  # 该文件由skipgram产生
id_HA1 = "../data/Spilt_B/yam/HA_Id_yamaga.csv"
root_path = '../data/Spilt_B/yam/combine_Pall'

def HA_vec(sourcefile, target_file):
    df = pd.read_csv(sourcefile)
    print(df.shape)
    print(df.columns[1])
    print(df.columns[0])

    # HA_evc保存文件地址
    # HA_evc = "./targetdata/allHA/HA_vec{}.csv".format(i)
    # HA_evc = "../data/SMITH/Data/HA_4_vec.csv"

    # HA_evc = "../data/W/Data/HA_vec_P.csv"
    name = (df.columns)
    print(name)
    s = df[name[1]]
    print("len(s)--{}".format(len(s)))
    # print(len(s[0]))
    row = ""
    count = 0
    with open(target_file, "a+", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        for i in range(len(s)):
            if i == 0:
                # 将毒株名写入首行
                row = name[1] + " "
            # 清除中括号和逗号
            row += str(s[i]).strip("[").strip("]") + " "
            row = row.replace(",", "")
            count = count + 1
        if count != 327:
            print("错误{}".format(count))
        print("长度------{}".format((row)))
        writer.writerow([row])

def vec_to_array(size):
    for i in range(size):
        # sourcefile = "../data/CFES/all/HA_{}.csv".format(i)
        # sourcefile = "../data/W/Data/combine_Pall/HA_{}.csv".format(i)
        sourcefile = "../data/Spilt_B/yam/combine_Pall/HA_{}.csv".format(i)
        # target_file = "../data/最新/value/combine_P.csv"
        target_file = "../data/Spilt_B/yam/HA_vec_P.csv"
        HA_vec(sourcefile, target_file)

def create_adj(cmbine_path, save_path1, save_path2, number):
    df = pd.read_csv(cmbine_path)
    print(df)

    # data = df["distance"].values.round(8)
    data = df["distance"].values
    # data_value = data.reshape(data.size)
    row = df['strainName1'].values
    # row_data = row.reshape(row.size)
    col = df["strainName2"].values
    # col_data = col.reshape(col.size)
    # if number == 1:
    #     n_rows = n_cols = 253
    # elif number == 0:
    #     n_rows = n_cols = 402
    # else:
    #     n_rows = n_cols = 139

    n_rows = n_cols = number

    print(n_cols)
    print(n_rows)
    matrix = sp.coo_matrix((data, (row, col)), shape=(n_rows, n_cols))
    # print(matrix)
    # 将稀疏矩阵转换为密集矩阵
    dense_matrix = matrix.toarray()
    np.save(save_path1, dense_matrix)

    # 打印生成的密集矩阵
    print(dense_matrix.shape)
    print(dense_matrix)
    # print(dense_matrix[392, :])
    data = np.load(save_path1)

    # 将数据保存为csv格式
    np.savetxt(save_path2, data, delimiter=',')
    return dense_matrix


combine_path = "../data/Spilt_B/yam/combine_vic.csv"
save_path1 = "../data/Spilt_B/yam/adj.npy"
save_path2 = "../data/Spilt_B/yam/adj.csv"
number = 31

re = create_adj(combine_path, save_path1, save_path2, number)

def load_vec(HA1_path):
    data = np.genfromtxt(HA1_path)
    print(data.shape)
    name = data[:, 0:1]  # 毒株名
    vec = data[:, 1:]  # 毒株嵌入，没有卷积过
    # print(vec.shape)
    num = vec.shape[0]
    # vec = vec.reshape(num, 100, 327)  # 253 个 100*327 矩阵
    vec = vec.reshape(num, 327, 100)  # 253 个 327*100 矩阵
    # print(vec.shape)
    # print(vec[0])
    # print(vec[0].shape)
    vec = torch.FloatTensor(vec)  # 转换为tensor
    #
    inputs = vec.unsqueeze(1)  # 添加一维(253,1,100,327)  (253,1,327,100)
    return inputs


def createH_vec():

    HA_3gram_3 = "../data/IAV_H1N1/HA_merged_3.csv"  # 由 create_3gram产生的文件处理后
    gram_vec = "../data/protVec_100d_3grams_[].csv"  # 该文件由skipgram产生
    id_HA1 = "../data/IAV_H1N1/HA.csv"
    root_path = '../data/IAV_H1N1/combine_Pall'
    _3gramtovec(HA_3gram_3, gram_vec, id_HA1, root_path)

    # size = 237
    size = 170

    # def vec_to_array(size):
    for i in range(size):
        sourcefile = root_path + "/HA_{}.csv".format(i)
        target_file = "../data/2005/HA_vec_P.csv"
        HA_vec(sourcefile, target_file)

def HAtodic():
    # 读取CSV文件1和CSV文件2
    file1_df = pd.read_csv('../data/MYlog2/id_HA.csv')
    file2_df = pd.read_csv('../data/MYlog2/combine_se.csv')

    # 创建一个字典，将CSV文件1的'id'和'HA'列映射到'virusStrain'和'serumStrain'列
    replacement_dict = dict(zip(file1_df['HA'], file1_df['id']))
    print(len(replacement_dict))

    # 使用字典替换CSV文件2的'virusStrain'和'serumStrain'列
    file2_df['virusStrain'] = file2_df['virusStrain'].map(replacement_dict)

    file2_df['serumStrain'] = file2_df['serumStrain'].map(replacement_dict)

    # 保存修改后的CSV文件2
    file2_df.to_csv('../data/MYlog2/combine.csv', index=False)


def load_vec(HA1_path):
    data = np.genfromtxt(HA1_path)
    name = data[:, 0:1]  # 毒株名
    vec = data[:, 1:]  # 毒株嵌入，没有卷积过
    # print(vec.shape)
    num = vec.shape[0]
    # vec = vec.reshape(num, 100, 327)  # 253 个 100*327 矩阵
    vec = vec.reshape(num, 327, 100)  # 253 个 327*100 矩阵
    # print(vec.shape)
    # print(vec[0])
    # print(vec[0].shape)
    vec = torch.FloatTensor(vec)  # 转换为tensor
    #
    inputs = vec.unsqueeze(1)  # 添加一维(253,1,100,327)  (253,1,327,100)
    return inputs


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
