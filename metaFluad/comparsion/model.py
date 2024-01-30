import os
import pandas as pd
import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes // 16, planes, kernel_size=1)
        # 442176 2080000
        self.output_linear = torch.nn.Linear(2092800, 2)

    def forward(self, x):
        out = F.relu(x)
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(out))  # [32,64,325,100]

        # Squeeze
        w = F.avg_pool2d(out, [out.size(2), out.size(3)])  # [32,plane,1,1]

        w = F.relu(self.fc1(w))

        w = F.sigmoid(self.fc2(w))

        # Excitation
        out = out * w

        out += shortcut
        out = out.reshape(out.shape[0], -1)
        # print(out.shape)
        out = self.output_linear(out)
        return out



def Liao():
# distances_csv_path = "data/antigenic/H3N2_distances.csv"
# sequence_csv_path = "data/sequence/H3N2_sequence.csv"
# H3N2_Antigenic_dist = pd.read_csv(distances_csv_path)
# H3N2_seq = pd.read_csv(sequence_csv_path)
# Liao_feature_H3N2 = Liao_feature_engineering(H3N2_Antigenic_dist, H3N2_seq, 1)

    print("Liao")
    Liao_path = "data/train/Liao_H3N2.csv"
    Liao_feature_H3N2 = pd.read_csv(Liao_path)
    # Liao_feature_H3N2 = Liao_feature_H3N2.sample(frac=1, random_state=22).reset_index(drop=True)
    print(Liao_feature_H3N2.shape)

# 划分特征和标签
    X = Liao_feature_H3N2.iloc[:, :-1]  # 所有行，除最后一列之外的列
    y = Liao_feature_H3N2.iloc[:, -1]  # 所有行，只有最后一列
    # print(X.shape)
    # print(y.shape)
# 首先分离出测试集，占原始数据的 20%
    X_remaining, X_test, y_remaining, y_test = train_test_split(X, y, test_size=0.3, random_state=22)

    # 划分训练集和验证集的函数
    def split_train_val(X, y, train_size):
        # train_size 是相对于剩余数据的比例
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size, random_state=22)
        return X_train, X_val, y_train, y_val

    # 从30%到80%以10%为步长划分训练数据集并训练模型
    for train_percent in range(30, 90, 10):
        train_size = train_percent / 100.0
        X_train, X_val, y_train, y_val = split_train_val(X_remaining, y_remaining, train_size)

        # 创建模型实例
        regressor = LinearRegression()

        # 训练模型
        regressor.fit(X_train, y_train)

        # 在测试集上进行预测
        y_pred = regressor.predict(X_test)

        # 计算评估指标
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        # 输出结果
        print(f"Training size: {train_size * 100}% of remaining 80% data (or {train_size * 80}% of total data)")
        print(f"R2 on test set: {r2}")
        print(f"MAE on test set: {mae}")
        print(f"MSE on test set: {mse}\n")



    #
    # # 创建线性回归模型实例
    # model = LinearRegression()
    #
    # #############交叉验证#######################
    # # 定义评估指标列表
    # scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    #
    # # 应用五折交叉验证并获取详细信息
    # cv_results = cross_validate(model, X, y, cv=5, scoring=scoring_metrics, return_train_score=False)
    #
    # # 输出每一折的负MSE和负MAE以及R2分数
    # print(f"Negative MSE scores for each fold: {cv_results['test_neg_mean_squared_error']}")
    # print(f"Negative MAE scores for each fold: {cv_results['test_neg_mean_absolute_error']}")
    # print(f"R^2 scores for each fold: {cv_results['test_r2']}")
    #
    # # 输出交叉验证的平均负MSE、负MAE和R2分数
    # print(f"Average negative MSE score: {cv_results['test_neg_mean_squared_error'].mean()}")
    # print(f"Average negative MAE score: {cv_results['test_neg_mean_absolute_error'].mean()}")
    # print(f"Average R^2 score: {cv_results['test_r2'].mean()}")
    #
    # # 转换成正数形式的MSE和MAE
    # mse_scores = -cv_results['test_neg_mean_squared_error']
    # mae_scores = -cv_results['test_neg_mean_absolute_error']
    #
    # # # 输出正数形式的MSE和MAE
    # # print(f"MSE scores for each fold: {mse_scores}")
    # # print(f"MAE scores for each fold: {mae_scores}")
    #
    # print("-------------------------------------------------------")
    # print("Liao")
    # # 输出正数形式的平均MSE和MAE
    # print(f"Average MSE score: {mse_scores.mean():.6f}")
    # print(f"Average MAE score: {mae_scores.mean():.6f}")
    # print(f"Average R^2 score: {cv_results['test_r2'].mean():.6f}")
    # print("-------------------------------------------------------")
    # #############交叉验证#######################


# Liao()

import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def Yao_model():
    print("Yao")
    Yao_path = "data/train/Yuhua_Yao_H3N2.csv"
    Yao_feature_H3N2 = pd.read_csv(Yao_path)
    # Yao_feature_H3N2 = Yao_feature_H3N2.sample(frac=1, random_state=22).reset_index(drop=True)
    # print(Yao_feature_H3N2.shape)

    X = Yao_feature_H3N2.iloc[:, 1:-1]  # 所有行，除最后一列之外的列
    y = Yao_feature_H3N2.iloc[:, -1]  # 所有行，只有最后一列

    print(X.shape)
    print(y.shape)

    # 填充缺失值
    # imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X = imputer.fit_transform(X)

    # 首先分离出测试集，占原始数据的 20%
    X_remaining, X_test, y_remaining, y_test = train_test_split(X, y, test_size=0.3, random_state=22)

    # 划分训练集和验证集的函数
    def split_train_val(X, y, train_size):
        # train_size 是相对于剩余数据的比例
        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size, random_state=22)
        return X_train, X_val, y_train, y_val

    # 从30%到80%以10%为步长划分训练数据集并训练模型
    for train_percent in range(30, 90, 10):
        train_size = train_percent / 100.0
        X_train, X_val, y_train, y_val = split_train_val(X_remaining, y_remaining, train_size)

        # 创建模型实例
        regressor = RandomForestRegressor(n_estimators=10, random_state=22)

        # 训练模型
        regressor.fit(X_train, y_train)

        # 在测试集上进行预测
        y_pred = regressor.predict(X_test)

        # 计算评估指标
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        # 输出结果
        print(f"Training size: {train_size * 100}% of remaining 80% data (or {train_size * 80}% of total data)")
        print(f"R2 on test set: {r2}")
        print(f"MAE on test set: {mae}")
        print(f"MSE on test set: {mse}\n")

    # # 实例化随机森林回归器
    # regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    #
    # # 定义评分指标
    # scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    #
    # # 执行五折交叉验证
    # cv_results = cross_validate(regressor, X_imputed, y, cv=5, scoring=scoring)
    #
    # # 计算平均值和标准差
    # mean_r2 = cv_results['test_r2'].mean()
    # mean_mse = -cv_results['test_neg_mean_squared_error'].mean()  # 负MSE转换为正值
    # mean_mae = -cv_results['test_neg_mean_absolute_error'].mean()  # 负MAE转换为正值
    #
    #
    # # 输出正数形式的平均MSE和MAE
    # print(f"Average MSE score: {mean_mse:.6f}")
    # print(f"Average MAE score: {mean_mae:.6f}")
    # print(f"Average R^2 score: {mean_r2:.6f}")

# Yao_model()

print("Lees")
def Lees():
    Lees_path = "data/train/Lees_H3N2_new_epitope_data.csv"
    Lees_feature_H3N2 = pd.read_csv(Lees_path)
    # Yao_feature_H3N2 = Yao_feature_H3N2.sample(frac=1, random_state=22).reset_index(drop=True)
    print(Lees_feature_H3N2.shape)

    X = Lees_feature_H3N2.iloc[:, 1:-1]  # 所有行，除最后一列之外的列

    y = Lees_feature_H3N2.iloc[:, -1]  # 所有行，只有最后一列

    print(X.shape)
    print(y.shape)

    # 划分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建线性回归模型
    model = LinearRegression()

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集的目标变量
    y_pred = model.predict(X_test)

    # 计算 MSE, MAE, R2
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 输出指标
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")

Lees()

def Lees5():
    Lees_path = "data/train/Lees_H3N2_new_epitope_data.csv"
    Lees_feature_H3N2 = pd.read_csv(Lees_path)
    # Yao_feature_H3N2 = Yao_feature_H3N2.sample(frac=1, random_state=22).reset_index(drop=True)
    print(Lees_feature_H3N2.shape)

    X = Lees_feature_H3N2.iloc[:, 1:-1]  # 所有行，除最后一列之外的列

    y = Lees_feature_H3N2.iloc[:, -1]  # 所有行，只有最后一列

    print(X.shape)
    print(y.shape)
    # 初始化模型
    model = RandomForestRegressor(random_state=42)

    # 设置五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 初始化记录性能指标的列表
    mse_scores = []
    mae_scores = []
    r2_scores = []

    # 执行五折交叉验证
    for train_index, test_index in kf.split(X):
        # 根据索引划分训练集和测试集
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 训练模型
        model.fit(X_train, y_train)

        # 做出预测
        y_pred = model.predict(X_test)

        # 计算性能指标
        mse_scores.append(mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    # 计算平均性能指标
    print("Average MSE:", np.mean(mse_scores))
    print("Average MAE:", np.mean(mae_scores))
    print("Average R2:", np.mean(r2_scores))

# Lees5()