import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import DataLoader

def load_vec(matrices_path):
    data = np.genfromtxt(matrices_path)
    print(data.shape)
    # name = data[:, 0:1]
    vec = data[:, 1:]
    # print(vec.shape)
    num = vec.shape[0]

    matrices = vec.reshape(num, 327, 100)
    # print(vec.shape)
    # print(vec[0])
    # print(vec[0].shape)
    matrices = torch.FloatTensor(vec)
    #
    # inputs = inputs.unsqueeze(1)  #  (253,1,327,100)
    return matrices


matrices_path = "data/H3N2/HA_vec.csv"
# result = load_vec(matrices_path)
# print(result.shape)

distances_csv_path = "data/H3N2/AH3N2_combine.csv"


def load_dis(dis_path):
    csv_data = pd.read_csv(dis_path)
    return csv_data

def extract_features(matrices,csv_data):
    features_list = []
    labels_list = []

    for _, row in csv_data.iterrows():
        matrix1 = matrices[int(row['strainName1'])]
        matrix2 = matrices[int(row['strainName2'])]

        feature1 = matrix1
        feature2 = matrix2

        combined_feature = torch.cat((feature1, feature2), dim=0)
        features_list.append(combined_feature)
        labels_list.append(row['distance'])


        features_tensor = torch.stack(features_list)
        labels_tensor = torch.tensor(labels_list, dtype=torch.float32)


        dataset = TensorDataset(features_tensor, labels_tensor)


class MatrixDistanceDataset(Dataset):
    def __init__(self, matrices_path, distances_csv_path):


        features_data = np.genfromtxt(matrices_path)

        self.names = features_data[:, 0:1]  #
        self.features = torch.FloatTensor(features_data[:, 1:])


        distances_data = pd.read_csv(distances_csv_path)
        self.index_pairs = list(zip(distances_data.iloc[:, 0], distances_data.iloc[:, 1]))
        self.distances = torch.FloatTensor(distances_data.iloc[:, 2].values)

    def __len__(self):

        return len(self.index_pairs)

    def __getitem__(self, idx):

        index_pair = self.index_pairs[idx]


        feature1 = self.features[index_pair[0]]
        feature2 = self.features[index_pair[1]]


        distance = self.distances[idx]

        return (feature1, feature2), distance


dataset = MatrixDistanceDataset(matrices_path, distances_csv_path)

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    (features1, features2), distances = batch
    print(f"Batch features1 shape: {features1.shape}")
    print(f"Batch features2 shape: {features2.shape}")
    print(f"Batch distances shape: {distances.shape}")