import pandas as pd
import numpy as np
import random
import torch
import warnings
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib import pyplot as plt


##################################################################################################
# convert distance values into label, distance
def calculate_label(antigenic_data):
    return antigenic_data['distance'].tolist()

group1 = ['A', 'I', 'L', 'M', 'P', 'V']
group2 = ['F', 'W', 'Y']
group3 = ['N', 'Q', 'S', 'T']
group4 = ['D', 'E', 'H', 'K', 'R']
group5 = ['C']
group6 = ['G']

# Yu-Chieh Liao method (Bioinformatics models for predicting antigenic variants of influenza A/H3N2 virus)

def Liao_feature_engineering(distance_input, seq_input, subtype):  # distance_input, seq_input都是dataframe数据
    distance_label = calculate_label(distance_input)
    label = {'label': distance_label}
    label = pd.DataFrame(label)
    # 生成一个与 distance_input 行数相同的序列，用于新DataFrame的索引
    index = pd.Series(np.arange(distance_input.shape[0]))
    # 蛋白质序列的长度
    length = len(seq_input['seq'].iloc[0])
    if subtype == 0:
        columns = list(range(1, 328, 1))
    elif subtype == 1:
        columns = list(range(1, 330, 1))
    elif subtype == 2:
        columns = list(range(1, 321, 1))
    for col in range(len(columns)):
        columns[col] = str(columns[col])
    Mut_feature = pd.DataFrame(index=index, columns=columns)

    for i in range(0, distance_input.shape[0]):
        strain_1 = []
        strain_2 = []
        for j in range(0, seq_input.shape[0]):
            if seq_input['description'].iloc[j].upper() == distance_input['Strain1'].iloc[i].upper():
                strain_1 = seq_input['seq'].iloc[j].upper()
            if seq_input['description'].iloc[j].upper() == distance_input['Strain2'].iloc[i].upper():
                strain_2 = seq_input['seq'].iloc[j].upper()
        for a in range(0, length):
            if strain_1[a] in group1 and strain_2[a] in group1:
                Mut_feature.iloc[i][a] = 0
            elif strain_1[a] in group2 and strain_2[a] in group2:
                Mut_feature.iloc[i][a] = 0
            elif strain_1[a] in group3 and strain_2[a] in group3:
                Mut_feature.iloc[i][a] = 0
            elif strain_1[a] in group4 and strain_2[a] in group4:
                Mut_feature.iloc[i][a] = 0
            elif strain_1[a] in group5 and strain_2[a] in group5:
                Mut_feature.iloc[i][a] = 0
            elif strain_1[a] in group6 and strain_2[a] in group6:
                Mut_feature.iloc[i][a] = 0
            else:
                Mut_feature.iloc[i][a] = 1
    Mut_feature = Mut_feature.join(label)
    return (Mut_feature)

# Liao_feature_H1N1 = Liao_feature_engineering(H1N1_Antigenic_dist, H1N1_seq, 0)
# Liao_feature_H1N1.to_csv('training/Yu-Chieh Liao/H1N1.csv')
# Liao_feature_H3N2 = Liao_feature_engineering(H3N2_Antigenic_dist, H3N2_seq, 1)
# Liao_feature_H3N2.to_csv('data/train/Liao_H3N2.csv')
# Liao_feature_H5N1 = Liao_feature_engineering(H5N1_Antigenic_dist, H5N1_seq, 2)
# Liao_feature_H5N1.to_csv('training/Yu-Chieh Liao/H5N1.csv')

#################################################################

print("Yao")
#################################################################


def replace_uncertain_amino_acids(amino_acids):
    """
    Randomly selects replacements for all uncertain amino acids.
    Expects and returns a string.
    """
    replacements = {'B': 'DN',
                    'J': 'IL',
                    'Z': 'EQ',
                    'X': 'ACDEFGHIKLMNPQRSTVWY'}

    for uncertain in replacements.keys():
        amino_acids = amino_acids.replace(uncertain, random.choice(replacements[uncertain]))

    return amino_acids


# Yao_embedding = pd.read_csv('data/sequence/NIEK910102_matrix.csv')
# Yao_pair_aa = []
# Yao_pair_aa_value = []
#
# for i in range(1, Yao_embedding.shape[0]): #start from index 1
#     aa_pair = 0
#     aa_value = 0
#     for j in range(1, Yao_embedding.shape[1]):
#         aa_pair = Yao_embedding.iloc[i][0] + Yao_embedding.iloc[0][j]
#         aa_value = Yao_embedding.iloc[i][j]
#         if not math.isnan(float(aa_value)):
#             Yao_pair_aa.append(aa_pair)
#             Yao_pair_aa_value.append(aa_value)
# Yao_embedding_table = pd.DataFrame({'aa_pair': Yao_pair_aa, 'aa_value': Yao_pair_aa_value})


def Yao_feature_engineering(distance_input, seq_input, subtype):
    distance_label = calculate_label(distance_input)
    label = {'label': distance_label}
    label = pd.DataFrame(label)

    index = pd.Series(np.arange(distance_input.shape[0]))
    length = len(seq_input['seq'].iloc[0])
    if subtype == 0:
        columns = list(range(1, 328, 1))
    elif subtype == 1:
        columns = list(range(1, 330, 1))
    elif subtype == 2:
        columns = list(range(1, 321, 1))
    for col in range(len(columns)):
        columns[col] = str(columns[col])
    Mut_feature = pd.DataFrame(index=index, columns=columns)

    for i in range(0, distance_input.shape[0]):
        strain_1 = []
        strain_2 = []
        for j in range(0, seq_input.shape[0]):
            if seq_input['description'].iloc[j].upper() == distance_input['Strain1'].iloc[i].upper():
                strain_1 = seq_input['seq'].iloc[j].upper()
            if seq_input['description'].iloc[j].upper() == distance_input['Strain2'].iloc[i].upper():
                strain_2 = seq_input['seq'].iloc[j].upper()
        for a in range(0, length):
            aa_pair_1 = strain_1[a] + strain_2[a]
            aa_pair_2 = strain_2[a] + strain_1[a]
            for b in range(0, Yao_embedding_table.shape[0]):
                if strain_1[a] == '-' or strain_2[a] == '-':
                    Mut_feature.iloc[i][a] = 0
                    break
                elif aa_pair_1 == Yao_embedding_table['aa_pair'].iloc[b] or aa_pair_2 == \
                        Yao_embedding_table['aa_pair'].iloc[b]:
                    Mut_feature.iloc[i][a] = Yao_embedding_table['aa_value'].iloc[b]
                    break

    Mut_feature = Mut_feature.join(label)
    return (Mut_feature)

#H1N1_seq = replace_uncertain_amino_acids(H1N1_seq)
#Yao_feature_H1N1 = Yao_feature_engineering(H1N1_Antigenic_dist, H1N1_seq, 0)
#Yao_feature_H1N1.to_csv('training/Yuhua Yao/H1N1.csv')
#
# H3N2_seq = replace_uncertain_amino_acids(H3N2_seq)
# Yao_feature_H3N2 = Yao_feature_engineering(H3N2_Antigenic_dist, H3N2_seq, 1)
# Yao_feature_H3N2.to_csv('data/train/Yuhua_Yao_H3N2.csv')

#H5N1_seq = replace_uncertain_amino_acids(H5N1_seq)
#Yao_feature_H5N1 = Yao_feature_engineering(H5N1_Antigenic_dist, H5N1_seq, 2)
#Yao_feature_H5N1.to_csv('training/Yuhua Yao/H5N1.csv')





########################################################################
# Min-Shi Lee's method (Predicting Antigenic Variants of Influenza A/H3N2 Viruses)
def distance_mutation(distance_input, seq_input):
    num_mut_list = []
    for i in range(0, distance_input.shape[0]):
        mutation_count = 0
        strain_1 = []
        strain_2 = []
        for j in range(0, seq_input.shape[0]):
            if seq_input['description'].iloc[j] == distance_input['Strain1'].iloc[i]:
                strain_1 = seq_input['seq'].iloc[j]
            if seq_input['description'].iloc[j] == distance_input['Strain2'].iloc[i]:
                strain_2 = seq_input['seq'].iloc[j]
        mutation_count = sum(0 if c1 == c2 else 1 for c1, c2 in zip(strain_1, strain_2))
        num_mut_list.append(mutation_count)
    return num_mut_list


# H1N1_num_mut_list = distance_mutation(H1N1_Antigenic_dist, H1N1_seq)
# H1N1_Antigenic_dist_list = list(H1N1_Antigenic_dist['Distance'])

# H3N2_num_mut_list = distance_mutation(H3N2_Antigenic_dist, H3N2_seq)
# H3N2_antigenic_dist_list = list(H3N2_Antigenic_dist['Distance'])

# H5N1_num_mut_list = distance_mutation(H5N1_Antigenic_dist, H5N1_seq)
# H5N1_antigenic_dist_list = list(H5N1_Antigenic_dist['Distance'])

# plt.scatter(H3N2_num_mut_list, y_pred, c='b')



#William Lees's work (A computational analysis of the antigenic properties of haemagglutinin in influenza A H3N2)
#divide residue sites based on proposed five epitope regions

print("Lees")
def William_Lees(distance_input, seq_input):
    # feature generation for epitope and regional method
    def generate_feature(region_type, distance_input, seq_input):
        distance_label = calculate_label(distance_input)
        label = {'label': distance_label}
        label = pd.DataFrame(label)

        index = pd.Series(np.arange(distance_input.shape[0]))
        if len(region_type) == 2:
            columns = ['epitope_a', 'epitope_b']
        elif len(region_type) == 5:
            columns = ['new_epitope_a', 'new_epitope_b', 'new_epitope_c', 'new_epitope_d', 'new_epitope_e']
        elif len(region_type) == 10:
            columns = ['regional_1', 'regional_2', 'regional_3', 'regional_4', 'regional_5',
                       'regional_6', 'regional_7', 'regional_8', 'regional_9', 'regional_10', ]
        Mut_feature = pd.DataFrame(index=index, columns=columns)

        for region in region_type:
            site = region_type[region]
            for i in range(0, distance_input.shape[0]):
                mut_count = 0
                for a in range(0, len(site)):
                    value_1 = 0
                    value_2 = 0
                    for j in range(0, seq_input.shape[0]):
                        if seq_input['description'].iloc[j].upper() == distance_input['Strain1'].iloc[i].upper():
                            value_1 = seq_input['seq'].iloc[j][
                                site[a] - 1]  # index is 0 by default, but the site starts from 1
                        if seq_input['description'].iloc[j].upper() == distance_input['Strain2'].iloc[i].upper():
                            value_2 = seq_input['seq'].iloc[j][site[a] - 1]
                    if value_1 == value_2:
                        mut_count = mut_count + 0
                    else:
                        mut_count = mut_count + 1

                Mut_feature[region].iloc[i] = mut_count
        Mut_feature = Mut_feature.join(label)
        return (Mut_feature)

    H3N2_new_epitope_a = [71, 72, 98, 122, 124, 126, 127, 130, 131, 132, 133, 135, 137, 138, 140, 141, 142, 143, 144, 145, 146, 148, 149,
                         150, 151, 152, 168, 255]
    H3N2_new_epitope_b = [128, 129, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 186, 187, 188, 189, 190, 192, 193, 194, 196,
                         197, 198, 199]
    H3N2_new_epitope_c = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 271, 272, 273, 274, 275, 276, 278, 279, 280, 282,
                         284, 285, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 307,
                         308, 309, 310, 311, 312, 313, 314]
    H3N2_new_epitope_d = [95, 96, 97, 99, 100, 101, 102, 103, 104, 105, 107, 117, 118, 120, 121, 166, 167, 169, 170, 171, 172, 173, 174,
                         175, 176, 177, 178, 179, 180, 182, 183, 184, 200, 201, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
                         214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235,
                         236, 238, 239, 240, 242, 243, 244, 245, 246, 247, 248, 257, 258]
    H3N2_new_epitope_e = [56, 57, 58, 59, 60, 62, 63, 64, 65, 67, 68, 69, 70, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
                         88, 89, 90, 91, 92, 93, 94, 109, 110, 111, 112, 113, 114, 115, 119, 259, 260, 261, 262, 263, 264, 265, 267, 268,
                         268, 270]

    H3N2_new_epitopes = {'new_epitope_a': H3N2_new_epitope_a, 'new_epitope_b': H3N2_new_epitope_b, 'new_epitope_c': H3N2_new_epitope_c,
                   'new_epitope_d': H3N2_new_epitope_d, 'new_epitope_e': H3N2_new_epitope_e}

    H3N2_epitope_data = generate_feature(H3N2_new_epitopes, distance_input, seq_input)
    H3N2_epitope_data.to_csv('data/train/Lees_H3N2_new_epitope_data.csv')


distance_path = "data/antigenic/H3N2_distances.csv"
distance_input = pd.read_csv(distance_path)

sequence_path = "data/sequence/H3N2_sequence.csv"
seq_input = pd.read_csv(sequence_path)

William_Lees(distance_input, seq_input)




