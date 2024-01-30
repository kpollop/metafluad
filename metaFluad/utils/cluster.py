import csv

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE, MDS

data = pd.read_csv('H3N2_embedding.csv', header=None)  # H3N2


# tsne = TSNE(n_components=2, perplexity=5, method='exact', init='pca', n_iter=1000, early_exaggeration=30)
tsne = TSNE(n_components=2, perplexity=8, method='exact', init='pca', n_iter=1000, early_exaggeration=20)
reduced_data = tsne.fit_transform(data)
print(reduced_data)
m = 1
n = 21
k = 9

save_path = '{}_1-{}_{}cluster.csv'.format(m, n, k)
# 使用K均值聚类
kmeans = KMeans(n_clusters=k)  # 设置聚类数量

clusters = kmeans.fit_predict(reduced_data)

# 保存聚类结果
data = pd.read_csv('H3N2_HA.csv')
labels = kmeans.labels_ + 1
data['labels'] = labels

tsne_df = pd.DataFrame(reduced_data, columns=['x', 'y'])
# 将 t-SNE 结果的两列添加到原始 DataFrame
data['x'] = tsne_df['x']
data['y'] = tsne_df['y']
# 保存具有聚类标签的新CSV文件
data.to_csv(save_path, index=False)

# 获取tab20调色板的所有颜色
tab20_colors = plt.cm.tab20.colors
# 选择前17种颜色
colors = tab20_colors[:17]


colors = [
    '#1f77b4',  # 蓝色
    '#ff7f0e',  # 橙色
    '#2ca02c',  # 绿色
    '#d62728',  # 红色
    '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    '#e377c2',  # 粉红色
    '#7f7f7f',  # 灰色
    '#bcbd22',  # 黄绿色
    '#17becf',  # 青绿色
    '#ff9896',  # 淡红色
    '#98df8a',  # 淡绿色
    '#c5b0d5',  # 淡紫色
    '#c49c94',  # 淡棕色
    '#f7b6d2',  # 淡粉色
    '#aa7111', 'lime', 'teal', 'navy', 'olive', 'maroon'
]

# 可视化聚类结果，每个类别使用不同颜色
plt.figure()

special_indices = [10, 1]  # 指定要特殊显示的点的索引
for i in range(k):  # 假设有11个聚类
    plt.scatter(
        reduced_data[clusters == i, 0],
        reduced_data[clusters == i, 1],
        # reduced_data[clusters == i, 2],
        c=colors[i],
        label=f'Cluster {int(i + 1)}',
        # label=f'Cluster {i + 1}'
        s=20
    )


plt.title('Clustering Visualization')

plt.legend(fontsize="5")
plt.show()

x = data['x']
y = data['y']
labels = data['labels']

# 获取tab20调色板的所有颜色
tab20_colors = plt.cm.tab20.colors
# 选择前17种颜色
colors = tab20_colors[:17]

unique_labels = labels.unique()

# unique_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]

plt.figure()
# 隐藏坐标轴
plt.axis('off')
# 隐藏整个框架
plt.gca().set_frame_on(False)

# plt.figure(figsize=(5, 2.8), dpi=120)
plt.tight_layout()

for i, label in enumerate(unique_labels):
    cluster_data = data[data['labels'] == label]
    plt.scatter(cluster_data['x'], cluster_data['y'],
                label=classifer[i],
                # label=f'Cluster {label}',
                # c=[colors(i)],
                c=colors[i],
                s=16)

special_ids = [146, 5, 104, 121,
               93, 6, 10, 103,
               108, 48, 25, 56, 123,
               64, 138, 30, 101, 53]  # 指定要特殊显示的点的索引  # 例如，特定的ID列表


special_markers = ['o', 's', 'D']  # 对应特定ID的标记样式

star = {146: 'A/BRISBANE/10/2007', 5: 'A/KANSAS/14/2017',
        104: 'A/FUJIAN/411/2002', 121: 'A/SWITZERLAND/9715293/2013',
        93: 'A/PERTH/16/2009', 6: 'A/VICTORIA/361/2011',
        10: 'A/DARWIN/9/2021', 103: 'A/WELLINGTON/1/2004',
        108: 'A/CALIFORNIA/7/2004', 48: 'A/HONGKONG/4801/2014',
        25: 'A/HONGKONG/45/2019', 56: 'A/CAMBODIA/E0826360/2020',
        64: 'A/SOUTHAUSTRALIA/34/2019', 138: 'A/HONGKONG/2671/2019',
        30: 'A/SINGAPORE/INFIMH-16-0019/2016', 101: 'A/WISCONSIN/67/2005',
        53: 'A/SWITZERLAND/8060/2017', 123: 'A/TEXAS/50/2012'
        }

for i, special_id in enumerate(special_ids):
    special_data = data[data['id'] == special_id]
    marker = special_markers[2]
    plt.scatter(special_data['x'], special_data['y'], color='#333aaa',
                marker='o', facecolors='none', edgecolors='red',
                s=20)
    # 添加标注
    for x, y in zip(special_data['x'], special_data['y']):
        label = star.get(special_id, '')  # 从star字典中获取对应ID的标注文本
        plt.annotate(label,  # 这里的label是要显示的文本
                     (x, y),  # 这是点的位置
                     textcoords="offset points",  # 如何解释xytext
                     xytext=(0, 10),  # 文本的位置偏移量
                     ha='center',  # 水平居中对齐文本
                     fontsize="4")
    # 绘制空心圆代表数据点
    # scatter = ax.scatter(x, y, marker='o', facecolors='none', edgecolors='red', s=100)  # s参数控制点的大小

plt.legend(fontsize="3")
plt.savefig('cluster/pic/cluster_{}_{}_{}.svg'.format(m, n, k))  # 保存为 SVG 文件
plt.show()
