import csv

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE, MDS

data = pd.read_csv('cluster/H3N2_embedding.csv', header=None)  # H3N2

tsne = TSNE(n_components=2, perplexity=8, method='exact', init='pca', n_iter=1000, early_exaggeration=20)
reduced_data = tsne.fit_transform(data)
print(reduced_data)
m = 2
n = 2
k = 10

save_path = 'cluster/csv/{}-{}_k={}-0418cluster.csv'.format(m, n, k)

kmeans = KMeans(n_clusters=k)

clusters = kmeans.fit_predict(reduced_data)

print(clusters.shape)


data = pd.read_csv('data/H3N2/AH3N2_HA.csv')

labels = kmeans.labels_ + 1
data['labels'] = labels

tsne_df = pd.DataFrame(reduced_data, columns=['x', 'y'])

data['x'] = tsne_df['x']
data['y'] = tsne_df['y']

data.to_csv(save_path, index=False)
# data.to_csv('cluster/{}_1-20_{}cluster_del4.csv'.format(m, k), index=False)
print(labels)


# 定义颜色映射
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
          '#800080', '#189e0f', '#333aaa', '#555777', '#aa7111', 'lime', 'teal', 'navy', 'olive', 'maroon']


tab20_colors = plt.cm.tab20.colors
colors = tab20_colors[:17]


plt.figure()
special_indices = [10, 1]
for i in range(k):
    plt.scatter(
        reduced_data[clusters == i, 0],
        reduced_data[clusters == i, 1],
        # reduced_data[clusters == i, 2],
        c=colors[i],
        label=f'Cluster {int(i + 1)}',
        # label=f'Cluster {i + 1}'
        s=20
    )

# plt.title('Clustering Visualization')


plt.title('Clustering Visualization')

plt.legend(fontsize="5")
plt.show()

x = data['x']
y = data['y']
labels = data['labels']

classifer = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

tab20_colors = plt.cm.tab20.colors

colors = tab20_colors[:17]

unique_labels = labels.unique()

unique_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
print(unique_labels)
plt.figure(figsize=(5, 5), dpi=300)

plt.axis('off')

plt.gca().set_frame_on(False)

plt.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.01)

for i, label in enumerate(unique_labels):
    cluster_data = data[data['labels'] == label]
    plt.scatter(cluster_data['x'], cluster_data['y'],
                label=classifer[i],

                c=colors[i],
                s=16)

special_ids = [146, 5, 104, 121,
               93, 6, 10, 103,
               108, 48, 25, 56, 123,
               64, 138, 30, 101, 53]
special_markers = ['o', 's', 'D']

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



offsets = {
    25: (0, -10),
    103: (0, -7),
    146: (0, -7),
    138: (0, -3),

}

for i, special_id in enumerate(special_ids):
    special_data = data[data['id'] == special_id]
    marker = special_markers[2]
    plt.scatter(special_data['x'], special_data['y'], color='#333aaa',
                marker='*', facecolors='none', edgecolors='red',
                s=50)

    for x, y in zip(special_data['x'], special_data['y']):
        label = star.get(special_id, '')  #

        xytext_offset = offsets.get(special_id, (0, 7))  #
        plt.annotate(label,
                     (x, y),
                     textcoords="offset points",  #
                     xytext=xytext_offset,
                     ha='center',
                     fontsize="5")

plt.savefig('cluster/pic/cluster_{}_{}_{}.svg'.format(m, n, k))
plt.savefig('cluster/pic/cluster_{}_{}_{}.eps'.format(m, n, k))
plt.savefig('cluster/pic/cluster_{}_{}_{}.png'.format(m, n, k))
plt.show()
