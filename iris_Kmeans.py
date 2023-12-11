import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, kmeans_plusplus
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from KMeansClustering import KMeansClustering


df = pd.read_csv("iris.data", header=None)
data = df.iloc[:, 0:4]
# print(data)

pca = PCA(2)
data = pca.fit_transform(data)
# print(data)
# print(data.shape)


# Using K-Means implement from scratch
kmeans = KMeansClustering(k=3)
label = kmeans.fit(data)
# print(label)

# plt.scatter(data[:, 0], data[:, 1], c=label)
# plt.scatter(
#     kmeans.centroids[:, 0], kmeans.centroids[:, 1], marker="*", s=200, color="k"
# )
# plt.show()


# Using K-Means from library sklearn
kmeans = KMeans(n_clusters=3)
label = kmeans.fit_predict(data)
filtered_label0 = data[label == 0]
filtered_label1 = data[label == 1]
filtered_label2 = data[label == 2]
# filtered_label3 = data[label == 3]

# fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(111, projection="3d")

plt.scatter(filtered_label0[:, 0], filtered_label0[:, 1], color="red")
plt.scatter(filtered_label1[:, 0], filtered_label1[:, 1], color="green")
plt.scatter(filtered_label2[:, 0], filtered_label2[:, 1], color="blue")
# plt.scatter(filtered_label3[:, 0], filtered_label3[:, 1], color="yellow")

centroids = kmeans.cluster_centers_
u_labels = np.unique(label)


plt.scatter(centroids[:, 0], centroids[:, 1], s=80, marker="*", color="k")
plt.show()


# dbs = davies_bouldin_score(data, label)
# print(dbs)
