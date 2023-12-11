import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("iris.data")

data = df.iloc[:, 0:4]
# print(data)

linked_single = linkage(data, "single", metric="euclidean")

linked_complete = linkage(data, "complete", metric="euclidean")

# Single Linkage Dendrogram
plt.figure(figsize=(10, 15))
dendrogram(linked_single, orientation="right")
plt.title("Single Linkage Dendrogram")
plt.ylabel("Cluster")
plt.xlabel("Distance")

# Complete Linkage Dendrogram
plt.figure(figsize=(10, 15))
dendrogram(linked_complete, orientation="right")
plt.title("Complete Linkage Dendrogram")
plt.xlabel("Distance")
plt.ylabel("Cluster")
plt.show()
