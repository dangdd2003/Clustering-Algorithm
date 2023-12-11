import numpy as np


# Implement basic K-means from scrach
class KMeansClustering:
    def __init__(self, k=3):
        self.k = k
        self.controids = None

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point) ** 2, axis=1))

    def fit(self, X, max_iterations=200):
        # random the controids near the data points
        self.centroids = np.random.uniform(
            np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1])
        )

        for _ in range(max_iterations):
            y = []

            for data_point in X:
                distances = KMeansClustering.euclidean_distance(
                    data_point, self.centroids
                )
                # return the index of the cluster which has the smallest distances with nearest centroids
                cluster_no = np.argmin(distances)
                y.append(cluster_no)

            y = np.array(y)

            cluster_indices = []

            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))

            cluster_centers = []

            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])

            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centers)

        return y
