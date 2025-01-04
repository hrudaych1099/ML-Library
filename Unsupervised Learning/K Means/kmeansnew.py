#importing required data sstack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Initialize centroids
def initialize_centroids(X, k):
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]

#Assign clusters
def assign_clusters(X, centroids):
    """Assign each data point to the nearest centroid.""" #use euclidean distance basically
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

#Changing the centroids by assigning the given data points once again to assort into certain clusters by considering their initial centroid
def update_centroids(X, labels, k):
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])

#K-Means algorithm
def k_means(X, k, max_iters=100, tolerance=1e-4):
    # Initialize centroids
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k)
        if np.linalg.norm(new_centroids - centroids) < tolerance:
            break
        centroids = new_centroids
    return centroids, labels
#now i will import the unsupervised data
data = pd.read_csv("UnsupervisedLearning\\K Means\\unsupervised_data.csv")
X = data.iloc[:,1:7]
X = np.array(X)

#Perform K-Means clustering ( always prefer odd number of K to prevent ambiguities but even number gave me best results here
k = 2 #i decided to play with the value of K and found that k =2 gave best results for the data
centroids, labels = k_means(X, k)

#Plot the results
def plot_clusters(X, labels, centroids):
    for i in range(k):
        cluster_points = X[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i + 1}")
    plt.scatter(centroids[:, 0], centroids[:, 1], c='yellow', marker='x', s=200, label="Centroids")
    plt.legend()
    plt.title("K-Means Clustering")
    plt.show()

plot_clusters(X, labels, centroids)

