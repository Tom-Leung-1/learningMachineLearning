from __future__ import print_function
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics
import math
import random
cdict = {0:"purple", 1: 'red', 2: 'blue', 3: 'green', 4:"yellow", 5:"orange",6:"brown"}

def create_dataset():
    # Generate sample points
    centers = [[3,5], [5,1], [8,2], [6,8], [9,7]]
    X, y = make_blobs(n_samples=1000,centers=centers,cluster_std=[0.5, 0.5, 1, 1, 1],random_state=3320)
    ####################################################################
    # you need to
    #   1. Plot the data points in a scatter plot.
    #   2. Use color to represents the clusters.
    #
    # YOUR CODE HERE!
    ####################################################################

    fig, ax = plt.subplots()
    for g in np.unique(y):
        ix = np.where(y == g)
        ax.scatter(X[ix, 0], X[ix, 1], c=cdict[g])
    plt.show()
    return [X, y]

def my_clustering(X, y, n_clusters):
    # =======================================
    # you need to
    #   1. Implement the k-means by yourself
    #   and cluster samples into n_clusters clusters using your own k-means
    #
    #   2. Print out all cluster centers and sizes.
    #
    #   3. Plot all clusters formed,
    #   and use different colors to represent clusters defined by k-means.
    #   Draw a marker (e.g., a circle or the cluster id) at each cluster center.
    #
    #   4. Return scores like this: return [score, score, score, score]
    #
    # YOUR CODE HERE!
    ####################################################################
    iteration = 0
    idx = random.sample(range(X.shape[0]), n_clusters)
    cluster = X[idx, :]
    min_choice = None
    while True:
        # iteration += 1
        # print("iteration:", iteration)
        A = None # array: row as cluster, column as dist
        for i in range(cluster.shape[0]):
            B = np.apply_along_axis(distance_p2, 1, X, b=cluster[i,:])
            if A is None:
                A = B
            else:
                A = np.vstack((A, B))
        min_choice = np.apply_along_axis(find_min_idx, 0, A)
        new_cluster = update_cluster(cluster, X, min_choice)
        diff = cluster - new_cluster
        dist = np.apply_along_axis(euclidean_distance, 1, diff)
        cluster = new_cluster
        if less_than_eps(dist, 0.01):
            break
    show_plot(X, cluster, min_choice)
    print("cluster:", cluster)
    cluster_size = get_cluster_size(min_choice, cluster)
    print("cluster_size:", cluster_size)
    return calculate_metrics(X, min_choice, y)  # You won't need this line when you are done

def get_cluster_size(min_choice, cluster):
    cluster_size = []
    for d in range(len(cluster)):
        ix = np.where(min_choice == d)
        cluster_size.append(len(list(ix)[0]))
    return cluster_size

def calculate_metrics(X, min_choice, y ):
    ari_score = metrics.adjusted_rand_score(y, min_choice)
    mri_score = metrics.mutual_info_score(y, min_choice)
    v_measure_score = metrics.v_measure_score(y, min_choice)
    silhouette_avg = metrics.silhouette_score(X, min_choice, metric='euclidean')
    return [ari_score, mri_score, v_measure_score, silhouette_avg]

def show_plot(X, cluster, min_choice):
    fig, ax = plt.subplots()
    for g in np.unique(min_choice):
        ix = np.where(min_choice == g)
        ax.scatter(X[ix, 0], X[ix, 1], c=cdict[g])
    for clus in cluster:
        ax.scatter(clus[0], clus[1], c="black", marker="v")
    plt.show()

def euclidean_distance(diff):
    return math.sqrt(diff[0] **2 + diff[1]**2)

def less_than_eps(dist, epsilon):
    for d in dist:
        if d > epsilon:
            return False
    return True

def update_cluster(cluster, X, min_choice):
    new_cluster = None
    for clus in range(cluster.shape[0]):
        idx_list = []
        for idx, choice in enumerate(min_choice):
            if choice == clus:
                idx_list.append(idx)
        B = np.sum(X[idx_list,:], axis=0)/ len(idx_list)
        if new_cluster is None:
            new_cluster = B
        else:
            new_cluster = np.vstack([new_cluster, B])
    return new_cluster

def distance_p2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def find_min_idx(a):
    min = a[0]
    for val in a:
        if min > val:
            min = val
    for idx, val in enumerate(a):
        if min == val:
            return idx
def main():
    X, y = create_dataset()
    range_n_clusters = [2, 3, 4, 5, 6, 7]
    ari_score = [None] * len(range_n_clusters)
    mri_score = [None] * len(range_n_clusters)
    v_measure_score = [None] * len(range_n_clusters)
    silhouette_avg = [None] * len(range_n_clusters)

    for n_clusters in range_n_clusters:
        i = n_clusters - range_n_clusters[0]
        print("Number of clusters is: ", n_clusters)
        # Implement the k-means by yourself in the function my_clustering
        [ari_score[i], mri_score[i], v_measure_score[i], silhouette_avg[i]] = my_clustering(X, y, n_clusters)
        print('The ARI score is: ', ari_score[i])
        print('The MRI score is: ', mri_score[i])
        print('The v-measure score is: ', v_measure_score[i])
        print('The average silhouette score is: ', silhouette_avg[i])
    cluster_analysis(ari_score, mri_score, v_measure_score, silhouette_avg, range_n_clusters)
    ####################################################################
    # Plot scores of all four evaluation metrics as functions of n_clusters in a single figure.
    #
    # YOUR CODE HERE!
    ####################################################################

def cluster_analysis(ari_score, mri_score, v_measure_score, silhouette_avg, range_n_clusters):
    plt.plot(range_n_clusters, ari_score, label="ari_score")
    plt.plot(range_n_clusters, mri_score, label="mri_score")
    plt.plot(range_n_clusters, v_measure_score, label="v_measure_score")
    plt.plot(range_n_clusters, silhouette_avg, label="silhouette_avg")
    plt.xlabel('number of clusters')
    plt.ylabel('metrics score')
    plt.title('scores performance on number of clusters')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

