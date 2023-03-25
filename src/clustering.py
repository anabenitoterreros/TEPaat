# load and manipulate functions
import pandas as pd
import numpy as np
# plot functions
import matplotlib.pyplot as plt
import seaborn as sns
#
# cluster function
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import hdbscan
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score

#3 import dimensionality module
from src.dimensionality import DimensionalityReduction

class ClusterAnalysis(DimensionalityReduction):
    def __init__(self, data, clusters):
        self.data = data
        self.clusters = clusters
    
    def create_KMeans(self, state = 1, init = 10):
        # create algorithm
        kmeans = KMeans(n_clusters = self.clusters, random_state = state, n_init = init)
        # fit data
        kmeans_with_DR = kmeans.fit(self.data)
        # predict cluster
        self.kmean_data = kmeans_with_DR.predict(self.data)
        
        return self.kmean_data

    def create_DBSCAN(self, eps_ = 0.1, min_sample_ = 5):
        # create algorithm
        dbscan = DBSCAN(eps = eps_, min_samples = min_sample_)
        # fit data and predict cluster
        self.dbscan_data = dbscan.fit_predict(self.data)

        return self.dbscan_data
    
    def create_AgglomerativeClustering(self):
        # create algorithm
        aggloClust = AgglomerativeClustering(n_clusters = self.clusters)
        # fit data
        aggloClust_with_DR = aggloClust.fit(self.data)
        # predict cluster
        self.aggloClust_data = aggloClust_with_DR.labels_

        return self.aggloClust_data

    def create_HDBSCAN(self, min_cluster_size_ = 3, min_samples_ = 50):
        # instantiate algorithm
        hdbscan_ = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size_, min_samples = min_samples_)
        # fit data and predict cluster
        self.HDBSCAN_data = hdbscan_.fit_predict(self.data)

        return self.HDBSCAN_data

    def plot_2D_scatter(self, type = ['PCA', 'Kmeans']):
        """
        params: data (a 2D data) & labels (a dictionary)
        returns: profile of scatterplot of transformed data
        """
        if type[1] == 'Kmeans':
            labels = self.kmean_data

        if type[1] == 'DBSCAN':
            labels = self.dbscan_data

        if type[1] == 'AgglomerativeClustering':
            labels = self.aggloClust_data

        if type[1] == 'HDBSCAN':
            labels = self.kmean_data

        plt.figure(figsize=(6, 4))
        plt.scatter(x = self.data[:, 0], y = self.data[:, 1], c = labels, s = 10,  cmap = 'hsv')

        plt.xlabel(f"{type[0]} 1")
        plt.ylabel(f"{type[0]} 2")
        plt.title(type[1])

        plt.xlim([self.data[:, 0].min() - 0.5, self.data[:, 0].max() + 0.5])
        plt.ylim([self.data[:, 1].min() - 0.5, self.data[:, 1].max() + 0.5])
        plt.legend('',frameon=False)
        plt.savefig(f'./reports/{type[0]}_{type[1]}_2D_scatterplot.png') 

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)

   

    def compare_cluster(self, fig_size = (10, 4), type = 'UMAP'):
        # plot cluster
        fig = plt.figure(figsize = fig_size)
        sns.set_style("whitegrid", {'axes.grid' : False})

        x = self.data[:, 0]
        y = self.data[:, 1]

        # KMeans
        axs = fig.add_subplot(141)

        axs.scatter(x, y, c = self.kmean_data, s = 10, cmap = 'hsv'); axs.set_title('KMeans'); axs.set_xlim([x.min() - 0.5, x.max() + 0.5]); axs.set_ylim([y.min() - 0.5, y.max() + 0.5])

        # AgglomerativeClustering
        axs = fig.add_subplot(142)

        axs.scatter(x, y, c = self.aggloClust_data, s = 10, cmap = 'hsv'); axs.set_title('AgglomerativeClustering'); axs.set_xlim([x.min() - 0.5, x.max() + 0.5]); axs.set_ylim([y.min() - 0.5, y.max() + 0.5])

        # DBSCAN
        axs = fig.add_subplot(143)

        axs.scatter(x, y, c = self.dbscan_data, s = 10, cmap = 'hsv'); axs.set_title('DBSCAN'); axs.set_xlim([x.min() - 0.5, x.max() + 0.5]); axs.set_ylim([y.min() - 0.5, y.max() + 0.5])

        # Spectral Embedding
        axs = fig.add_subplot(144)

        axs.scatter(x, y, c = self.HDBSCAN_data, s = 10, cmap = 'hsv'); axs.set_title('HDSCAN'); axs.set_xlim([x.min() - 0.5, x.max() + 0.5]); axs.set_ylim([y.min() - 0.5, y.max() + 0.5])

        # display the figure
        plt.show()

        # save figure
        plt.savefig(f'./reports/{type}_allClustering_2D_scatterplot.png')

    