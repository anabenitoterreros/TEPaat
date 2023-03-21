# load and manipulate functions
import pandas as pd
import numpy as np
# plot functions
import matplotlib.pyplot as plt
import seaborn as sns
# dimensionality function
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding

class DimensionalityReduction():
    def __init__(self, data):
        self.data = data
        
    def fit_PCA(self, n_comp = 2):
        """
        params: number of components
        returns: pca model and transformed data
        """
        self.pca = PCA(n_components = n_comp, random_state = 104)
        self.pca.fit(self.data)
        self.pca_data = self.pca.transform(self.data)
        
        return self.pca_data

    def fit_tSNE(self, n_comp = 2, perplexity = 5):
        """
        params: number of components
        returns: pca model and transformed data
        """
        tSNE = TSNE(
                    n_components = n_comp, learning_rate = 'auto',
                    init = 'random', perplexity = perplexity
                    )

        self.tSNE_data = tSNE.fit_transform(self.data)

        return self.tSNE_data

    def fit_fastICA(self, n_comp = 2):
        """
        params: number of components
        returns: fastICA model and transformed data
        """
        fastICA = FastICA(n_components = n_comp, random_state = 0, whiten='unit-variance')
        
        self.fastICA_data = fastICA.fit_transform(self.data)

        return self.fastICA_data
    
    def fit_isomap(self, n_comp = 2, neighbors = 5):
        """
        params: number of components
        returns: fastICA model and transformed data
        """
        ISOMAP = Isomap(n_neighbors = neighbors, radius=None, n_components = n_comp)
        
        self.ISOMAP_data = ISOMAP.fit_transform(self.data)

        return self.ISOMAP_data

    def fit_SpectralEmbedding(self, n_comp = 2):
        """
        params: number of components
        returns: fastICA model and transformed data
        """
        embedding = SpectralEmbedding(n_components = n_comp)
        self.embedding_data = embedding.fit_transform(self.data)

        return self.embedding_data
        

    def plot_pcaVariance(self):
        """
        returns: profile of cummulative sum of variance
        """
        plt.figure(figsize=(6, 4))

        explained_variance = np.cumsum(self.pca.explained_variance_ratio_)
        plt.plot(explained_variance)
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance')
        plt.savefig('./reports/elbow_plot.png', dpi=100)

    def plot_2D_scatter(self, data, labels = {'x': 'PCA 1', 'y': 'PCA 2', 'type': 'PCA'}):
        """
        params: data (a 2D data) & labels (a dictionary)
        returns: profile of scatterplot of transformed data
        """
        title = labels['type']

        plt.figure(figsize=(6, 4))
        sns.scatterplot( x = data[:, 0], y = data[:, 1], s = 70)

        plt.xlabel(labels['x'])
        plt.ylabel(labels['y'])
        plt.title(title)
        plt.savefig(f'./reports/{title}_2D_scatterplot.png') 

    def plot_3D_scatter(self, data, labels = {'x': 'PCA 1', 'y': 'PCA 2', 'z': 'PCA 3', 'type': 'PCA'}):
        """
        params: data (a 3D data) & labels (a dictionary)
        returns: profile of scatterplot of transformed data
        """
        title = labels['type']

        fig = plt.figure(figsize=(6,4))
        sns.set_style("whitegrid", {'axes.grid' : False})

        axs = fig.add_subplot(111, projection='3d')

        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        axs.scatter(x, y, z, c = x, marker='o')

        axs.set_xlabel(labels['x'])
        axs.set_ylabel(labels['y'])
        axs.set_zlabel(labels['z'])

        plt.show()
        plt.savefig(f'./reports/{title}_3D_scatterplot.png')

    def plot_perplexity_effect(self, n_comp = 2, perpelixty_list = [5, 10, 15, 20, 25, 30]):
        """
        params: list of perplexity
        returns: profile of scatterplot at varying perplexity
        """
        result_tSNE_data = {}

        for pp in perpelixty_list:
            
            tSNE = TSNE(
                        n_components = n_comp, learning_rate = 'auto',
                        init = 'random', perplexity = pp)

            tSNE_data = tSNE.fit_transform(self.data)

            result_tSNE_data['pp' + str(pp)] = tSNE_data
        
        
        if n_comp == 2:
            #
            # create a figure and a subplot grid with 3 rows and 2 columns
            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(6, 4))
            #
            # create scatter plots on the subplots using Seaborn
            sns.scatterplot(x = result_tSNE_data['pp5'][:, 0], y = result_tSNE_data['pp5'][:, 1], s = 70, ax=axs[0, 0])
            sns.scatterplot(x = result_tSNE_data['pp10'][:, 0], y = result_tSNE_data['pp10'][:, 1], s = 70, ax=axs[0, 1])
            sns.scatterplot(x = result_tSNE_data['pp15'][:, 0], y = result_tSNE_data['pp15'][:, 1], s = 70, ax=axs[1, 0])
            sns.scatterplot(x = result_tSNE_data['pp20'][:, 0], y = result_tSNE_data['pp20'][:, 1], s = 70, ax=axs[1, 1])
            sns.scatterplot(x = result_tSNE_data['pp25'][:, 0], y = result_tSNE_data['pp25'][:, 1], s = 70, ax=axs[2, 0])
            sns.scatterplot(x = result_tSNE_data['pp30'][:, 0], y = result_tSNE_data['pp30'][:, 1], s = 70, ax=axs[2, 1])

            # set titles for each subplot
            axs[0, 0].set_title(f'perplexity = {perpelixty_list[0]}'); axs[0, 0].set_ylabel('tSNE 2')
            axs[0, 1].set_title(f'perplexity = {perpelixty_list[1]}')
            axs[1, 0].set_title(f'perplexity = {perpelixty_list[2]}'); axs[1, 0].set_ylabel('tSNE 2')
            axs[1, 1].set_title(f'perplexity = {perpelixty_list[3]}')
            axs[2, 0].set_title(f'perplexity = {perpelixty_list[4]}'); axs[2, 0].set_ylabel('tSNE 2'); axs[2, 0].set_xlabel('tSNE 1')
            axs[2, 1].set_title(f'perplexity = {perpelixty_list[5]}'); axs[2, 1].set_xlabel('tSNE 1')

            # adjust the spacing between subplots
            fig.tight_layout()
            # display the figure
            plt.show()
            # save figure
            plt.savefig('./reports/tSNE_2D_scatterplot_manyPerplexity.png')

        if n_comp == 3:
            #
            fig = plt.figure(figsize=(6, 4), layout='constrained')
            sns.set_style("whitegrid", {'axes.grid' : False})
            # create scatter plots on the subplots using Seaborn
            axs321 = fig.add_subplot(321, projection='3d')
            x = result_tSNE_data['pp5'][:, 0]; y = result_tSNE_data['pp5'][:, 1]; z = result_tSNE_data['pp5'][:, 2]
            axs321.scatter(x, y, z, c = x, marker='o')
            axs322 = fig.add_subplot(322, projection='3d')
            x = result_tSNE_data['pp10'][:, 0]; y = result_tSNE_data['pp10'][:, 1]; z = result_tSNE_data['pp10'][:, 2]
            axs322.scatter(x, y, z, c = x, marker='o')
            axs323 = fig.add_subplot(323, projection='3d')
            x = result_tSNE_data['pp15'][:, 0]; y = result_tSNE_data['pp15'][:, 1]; z = result_tSNE_data['pp15'][:, 2]
            axs323.scatter(x, y, z, c = x, marker='o')
            axs324 = fig.add_subplot(324, projection='3d')
            x = result_tSNE_data['pp20'][:, 0]; y = result_tSNE_data['pp20'][:, 1]; z = result_tSNE_data['pp20'][:, 2]
            axs324.scatter(x, y, z, c = x, marker='o')
            axs325 = fig.add_subplot(325, projection='3d')
            x = result_tSNE_data['pp25'][:, 0]; y = result_tSNE_data['pp25'][:, 1]; z = result_tSNE_data['pp25'][:, 2]
            axs325.scatter(x, y, z, c = x, marker='o')
            axs326 = fig.add_subplot(326, projection='3d')
            x = result_tSNE_data['pp30'][:, 0]; y = result_tSNE_data['pp30'][:, 1]; z = result_tSNE_data['pp30'][:, 2]
            axs326.scatter(x, y, z, c = x, marker='o')
            #
            # set titles for each subplot
            axs321.set_title(f'perplexity = {perpelixty_list[0]}'); axs321.set_xlabel('tSNE 1'); axs321.set_ylabel('tSNE 2'); axs321.set_zlabel('tSNE 3')
            axs322.set_title(f'perplexity = {perpelixty_list[1]}'); axs322.set_xlabel('tSNE 1'); axs322.set_ylabel('tSNE 2'); axs322.set_zlabel('tSNE 3')
            axs323.set_title(f'perplexity = {perpelixty_list[2]}'); axs323.set_xlabel('tSNE 1'); axs323.set_ylabel('tSNE 2'); axs323.set_zlabel('tSNE 3')
            axs324.set_title(f'perplexity = {perpelixty_list[3]}'); axs324.set_xlabel('tSNE 1'); axs324.set_ylabel('tSNE 2'); axs324.set_zlabel('tSNE 3')
            axs325.set_title(f'perplexity = {perpelixty_list[4]}'); axs325.set_xlabel('tSNE 1'); axs325.set_ylabel('tSNE 2'); axs325.set_zlabel('tSNE 3')
            axs326.set_title(f'perplexity = {perpelixty_list[5]}'); axs326.set_xlabel('tSNE 1'); axs326.set_ylabel('tSNE 2'); axs326.set_zlabel('tSNE 3')
            # display the figure
            plt.show()
            # save figure
            plt.savefig('./reports/tSNE_3D_scatterplot_manyPerplexity.png')

    
    def plot_neighbors_effect(self, n_comp = 2, neighborsint = [5, 10, 15, 20, 25, 30]):
        """
        params: list of perplexity
        returns: profile of scatterplot at varying neighbor number
        """

        result_Ismap_data = {}

        for nb in neighborsint:

            Ismap = Isomap(n_neighbors=nb, radius=None, n_components = n_comp)

            ismap_data = Ismap.fit_transform(self.data)
            
            result_Ismap_data['nb' + str(nb)] = ismap_data


        if n_comp == 2:
            # create a figure and a subplot grid with 3 rows and 2 columns
            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(6, 4))

            # create scatter plots on the subplots using Seaborn
            sns.scatterplot(x = result_Ismap_data['nb5'][:, 0], y = result_Ismap_data['nb5'][:, 1], s = 70, ax=axs[0, 0])
            sns.scatterplot(x = result_Ismap_data['nb10'][:, 0], y = result_Ismap_data['nb10'][:, 1], s = 70, ax=axs[0, 1])
            sns.scatterplot(x = result_Ismap_data['nb15'][:, 0], y = result_Ismap_data['nb15'][:, 1], s = 70, ax=axs[1, 0])
            sns.scatterplot(x = result_Ismap_data['nb20'][:, 0], y = result_Ismap_data['nb20'][:, 1], s = 70, ax=axs[1, 1])
            sns.scatterplot(x = result_Ismap_data['nb25'][:, 0], y = result_Ismap_data['nb25'][:, 1], s = 70, ax=axs[2, 0])
            sns.scatterplot(x = result_Ismap_data['nb30'][:, 0], y = result_Ismap_data['nb30'][:, 1], s = 70, ax=axs[2, 1])

            # set titles for each subplot
            axs[0, 0].set_title(f'n_neighbors = {neighborsint[0]}'); axs[0, 0].set_ylabel('Isomap 2')
            axs[0, 1].set_title(f'n_neighbors = {neighborsint[1]}')
            axs[1, 0].set_title(f'n_neighbors = {neighborsint[2]}'); axs[1, 0].set_ylabel('Isomap 2')
            axs[1, 1].set_title(f'n_neighbors = {neighborsint[3]}')
            axs[2, 0].set_title(f'n_neighbors = {neighborsint[4]}'); axs[2, 0].set_ylabel('Isomap 2'); axs[2, 0].set_xlabel('Isomap 1')
            axs[2, 1].set_title('n_neighbors = {neighborsint[5]}'); axs[2, 1].set_xlabel('Isomap 1')

            # adjust the spacing between subplots
            fig.tight_layout()

            # display the figure
            plt.show()

            # save figure
            plt.savefig('./reports/isomap_2D_scatterplot_manyNeighbors.png')

        if n_comp == 3:
            #
            fig = plt.figure(figsize=(6, 4), layout='constrained')
            sns.set_style("whitegrid", {'axes.grid' : False})
            # create scatter plots on the subplots using Seaborn
            axs321 = fig.add_subplot(321, projection='3d')
            x = result_Ismap_data['nb5'][:, 0]; y = result_Ismap_data['nb5'][:, 1]; z = result_Ismap_data['nb5'][:, 2]
            axs321.scatter(x, y, z, c = x, marker='o')
            axs322 = fig.add_subplot(322, projection='3d')
            x = result_Ismap_data['nb10'][:, 0]; y = result_Ismap_data['nb10'][:, 1]; z = result_Ismap_data['nb10'][:, 2]
            axs322.scatter(x, y, z, c = x, marker='o')
            axs323 = fig.add_subplot(323, projection='3d')
            x = result_Ismap_data['nb15'][:, 0]; y = result_Ismap_data['nb15'][:, 1]; z = result_Ismap_data['nb15'][:, 2]
            axs323.scatter(x, y, z, c = x, marker='o')
            axs324 = fig.add_subplot(324, projection='3d')
            x = result_Ismap_data['nb20'][:, 0]; y = result_Ismap_data['nb20'][:, 1]; z = result_Ismap_data['nb20'][:, 2]
            axs324.scatter(x, y, z, c = x, marker='o')
            axs325 = fig.add_subplot(325, projection='3d')
            x = result_Ismap_data['nb25'][:, 0]; y = result_Ismap_data['nb25'][:, 1]; z = result_Ismap_data['nb25'][:, 2]
            axs325.scatter(x, y, z, c = x, marker='o')
            axs326 = fig.add_subplot(326, projection='3d')
            x = result_Ismap_data['nb30'][:, 0]; y = result_Ismap_data['nb30'][:, 1]; z = result_Ismap_data['nb30'][:, 2]
            axs326.scatter(x, y, z, c = x, marker='o')
            #
            # set titles for each subplot
            axs321.set_title(f'n_neighbors = {neighborsint[0]}'); axs321.set_xlabel('iSM 1'); axs321.set_ylabel('iSM 2'); axs321.set_zlabel('iSM 3')
            axs322.set_title(f'n_neighbors = {neighborsint[1]}'); axs322.set_xlabel('iSM 1'); axs322.set_ylabel('iSM 2'); axs322.set_zlabel('iSM 3')
            axs323.set_title(f'n_neighbors = {neighborsint[2]}'); axs323.set_xlabel('iSM 1'); axs323.set_ylabel('iSM 2'); axs323.set_zlabel('iSM 3')
            axs324.set_title(f'n_neighbors = {neighborsint[3]}'); axs324.set_xlabel('iSM 1'); axs324.set_ylabel('iSM 2'); axs324.set_zlabel('iSM 3')
            axs325.set_title(f'n_neighbors = {neighborsint[4]}'); axs325.set_xlabel('iSM 1'); axs325.set_ylabel('iSM 2'); axs325.set_zlabel('iSM 3')
            axs326.set_title(f'n_neighbors = {neighborsint[5]}'); axs326.set_xlabel('iSM 1'); axs326.set_ylabel('iSM 2'); axs326.set_zlabel('iSM 3')
            # display the figure
            plt.show()
            # save figure
            plt.savefig('./reports/tSNE_3D_scatterplot_manyNeighbors.png')