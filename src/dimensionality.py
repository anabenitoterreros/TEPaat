# load and manipulate functions
import pandas as pd
import numpy as np
# plot functions
import matplotlib.pyplot as plt
import seaborn as sns
# dimensionality function
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE, Isomap, SpectralEmbedding
import umap

class DimensionalityReduction():
    def __init__(self, data, n_comp):
        self.data = data
        self.n_comp = n_comp
        
    def fit_PCA(self):
        """
        params: number of components
        returns: pca model and transformed data
        """
        self.pca = PCA(n_components = self.n_comp, random_state = 104)
        self.pca.fit(self.data)
        self.pca_data = self.pca.transform(self.data)
        
        return self.pca_data

    def fit_tSNE(self, perplexity = 5):
        """
        params: number of components
        returns: pca model and transformed data
        """
        tSNE = TSNE(
                    n_components = self.n_comp, learning_rate = 'auto',
                    init = 'random', perplexity = perplexity
                    )

        self.tSNE_data = tSNE.fit_transform(self.data)

        return self.tSNE_data

    def fit_fastICA(self, iterations = 30000):
        """
        params: number of components
        returns: fastICA model and transformed data
        """
        fastICA = FastICA(n_components = self.n_comp, random_state = 0, whiten='unit-variance', max_iter = iterations)
        
        self.fastICA_data = fastICA.fit_transform(self.data)

        return self.fastICA_data
    
    def fit_isomap(self, neighbors = 5):
        """
        params: number of components
        returns: fastICA model and transformed data
        """
        ISOMAP = Isomap(n_neighbors = neighbors, radius=None, n_components = self.n_comp)
        
        self.ISOMAP_data = ISOMAP.fit_transform(self.data)

        return self.ISOMAP_data

    def fit_SpectralEmbedding(self):
        """
        params: number of components
        returns: fastICA model and transformed data
        """
        embedding = SpectralEmbedding(n_components = self.n_comp)
        self.embedding_data = embedding.fit_transform(self.data)

        return self.embedding_data
        
    def fit_UMAP(self, state = 16):
        """
        params: number of components
        returns: UMAP model and transformed data
        """
        umap_model = umap.UMAP(n_components = self.n_comp, random_state = state)
        self.umap_data = umap_model.fit_transform(self.data)

        return self.umap_data

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

    def plot_2D_scatter(self, data, type = 'PCA'):
        """
        params: data (a 2D data) & labels (a dictionary)
        returns: profile of scatterplot of transformed data
        """

        plt.figure(figsize=(6, 4))
        plt.scatter(x = data[:, 0], y = data[:, 1])

        plt.xlabel(f"{type} 1")
        plt.ylabel(f"{type} 2")
        plt.xlim([data[:, 0].min() - 0.5, data[:, 0].max() + 0.5])
        plt.ylim([data[:, 1].min() - 0.5, data[:, 1].max() + 0.5])
        plt.title(type)
        plt.savefig(f'./reports/{type}_2D_scatterplot.png') 

    def plot_3D_scatter(self, data, type = 'PCA'):
        """
        params: data (a 3D data) & labels (a dictionary)
        returns: profile of scatterplot of transformed data
        """

        fig = plt.figure(figsize=(6,4))
        sns.set_style("whitegrid", {'axes.grid' : False})

        axs = fig.add_subplot(111, projection='3d')

        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        axs.scatter(x, y, z, c = x, marker='o')

        axs.set_xlabel(f"{type} 1")
        axs.set_ylabel(f"{type} 3")
        axs.set_zlabel(f"{type} 3")
        axs.set_xlim(xmin = x.min(), xmax=x.max())
        axs.set_ylim(ymin = y.min(), ymax=y.max())

        plt.show()
        plt.savefig(f'./reports/{type}_3D_scatterplot.png')



    def compare_methods(self, fig_size = (6, 3)):
        """
        params: data (a 2D data) & labels (a dictionary)
        returns: profile of scatterplot of transformed data
        """
        fig = plt.figure(figsize = fig_size, layout='constrained')
        sns.set_style("whitegrid", {'axes.grid' : False})

        # A
        axs = fig.add_subplot(231)
        x = self.pca_data[:, 0]
        y = self.pca_data[:, 1]

        axs.scatter(x, y); axs.set_title('PCA')

        # B
        axs = fig.add_subplot(232)
        x = self.tSNE_data[:, 0]
        y = self.tSNE_data[:, 1]

        axs.scatter(x, y); axs.set_title('tSNE')

        # C
        axs = fig.add_subplot(233)
        x = self.fastICA_data[:, 0]
        y = self.fastICA_data[:, 1]

        axs.scatter(x, y); axs.set_title('ICA')

        # ISOMAP
        axs = fig.add_subplot(234)
        x = self.ISOMAP_data[:, 0]
        y = self.ISOMAP_data[:, 1]

        axs.scatter(x, y); axs.set_title('ISOMAP')

        # Spectral Embedding
        axs = fig.add_subplot(235)
        x = self.embedding_data[:, 0]
        y = self.embedding_data[:, 1]

        axs.scatter(x, y); axs.set_title('Spectral Embedding')

        # UMAP
        axs = fig.add_subplot(236)
        x = self.umap_data[:, 0]
        y = self.umap_data[:, 1]

        axs.scatter(x, y); axs.set_title('UMAP')

        # display the figure
        plt.show()

        # save figure
        plt.savefig(f'./reports/all_DRs_2D_scatterplot.png')


    def plot_perplexity_effect(self, perpelixty_list = [5, 10, 15, 20, 25, 30]):
        """
        params: list of perplexity
        returns: profile of scatterplot at varying perplexity
        """
        result_tSNE_data = {}

        for pp in perpelixty_list:
            
            tSNE = TSNE(
                        n_components = self.n_comp, learning_rate = 'auto',
                        init = 'random', perplexity = pp)

            tSNE_data = tSNE.fit_transform(self.data)

            result_tSNE_data['pp' + str(pp)] = tSNE_data
        
        
        if self.n_comp == 2:
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

        if self.n_comp == 3:
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

    
    def plot_neighbors_effect(self, neighborsint = [5, 10, 15, 20, 25, 30]):
        """
        params: list of perplexity
        returns: profile of scatterplot at varying neighbor number
        """

        result_Ismap_data = {}

        for nb in neighborsint:

            Ismap = Isomap(n_neighbors=nb, radius=None, n_components = self.n_comp)

            ismap_data = Ismap.fit_transform(self.data)
            
            result_Ismap_data['nb' + str(nb)] = ismap_data


        if self.n_comp == 2:
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

        if self.n_comp == 3:
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


    # Effect of min_samples
    def effect_of_umapneigbors(self, state = 16, neighbors = [2, 5, 10, 20, 30, 50], type = 'UMAP'):

        result_umap_2D_data = {}

        for idx, param in enumerate(neighbors):

            umap_2D = umap.UMAP(n_neighbors = param, n_components = self.n_comp, random_state = state)

            umap_2D_data = umap_2D.fit_transform(self.data)

            result_umap_2D_data['M' + str(idx)] = umap_2D_data

        # create a figure and a subplot grid with 3 rows and 2 columns

        fig = plt.figure(figsize=(6, 4), layout='constrained')
        sns.set_style("whitegrid", {'axes.grid' : False})
        
        # create scatter plots on the subplots using Seaborn
        x = self.data[:, 0]; y = self.data[:, 1]
        
        axs321 = fig.add_subplot(231)
        axs321.scatter(x, y, c = result_umap_2D_data['M0'],  s = 10, cmap = 'hsv')
        
        axs322 = fig.add_subplot(232)
        axs322.scatter(x, y, c = result_umap_2D_data['M1'],  s = 10, cmap = 'hsv')
        
        axs323 = fig.add_subplot(233)
        axs323.scatter(x, y, c = result_umap_2D_data['M2'],  s = 10, cmap = 'hsv')
        
        axs324 = fig.add_subplot(234)
        axs324.scatter(x, y, c = result_umap_2D_data['M3'],  s = 10, cmap = 'hsv')
        
        axs325 = fig.add_subplot(235)
        axs325.scatter(x, y, c = result_umap_2D_data['M4'],  s = 10, cmap = 'hsv')
        
        axs326 = fig.add_subplot(236)
        axs326.scatter(x, y, c = result_umap_2D_data['M5'],  s = 10, cmap = 'hsv')

        # set titles for each subplot
        axs321.set_title(f"n_neighbors : {neighbors[0]}"); axs321.set_xlabel(f'{type[0]} 1'); axs321.set_ylabel(f'{type[0]} 2')
        axs322.set_title(f"n_neighbors : {neighbors[1]}"); axs322.set_xlabel(f'{type[0]} 1'); axs322.set_ylabel(f'{type[0]} 2')
        axs323.set_title(f"n_neighbors : {neighbors[2]}"); axs323.set_xlabel(f'{type[0]} 1'); axs323.set_ylabel(f'{type[0]} 2')
        axs324.set_title(f"n_neighbors : {neighbors[3]}"); axs324.set_xlabel(f'{type[0]} 1'); axs324.set_ylabel(f'{type[0]} 2')
        axs325.set_title(f"n_neighbors : {neighbors[4]}"); axs325.set_xlabel(f'{type[0]} 1'); axs325.set_ylabel(f'{type[0]} 2')
        axs326.set_title(f"n_neighbors : {neighbors[5]}"); axs326.set_xlabel(f'{type[0]} 1'); axs326.set_ylabel(f'{type[0]} 2')

        # display the figure
        plt.show()

        # save figure
        plt.savefig(f'./reports/{type}_2D_scatterplot_manyNeighbors.png')