# load and manipulate functions
import pandas as pd
import numpy as np
# plot functions
import matplotlib.pyplot as plt
import seaborn as sns
# 
import time
import warnings
#
# cluster function
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice


class ClusterAnalysis():
    def __init__(self, data):
        self.data = data
    
    def SVMcluster(self):
        pass

    def ANNcluster(self):
        pass