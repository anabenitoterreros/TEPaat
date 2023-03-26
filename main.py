import pandas as pd
import pickle
from sklearn.metrics import accuracy_score

class package:
    def __init__(self):
        pass
        
    # read excel sheet
    def load_data(self, path_to_file, no_toplabel = True):
        """
        params: path, path to an excel file; no_toplabel indicates the presence of label columns in the first row
        return: a data frame"""

        if no_toplabel == True:
            col = None  
        else:
            col = 0

        data = pd.read_excel(path_to_file, index_col = col).dropna()
        
        return data
    
    # predict data based on fitted model
    def predict(self, X):
        """
        params: features X
        return: predictions """
        # load model
        scaler = pickle.load(open("./models/scaler_MinMax_model.pkl", 'rb'))
        umap = pickle.load(open("./models/umap_DBSCAN_model.pkl", 'rb'))
        ann = pickle.load(open("./models/ANN_DBSCAN_classifier_model.pkl", 'rb'))
        svc = pickle.load(open("./models/SVC_DBSCAN_classifier_model.pkl", 'rb'))
        # normalize data
        norm_data = scaler.predict(X)
        # reduce the data
        umap_data = umap.transform(norm_data)
        # classify data
        ann_data = ann.predict(umap_data)
        #svc_data = svc.predict(umap_data)
        return ann_data

    # compute the accuracy of a model
    def eval_accuracy(self, predictions, target):
        """
        params: predictions and target
        return: accuracy score """

        return accuracy_score(target, predictions)