import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import os

class TEPaat:
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
    def predict(self, X, type = "ANN"):
        """
        params: features X
        return: predictions """
        print(os.getcwd())
        # load models
        # same as those in the models folder of the main project directory 
        # ensure that the working directory is the same as trained models directory
        
        # try:
        #     os.chir("./trained_models")
        # except:
        #     os.chdir("./TEPaat/trained_models")

        scaler = pickle.load(open("scaler_MinMax_model.pkl", 'rb'))
        umap = pickle.load(open("umap_model.pkl", 'rb'))
        if type == "ANN":
            classifier = pickle.load(open("ANN_DBSCAN_classifier_model.pkl", 'rb'))
        if type == "SVC":
            classifier = pickle.load(open("SVC_DBSCAN_classifier_model.pkl", 'rb'))
        
        # normalize data
        norm_data = scaler.transform(X)
        # reduce the data
        umap_data = umap.transform(norm_data)
        # classify data
        pred_label_data = classifier.predict(umap_data)

        return pred_label_data

    # compute the accuracy of a model
    def eval_accuracy(self, predictions, target):
        """
        params: predictions and target
        return: accuracy score """

        # due to inconsistency in the labels of the ground truth label and the predicted label. A correct factor was introduced.
        
        target = target['label'].replace({2: 1, 3: 3, 6: 4, 8: 0, 13: 2})

        return accuracy_score(target, predictions)
