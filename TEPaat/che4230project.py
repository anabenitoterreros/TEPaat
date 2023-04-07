import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import wget
import zipfile

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
        # load models
        # same as those in the models folder of the main project directory 
        # ensure that the working directory is the same as trained models directory
  
        # get files
        url_umap = 'https://github.com/anabenitoterreros/TEPaat/blob/main/trained_models.zip'
        
        wget.download(url_umap)

        with zipfile.ZipFile("trained_models.zip","r") as zip_ref:
            zip_ref.extractall("targetdir")
    
        

        scaler = pickle.load(open("targetdir/scaler_MinMax_model.pkl", 'rb'))
        umap = pickle.load(open("targetdir/umap_model.pkl", 'rb'))
        if type == "ANN":
            classifier = pickle.load(open("targetdir/ANN_DBSCAN_classifier_model.pkl", 'rb'))
        if type == "SVC":
            classifier = pickle.load(open("targetdir/SVC_DBSCAN_classifier_model.pkl", 'rb'))
        
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
