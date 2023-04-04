import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

class DataPreprocessing():
    def __init__(self):
        pass
    
    # read excel sheet
    def read_data(self, path):
        data = pd.read_excel(path, index_col = 0).dropna()

        return data
    
    # load or merge data
    def load_data(self, input_path, target_path = None, is_dataframe = True, justFeatures = True):
        
        input = self.read_data(input_path)

        if (justFeatures == True )or (target_path == None): # returns only the features w/o label
            self.data = input
        else:
            target = self.read_data(target_path)
            self.data = pd.concat([input, target], axis=1)

        if is_dataframe == True:  # returns a dataframe
            self.data = self.data
        else:
            self.data = self.data.to_numpy()

        return self.data

    # scale data 
    def scale_data(self):
        # create scaler
        self.scaler = MinMaxScaler()
        # fit data and then transform
        norm_data = self.scaler.fit_transform(self.data)
        # convert norm_data to dataframe
        self.norm_data = pd.DataFrame(norm_data, columns=list(self.data.columns))
        return self.norm_data

    # split data
    def split_data(self, X, y = None, testratio = 0.2, state = 42):
        if y is None:
            X_train, X_test = train_test_split(X, test_size = testratio, random_state = state)
            return (X_train, X_test)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testratio, random_state = state)
            return (X_train, X_test, y_train, y_test)

    def save_model(self, filename = "scaler_MinMax"):
        """
        returns the fitted model.
        """
        # Save the model to a file using pickle
        pickle.dump(self.scaler, open(f'./models/{filename}_model.pkl', 'wb'))
