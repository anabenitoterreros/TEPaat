import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessing():
    def __init__(self):
        pass

    def read_data(self, path):
        data = pd.read_excel(path, index_col = 0).dropna()

        return data
    
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

    def scale_data(self):
        # create scaler
        scaler = MinMaxScaler()
        # fit data and then transform
        norm_data = scaler.fit_transform(self.data)
        # convert norm_data to dataframe
        self.norm_data = pd.DataFrame(norm_data, columns=list(self.data.columns))
        return self.norm_data

