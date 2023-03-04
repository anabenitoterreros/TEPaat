import pandas as pd

class DataPreprocessing():
    def __init__(self):
        pass

    def read_data(self, path):
        data = pd.read_excel(path, index_col = 0).dropna()

        return data
    
    def load_data(self, input_path, target_path, is_dataframe):
        input = self.read_data(input_path)
        target = self.read_data(target_path)

        self.data = pd.concat([input, target], axis=1)

        if is_dataframe == True:
            self.data = self.data
        else:
            self.data = self.data.to_numpy()

        return self.data
