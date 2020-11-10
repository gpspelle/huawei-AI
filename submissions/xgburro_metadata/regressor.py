import numpy as np
from sklearn import multioutput
import xgboost as xgb

class Regressor():
    def _init_(self):
        super()._init_()
        self.model = None

    def fit(self, X, y):
        # Get data and create train data loaders
        X_56 = []
        for sample in X:
            metadata = sample[0]
            inp = sample[1]

            metadata_input = [0] * 8 * 3
            for i, j in zip(range(8), range(0, 24, 3)):
                if i >= len(metadata):
                    metadata_input[j] = 0
                    metadata_input[j+1] = 0
                    metadata_input[j+2] = 0
                else:
                    metadata_input[j] = 2 if metadata[i][0] == 'EDFA' else 1
                    try:
                        metadata_input[j+1] = metadata[i][1][0]
                    except:
                        metadata_input[j+1] = 0
                    try:
                        metadata_input[j+2] = metadata[i][1][1]
                    except:
                        metadata_input[j+2] = 0

            real_inp = inp + metadata_input
            X_56.append(np.asarray(real_inp))

        X_56 = np.asarray(X_56)
        self.model = multioutput.MultiOutputRegressor(xgb.XGBRegressor()).fit(X_56, y)

    def predict(self, X):

        X_56 = []
        for sample in X:
            metadata = sample[0]
            inp = sample[1]

            metadata_input = [0] * 8 * 3
            for i, j in zip(range(8), range(0, 24, 3)):
                if i >= len(metadata):
                    metadata_input[j] = 0
                    metadata_input[j+1] = 0
                    metadata_input[j+2] = 0
                else:
                    try:
                        metadata_input[j] = 2 if metadata[i][0] == 'EDFA' else 1
                    except:
                        metadata_input[j] = 0 
                    try:
                        metadata_input[j+1] = metadata[i][1][0]
                    except:
                        metadata_input[j+1] = 0
                    try:
                        metadata_input[j+2] = metadata[i][1][1]
                    except:
                        metadata_input[j+2] = 0



            X_56.append(inp + metadata_input)

        X_56 = np.asarray(X_56)
        preds = self.model.predict(X_56)
        preds = preds * (preds > 0)
        return preds
