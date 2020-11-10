
import numpy as np
from sklearn import multioutput
import xgboost as xgb


class Regressor():
    def _init_(self):
        super()._init_()
        self.model = None

    def fit(self, X, y):
        # Get data and create train data loaders
        X_32 = np.array([_[1] for _ in X])
        self.model = multioutput.MultiOutputRegressor(xgb.XGBRegressor()).fit(X_32, y)

    def predict(self, X):
        dtest = np.array([_ for _ in X[:,1]])
        preds = self.model.predict(dtest)
        preds = preds * (preds > 0)
        return preds
