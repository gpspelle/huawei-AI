import numpy as np
from sklearn.metrics import make_scorer
import rampwf as rw
from sklearn import multioutput
import xgboost as xgb
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model.base import BaseEstimator
from problem import get_train_data

def EM99(y_true, y_pred):
    precision=3
    quant=0.99
    eps=1e-8

    if (y_pred < 0).any():
        return float('inf')

    ratio_err = np.array([(p + eps) / t for y_hat, y in zip(y_pred, y_true)
                        for p, t in zip(y_hat, y) if t != 0])
    # sorted absolute value of mw2dB ratio err
    score = np.percentile(np.abs(10 * np.log10(ratio_err)), 100 * quant)
    return score


class Regressor(BaseEstimator):

    def __init__(self, max_depth=2, n_estimators=60, learning_rate=0.1):
        super().__init__()
        self.model = None

        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.model = multioutput.MultiOutputRegressor(xgb.XGBRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate))

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
        self.model.fit(X_56, y, eval_metric="logloss", verbose=True)
        
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


parameters = [{
    'max_depth': [3, 4, 5, 6, 7],
    'n_estimators': [50, 250, 500, 1000],
    'learning_rate': [0.03, 0.05, 0.08, 0.1]
}]
EM99_score = make_scorer(EM99, greater_is_better=False)

'''
parameters = [{
    'max_depth': [3, 4],
    'learning_rate': [0.03, 0.05]
}]
'''

grid_search = GridSearchCV(
    estimator=Regressor(),
    param_grid=parameters,
    scoring = EM99_score,
    n_jobs = 8,
    cv = 4,
    verbose=True
)

x_train, y_train = get_train_data()

grid_search.fit(x_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)

