import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import warnings
warnings.filterwarnings('ignore')


class MinMaxScaler_1Bound(BaseEstimator, TransformerMixin):
    def __init__(self, with_max=True, X_max=True, X_min=True, X_scaled=True, X_bound=True):
        self.with_max = with_max
        self.X_max = X_max
        self.X_min = X_min
        self.X_scaled = X_scaled
        self.X_bound = X_bound

    # fit: save min & max value of X
    def fit(self, X=None):
        return self

    # transform: scaling X
    def transform(self, X):
        X_max = X.max()+10
        X_min = np.abs(X.min()-10)
        X_bound = X_max if X_max > np.abs(X_min) else X_min

        if self.with_max:
            self.max_ = X_bound

        X_scaled = X/X_bound

        return X_scaled


def scale_input(x1, x2, max):
    input = np.stack((x1, x2), axis=2)
    sc_input = (np.reshape(input, (-1, 1)) / max).reshape(input.shape)
    sc_input = np.transpose(sc_input, (1, 2, 0))

    return sc_input


def std_scaling(x, mean, std):
    return (x-mean)/std


def rev_std_scaling(x, mean, std):
    return x*std+mean