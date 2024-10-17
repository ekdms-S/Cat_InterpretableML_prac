import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from collections import defaultdict

from utils import pickle_load, pickle_dump, to_path
from data.data_utils import concatenate_data, load_feats, load_target
from data.scaler import std_scaling


class FeatsPreprocessor:
    def __init__(self, system='AgNi'):
        self.system = system
        self.system_dir = f'files/data/{system}'
        CO2_dict = pickle_load(os.path.join(self.system_dir, 'CO2.pkl'))

        conditions = list(CO2_dict.keys())
        catalyst, voltage = [], []
        for cond in conditions:
            catalyst.append(cond.split('_')[0])
            voltage.append(float(cond.split('_')[1]))
        self.catalyst = sorted(list(set(catalyst)))
        self.voltage = sorted(list(set(voltage)))[::-1]
    
    def preprocess_input(self):
        features = []
        for cat in self.catalyst:
            for v in self.voltage:
                feats = load_feats(cat, v)
                features.append(feats)
        features = concatenate_data(features)

        scaler = StandardScaler()
        sc_features = scaler.fit_transform(features)
        mean, std = scaler.mean_, np.sqrt(scaler.var_)

        pickle_dump({'mean': mean, 'std': std}, os.path.join(self.system_dir, 'feats_input_info.pkl'))

if __name__ == '__main__':
    preprcs = FeatsPreprocessor(system='AgNi')
    preprcs.preprocess_input()

    preprcs = FeatsPreprocessor(system='Zn')
    preprcs.preprocess_input()