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


class SubsetGenerator:
    def __init__(self, system='AgNi', extrap=False):
        self.system = system
        self.extrap = extrap
        self.system_dir = f'files/data/{system}'
        self.subsets = ['train', 'test']

        if extrap:
            self.input_info = pickle_load('files/data/AgNi/feats_input_info.pkl')
            self.target_info = pickle_load('files/data/AgNi/target_info.pkl')
        else:
            self.input_info = pickle_load(os.path.join(self.system_dir, 'feats_input_info.pkl'))
            self.target_info = pickle_load(os.path.join(self.system_dir, 'target_info.pkl'))

        CO2_dict = pickle_load(os.path.join(self.system_dir, 'CO2.pkl'))
        conditions = list(CO2_dict.keys())
        catalyst, voltage = [], []
        for cond in conditions:
            catalyst.append(cond.split('_')[0])
            voltage.append(float(cond.split('_')[1]))
        self.catalyst = sorted(list(set(catalyst)))
        self.voltage = sorted(list(set(voltage)))[::-1]

    @staticmethod
    def split_input_target(input, target, rand_seed=74):
        input_train, input_test = train_test_split(input, test_size=0.2, random_state=rand_seed)
        target_train, target_test = train_test_split(target, test_size=0.2, random_state=rand_seed)

        x = {'train': input_train, 'test': input_test}
        y = {'train': target_train, 'test': target_test}

        return x, y

    def __call__(self, rand_seed=72):
        x_list, y_list = defaultdict(list), defaultdict(list)
        for cat in self.catalyst:
            for v in self.voltage:
                feats = load_feats(cat, v)
                target = load_target(cat, v)
                if self.system == 'Zn' and self.extrap:
                    target = target[:, :3]

                sc_feats = std_scaling(feats, self.input_info['mean'], self.input_info['std'])
                sc_target = std_scaling(target, self.target_info['mean'], self.target_info['std'])
                if self.extrap:
                    pickle_dump(sc_feats[:9], os.path.join(self.system_dir, 'feats_input_extrap.pkl'))
                    pickle_dump(sc_target[:9], os.path.join(self.system_dir, 'feats_target_extrap.pkl'))
                    return

                x, y = self.split_input_target(sc_feats, sc_target)

                for s in self.subsets:
                    x_list[s].append(x[s])
                    y_list[s].append(y[s])

        input, target = {}, {}
        for set in x_list:
            input_set = concatenate_data(x_list[set])
            target_set = concatenate_data(y_list[set])
            input[set] = input_set
            target[set] = target_set

        if self.system == 'AgNi':
            shuffle_idx = np.arange(input['train'].shape[0])
            np.random.seed(rand_seed)
            np.random.shuffle(shuffle_idx)
            input['train'] = input['train'][shuffle_idx, :].astype(float)
            target['train'] = target['train'][shuffle_idx, :].astype(float)
        else:
            pass

        for s in input:
            pickle_dump(input[s], os.path.join(self.system_dir, f'feats_input_{s}.pkl'))
            pickle_dump(target[s], os.path.join(self.system_dir, f'feats_target_{s}.pkl'))


def FeatsDataset(subset, system='AgNi'):
    input = pickle_load(f'files/data/{system}/feats_input_{subset}.pkl')
    target = pickle_load(f'files/data/{system}/feats_target_{subset}.pkl')

    return input, target


if __name__ == '__main__':
    generator = SubsetGenerator(system='AgNi')
    generator()

    generator = SubsetGenerator(system='Zn')
    generator()

    generator = SubsetGenerator(system='Zn', extrap=True)
    generator()