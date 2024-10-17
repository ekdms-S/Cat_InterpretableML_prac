import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import defaultdict

from utils import *
from data.data_utils import *
from data.scaler import scale_input, std_scaling


class SubsetGenerator:
    def __init__(self, system='AgNi', extrap=False):
        self.system = system
        self.extrap = extrap
        self.system_dir = f'files/data/{system}'
        self.subsets = ['train', 'valid', 'test']

        ref = 'N2' if system == 'AgNi' else 'Ar'
        self.CO2_dict = pickle_load(os.path.join(self.system_dir, 'CO2.pkl'))
        self.ref_dict = pickle_load(os.path.join(self.system_dir, f'{ref}.pkl'))
        if extrap:
            self.input_info = pickle_load('files/data/AgNi/LSV_input_info.pkl')
            self.target_info = pickle_load('files/data/AgNi/target_info.pkl')
        else:
            self.input_info = pickle_load(os.path.join(self.system_dir, 'LSV_input_info.pkl'))
            self.target_info = pickle_load(os.path.join(self.system_dir, 'target_info.pkl'))

        conditions = list(self.CO2_dict.keys())
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

        num_train = int(input_train.shape[0] * 0.8)
        x_train, y_train = input_train[:num_train, :, :], target_train[:num_train, :]
        x_valid, y_valid = input_train[num_train:, :, :], target_train[num_train:, :]
        x_test, y_test = input_test, target_test

        x = {'train': x_train, 'valid': x_valid, 'test': x_test}
        y = {'train': y_train, 'valid': y_valid, 'test': y_test}

        return x, y

    def __call__(self, rand_seed=72):
        x_list, y_list = defaultdict(list), defaultdict(list)
        for cat in self.catalyst:
            for v in self.voltage:
                CO2_input, ref_input = self.CO2_dict[f'{cat}_{v}'], self.ref_dict[f'{cat}_{v}']
                target = load_target(cat, v)
                if self.system == 'Zn' and self.extrap:
                    target = target[:, :3]

                sc_input = scale_input(CO2_input, ref_input, self.input_info['max'])
                sc_target = std_scaling(target, self.target_info['mean'], self.target_info['std'])
                if self.extrap:
                    pickle_dump(sc_input[:9], os.path.join(self.system_dir, 'LSV_input_extrap.pkl'))
                    pickle_dump(sc_target[:9], os.path.join(self.system_dir, 'LSV_target_extrap.pkl'))
                    return

                x, y = self.split_input_target(sc_input, sc_target)

                if self.system == 'AgNi':
                    catalyst_dir = os.path.join(self.system_dir, cat)
                else:
                    catalyst_dir = self.system_dir
                pickle_dump(x['test'], os.path.join(catalyst_dir, f'LSV_input_test_{cat}_{v}.pkl'))
                pickle_dump(y['test'], os.path.join(catalyst_dir, f'LSV_target_test_{cat}_{v}.pkl'))

                for s in self.subsets:
                    x_list[s].append(x[s])
                    y_list[s].append(y[s])

        input, target = {}, {}
        for s in x_list:
            input_set = concatenate_data(x_list[s])
            target_set = concatenate_data(y_list[s])
            input[s] = input_set
            target[s] = target_set

        if self.system == 'AgNi':
            shuffle_idx = np.arange(input['train'].shape[0])
            np.random.seed(rand_seed)
            np.random.shuffle(shuffle_idx)
            input['train'] = input['train'][shuffle_idx, :].astype(float)
            target['train'] = target['train'][shuffle_idx, :].astype(float)
        else:
            pass

        for s in input:
            pickle_dump(input[s], os.path.join(self.system_dir, f'LSV_input_{s}.pkl'))
            pickle_dump(target[s], os.path.join(self.system_dir, f'LSV_target_{s}.pkl'))


class LSVDataset(Dataset):
    def __init__(self, subset, system='AgNi', catalyst=None, voltage=None):
        self.subset = subset
        self.system = system

        system_dir = f'files/data/{system}'
        if catalyst is not None and voltage is not None:
            assert subset == 'test'
            if system == 'AgNi':
                catalyst_dir = os.path.join(system_dir, catalyst)
            else:
                catalyst_dir = system_dir
            self.input = pickle_load(os.path.join(catalyst_dir, f'LSV_input_{subset}_{catalyst}_{voltage}.pkl'))
            self.target = pickle_load(os.path.join(catalyst_dir, f'LSV_target_{subset}_{catalyst}_{voltage}.pkl'))
        else:
            self.input = pickle_load(os.path.join(system_dir, f'LSV_input_{subset}.pkl'))
            self.target = pickle_load(os.path.join(system_dir, f'LSV_target_{subset}.pkl'))

    def __getitem__(self, idx):
        self.x = torch.from_numpy(self.input[idx, :, :])
        self.ys = torch.from_numpy(np.array([self.target[idx, :]]))

        return self.x, self.ys

    def __len__(self):
        return len(self.input)


if __name__ == '__main__':
    generator = SubsetGenerator(system='AgNi')
    generator()

    generator = SubsetGenerator(system='Zn')
    generator()

    generator = SubsetGenerator(system='Zn', extrap=True)
    generator()