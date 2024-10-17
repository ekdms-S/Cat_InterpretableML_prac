import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import math
import os

from utils import pickle_load, pickle_dump, to_path, create_directory
from data.data_utils import find_nearest_idx, concatenate_data, load_LSV


class FeatureExtraction:
    def __init__(self):
        self.V_34 = pickle_load('files/data/V_34.pkl')

        idx_i = []
        fixed_voltage = [-1.8, -2.0, -2.2, -2.4, -2.6, -2.8, -3.0, -3.2, -3.4]
        for i in fixed_voltage:
            idx = find_nearest_idx(self.V_34, i)
            idx_i.append(idx)

        self.idx_i = idx_i

        self.feats = ['ONSET',  # onset_V
                     'i_1.8', 'i_2.0', 'i_2.2', 'i_2.4', 'i_2.6', 'i_2.8', 'i_3.0', 'i_3.2', 'i_3.4',  # i_value
                     'DIFF_1.8-2.0', 'DIFF_2.0-2.2', 'DIFF_2.2-2.4', 'DIFF_2.4-2.6', 'DIFF_2.6-2.8', 'DIFF_2.8-3.0',
                     'DIFF_3.0-3.2', 'DIFF_3.2-3.4',  # i_diff
                     'AVG_1.8-2.0', 'AVG_2.0-2.2', 'AVG_2.2-2.4', 'AVG_2.4-2.6', 'AVG_2.6-2.8', 'AVG_2.8-3.0',
                     'AVG_3.0-3.2', 'AVG_3.2-3.4',  # i_avg
                     'TAN_1.8', 'TAN_2.0', 'TAN_2.2', 'TAN_2.4', 'TAN_2.6', 'TAN_2.8', 'TAN_3.0', 'TAN_3.2', 'TAN_3.4',
                     # i_tan
                     'TAFEL_2.4-2.6', 'TAFEL_2.6-2.8', 'TAFEL_2.8-3.0', 'TAFEL_3.0-3.2', 'TAFEL_3.2-3.4',  # i_tafel
                     'N2_DIFF_1.8', 'N2_DIFF_2.0', 'N2_DIFF_2.2', 'N2_DIFF_2.4', 'N2_DIFF_2.6', 'N2_DIFF_2.8',
                     'N2_DIFF_3.0', 'N2_DIFF_3.2', 'N2_DIFF_3.4']  # i_N2_diff

    # 1. onset potential
    def extract_onset_V(self, CO2_LSV):
        onset_i = -50
        idx_min = find_nearest_idx(self.V_34, -1.34)

        onset_V = []
        for i in range(self.len_data):
            current = CO2_LSV[idx_min:, i]
            idx = find_nearest_idx(current, onset_i) + idx_min
            onset = self.V_34[idx]
            onset_V.append(onset)

        onset_V = np.array(onset_V)

        return np.expand_dims(onset_V, axis=1)

    # 2. current value at voltage points
    def extract_i_value(self, CO2_LSV, voltage):
        feat = []
        for j in range(self.len_data):
            feat_ = []
            for i, idx in enumerate(self.idx_i):
                if voltage == -3.2:
                    i_feat = 0 if i >= len(self.idx_i) - 1 else CO2_LSV[idx, j]
                elif voltage == -3.4:
                    i_feat = CO2_LSV[idx, j]
                elif voltage == -3.0:
                    i_feat = 0 if i >= len(self.idx_i) - 2 else CO2_LSV[idx, j]
                else:
                    raise Exception(f'There is no {voltage}V condition')
                feat_.append(i_feat)
            feat.append(feat_)

        feat = np.array(feat)  # 2550, 9

        return feat

    # 3. current difference in voltage range
    def extract_i_diff(self, i_value, voltage):
        feat = []
        for j in range(self.len_data):
            feat_ = []
            for i in range(len(self.idx_i) - 1):
                if voltage == -3.2:
                    i_feat = 0 if i >= len(self.idx_i) - 2 else abs(i_value[j, i + 1] - i_value[j, i])
                elif voltage == -3.4:
                    i_feat = abs(i_value[j, i + 1] - i_value[j, i])
                elif voltage == -3.0:
                    i_feat = 0 if i >= len(self.idx_i) - 3 else abs(i_value[j, i + 1] - i_value[j, i])
                else:
                    raise Exception(f'There is no {voltage}V condition')
                feat_.append(i_feat)
            feat.append(feat_)

        feat = np.array(feat)

        return feat

    # 4. current average in voltage range
    def extract_i_avg(self, CO2_LSV, voltage):
        feat = []
        for j in range(self.len_data):
            feat_ = []
            for i in range(len(self.idx_i) - 1):
                if voltage == -3.2:
                    i_feat = 0 if i >= len(self.idx_i) - 2 else np.mean(CO2_LSV[self.idx_i[i]:self.idx_i[i + 1], j])
                elif voltage == -3.4:
                    i_feat = np.mean(CO2_LSV[self.idx_i[i]:self.idx_i[i + 1], j])
                elif voltage == -3.0:
                    i_feat = 0 if i >= len(self.idx_i) - 3 else np.mean(CO2_LSV[self.idx_i[i]:self.idx_i[i + 1], j])
                else:
                    raise Exception(f'There is no {voltage}V condition')
                feat_.append(i_feat)
            feat.append(feat_)

        feat = np.array(feat)

        return feat

    # 5. tangent line slope at voltage points
    def extract_i_tan(self, CO2_LSV, voltage):
        feat = []
        for j in range(self.len_data):
            feat_ = []
            for i, idx in enumerate(self.idx_i):
                if voltage == -3.2:
                    i_feat = 0 if i >= len(self.idx_i) - 1 else abs(
                        (CO2_LSV[idx - 1, j] - CO2_LSV[idx, j]) / (self.V_34[idx - 1] - self.V_34[idx]))
                elif voltage == -3.4:
                    i_feat = abs((CO2_LSV[idx - 1, j] - CO2_LSV[idx, j]) / (self.V_34[idx - 1] - self.V_34[idx]))
                elif voltage == -3.0:
                    i_feat = 0 if i >= len(self.idx_i) - 2 else abs(
                        (CO2_LSV[idx - 1, j] - CO2_LSV[idx, j]) / (self.V_34[idx - 1] - self.V_34[idx]))
                else:
                    raise Exception(f'There is no {voltage}V condition')
                feat_.append(i_feat)
            feat.append(feat_)

        feat = np.array(feat)

        return feat

    # 6. slope on tafel plot at voltage points
    def extract_tafel(self, i_value, voltage):
        feat = []
        for j in range(self.len_data):
            feat_ = []
            for i in range(len(self.idx_i) - 1):
                if voltage == -3.2:
                    if i >= len(self.idx_i) - 5 and i <= len(self.idx_i) - 3:
                        i_feat = abs((self.V_34[self.idx_i[i + 1]] - self.V_34[self.idx_i[i]]) / (
                                    math.log10(-i_value[j, i + 1]) - math.log10(-i_value[j, i])))
                    else:
                        i_feat = 0
                elif voltage == -3.4:
                    if i >= len(self.idx_i) - 2:
                        i_feat = abs((self.V_34[self.idx_i[i + 1]] - self.V_34[self.idx_i[i]]) / (
                                    math.log10(-i_value[j, i + 1]) - math.log10(-i_value[j, i])))
                    else:
                        i_feat = 0
                elif voltage == -3.0:
                    if i >= len(self.idx_i) - 6 and i <= len(self.idx_i) - 4:
                        i_feat = abs((self.V_34[self.idx_i[i + 1]] - self.V_34[self.idx_i[i]]) / (
                                    math.log10(-i_value[j, i + 1]) - math.log10(-i_value[j, i])))
                    else:
                        i_feat = 0
                else:
                    raise Exception(f'There is no {voltage}V condition')
                feat_.append(i_feat)
            feat.append(feat_)

        feat = np.array(feat)

        return feat[:, 3:]

    # 7. current difference between CO2 LSV and N2 LSV
    def extract_i_N2_diff(self, CO2_LSV, N2_LSV, voltage):
        feat = []
        for j in range(self.len_data):
            feat_ = []
            for i, idx in enumerate(self.idx_i):
                if voltage == -3.2:
                    i_feat = 0 if i >= len(self.idx_i) - 1 else CO2_LSV[idx, j] - N2_LSV[idx, j]
                elif voltage == -3.4:
                    i_feat = CO2_LSV[idx, j] - N2_LSV[idx, j]
                elif voltage == -3.0:
                    i_feat = 0 if i >= len(self.idx_i) - 2 else CO2_LSV[idx, j] - N2_LSV[idx, j]
                else:
                    raise Exception(f'There is no {voltage}V condition')
                feat_.append(i_feat)
            feat.append(feat_)

        feat = np.array(feat)

        return feat

    def _extract(self, CO2_LSV, N2_LSV, voltage):
        onset_V = self.extract_onset_V(CO2_LSV)
        i_value = self.extract_i_value(CO2_LSV, voltage)
        i_diff = self.extract_i_diff(i_value, voltage)
        i_avg = self.extract_i_avg(CO2_LSV, voltage)
        i_tan = self.extract_i_tan(CO2_LSV, voltage)
        tafel = self.extract_tafel(i_value, voltage)
        i_N2_diff = self.extract_i_N2_diff(CO2_LSV, N2_LSV, voltage)

        features = [onset_V, i_value, i_diff, i_avg, i_tan, tafel, i_N2_diff]
        features = concatenate_data(features, axis=1)

        return features

    def __call__(self, catalyst, voltage, save=True):
        CO2, N2 = load_LSV(catalyst, voltage)
        self.len_data = CO2.shape[1]

        features = self._extract(CO2, N2, voltage)
        feats_df = pd.DataFrame(features, columns=self.feats)

        if save:
            if catalyst in ['Ag', 'Ni']:
                feats_dir = f'files/data/AgNi/{catalyst}/features'
            else:
                feats_dir = 'files/data/Zn/features'
            create_directory(feats_dir, path_is_directory=True)
            feats_df.to_csv(to_path(os.path.join(feats_dir, f'{catalyst}_{voltage}_features.csv')), index=False)

        return feats_df


if __name__ == '__main__':
    extractor = FeatureExtraction()

    catalyst = ['Ag', 'Ni']
    voltage = [-3.2, -3.4, -3.0]
    for cat in catalyst:
        for v in voltage:
            extractor(cat, v)
    extractor('Zn', -3.4)