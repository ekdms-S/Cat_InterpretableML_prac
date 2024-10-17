import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from sklearn.preprocessing import StandardScaler
import logging
import pandas as pd

from data.LSV_loader import RawLSVLoader_AgNi, RawLSVLodaer_Zn
from data.data_utils import *
from data.scaler import *
from utils import *

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def extract_onset_V(CO2_LSV, onset_i=-50):
    V_34 = pickle_load('files/data/V_34.pkl')
    idx_min = find_nearest_idx(V_34, -1.34)

    onset_V = []
    for i in range(CO2_LSV.shape[1]):
        current = CO2_LSV[idx_min:, i]
        idx = find_nearest_idx(current, onset_i) + idx_min
        onset = V_34[idx]
        onset_V.append(onset)

    onset_V = np.array(onset_V)

    return onset_V

class LSVPreprocessor_AgNi:
    def __init__(self, load=False):
        self.catalyst = ['Ag', 'Ni']
        self.voltage = [-3.2, -3.4, -3.0]
        self.Dir = 'files/data/AgNi'
        create_directory(self.Dir, path_is_directory=True)
        for cat in self.catalyst:
            lsv_dir = os.path.join(self.Dir, f'{cat}/from_raw')
            create_directory(lsv_dir, path_is_directory=True)
        if load:
            self.load_all()

    def load_all(self):
        loader = RawLSVLoader_AgNi()

        n_trial_list = [75, 19, 13, 20, 1, 5] # Ag_32, Ag_34, Ag_30, Ni_32, Ni_34, Ni_30
        i = 0
        for cat in self.catalyst:
            for v in self.voltage:
                logger.info(f'Load {cat} {v}V data ...')
                lsv_dir = os.path.join(self.Dir, f'{cat}/from_raw')

                if cat == 'Ag' and v == -3.2:
                    exclude_data = [2179, 2209, 2251]
                elif cat == 'Ni' and v == -3.2:
                    exclude_data = [192]
                    exclude_data += np.arange(657, 700).tolist()
                else:
                    exclude_data = None

                n_trial = n_trial_list[i]
                if cat == 'Ag' and v == -3.4:
                    CO2_i, CO2_v, N2_i, N2_v = loader(
                        cat, v, n_trial, exclude_data,
                        CO2_exclude_data=None, N2_exclude_data_a=None, N2_exclude_data_b=None
                    )
                    N2_i_a, N2_i_b = N2_i[0], N2_i[1]
                    N2_v_a, N2_v_b = N2_v[0],N2_v[1]

                    pickle_dump(CO2_i, os.path.join(lsv_dir, f'{cat}_{v}_CO2_i.pkl'))
                    pickle_dump(CO2_v, os.path.join(lsv_dir, f'{cat}_{v}_CO2_v.pkl'))
                    pickle_dump(N2_i_a, os.path.join(lsv_dir, f'{cat}_{v}_N2_i_a.pkl'))
                    pickle_dump(N2_i_b, os.path.join(lsv_dir, f'{cat}_{v}_N2_i_b.pkl'))
                    pickle_dump(N2_v_a, os.path.join(lsv_dir, f'{cat}_{v}_N2_v_a.pkl'))
                    pickle_dump(N2_v_b, os.path.join(lsv_dir, f'{cat}_{v}_N2_v_b.pkl'))
                else:
                    CO2_i, CO2_v, N2_i, N2_v = loader(
                        cat, v, n_trial, exclude_data
                    )

                    pickle_dump(CO2_i, os.path.join(lsv_dir, f'{cat}_{v}_CO2_i.pkl'))
                    pickle_dump(CO2_v, os.path.join(lsv_dir, f'{cat}_{v}_CO2_v.pkl'))
                    pickle_dump(N2_i, os.path.join(lsv_dir, f'{cat}_{v}_N2_i.pkl'))
                    pickle_dump(N2_v, os.path.join(lsv_dir, f'{cat}_{v}_N2_v.pkl'))

                i += 1

    def preprocess_input(self, plot_for_check=False):
        CO2, N2 = [], []
        CO2_dict, N2_dict = {}, {}
        for cat in self.catalyst:
            for v in self.voltage:
                logger.info(f'Curve Fitting: {cat} {v}V ...')
                lsv_dir = os.path.join(self.Dir, f'{cat}/from_raw')

                CO2_i = pickle_load(os.path.join(lsv_dir, f'{cat}_{v}_CO2_i.pkl'))
                CO2_v = pickle_load(os.path.join(lsv_dir, f'{cat}_{v}_CO2_v.pkl'))
                if cat == 'Ag' and v == -3.4:
                    N2_i_a = pickle_load(os.path.join(lsv_dir, f'{cat}_{v}_N2_i_a.pkl'))
                    N2_i_b = pickle_load(os.path.join(lsv_dir, f'{cat}_{v}_N2_i_b.pkl'))
                    N2_v_a = pickle_load(os.path.join(lsv_dir, f'{cat}_{v}_N2_v_a.pkl'))
                    N2_v_b = pickle_load(os.path.join(lsv_dir, f'{cat}_{v}_N2_v_b.pkl'))

                    CO2_fit, N2_fit = curve_fitting(
                        cat, v, CO2_i, CO2_v, N2_i_a, N2_v_a,
                        Ag_34_N2_i_b=N2_i_b, Ag_34_N2_v_b=N2_v_b
                    )
                else:
                    N2_i = pickle_load(os.path.join(lsv_dir, f'{cat}_{v}_N2_i.pkl'))
                    N2_v = pickle_load(os.path.join(lsv_dir, f'{cat}_{v}_N2_v.pkl'))

                    CO2_fit, N2_fit = curve_fitting(
                        cat, v, CO2_i, CO2_v, N2_i, N2_v
                    )

                pickle_dump(CO2_fit, os.path.join(self.Dir, f'{cat}/{cat}_{v}_CO2.pkl'))
                pickle_dump(N2_fit, os.path.join(self.Dir, f'{cat}/{cat}_{v}_N2.pkl'))

                onset_30 = extract_onset_V(CO2_fit, onset_i=-30)
                onset_50 = extract_onset_V(CO2_fit, onset_i=-50)
                pickle_dump(onset_30, os.path.join(self.Dir, f'{cat}/{cat}_{v}_onset_-30.pkl'))
                pickle_dump(onset_50, os.path.join(self.Dir, f'{cat}/{cat}_{v}_onset_-50.pkl'))

                CO2_pad = zero_padding(CO2_fit)
                N2_pad = zero_padding(N2_fit)

                CO2.append(CO2_pad)
                N2.append(N2_pad)
                CO2_dict[f'{cat}_{v}'] = CO2_pad
                N2_dict[f'{cat}_{v}'] = N2_pad

        CO2 = concatenate_data(CO2, axis=1)
        N2 = concatenate_data(N2, axis=1)
        current = np.stack((CO2.T, N2.T), axis=2)

        mm1b_scaler = MinMaxScaler_1Bound(with_max=True)
        sc_input = mm1b_scaler.fit_transform(np.reshape(current, (-1, 1))).reshape(current.shape)
        if plot_for_check:
            plot_scaled_result(current, sc_input)

        pickle_dump(CO2_dict, os.path.join(self.Dir, 'CO2.pkl'))
        pickle_dump(N2_dict, os.path.join(self.Dir, 'N2.pkl'))
        pickle_dump({'max': mm1b_scaler.max_}, os.path.join(self.Dir, 'LSV_input_info.pkl'))

    def preprocess_target(self):
        target = []
        for cat in self.catalyst:
            for v in self.voltage:
                tg = pd.read_csv(to_path(f'files/raw/{cat}/{v}V/targets.csv')).to_numpy()
                pickle_dump(tg, os.path.join(self.Dir, f'{cat}/{cat}_{v}_target.pkl'))
                target.append(tg)

        target = concatenate_data(target, axis=0)

        scaler = StandardScaler()
        sc_target = scaler.fit_transform(target)
        mean, std = scaler.mean_, np.sqrt(scaler.var_)

        pickle_dump({'mean': mean, 'std': std}, os.path.join(self.Dir, 'target_info.pkl'))


class LSVPreprocessor_Zn:
    def __init__(self, load=False):
        self.Dir = 'files/data/Zn'
        create_directory(self.Dir, path_is_directory=True)
        self.type = type
        if load:
            logger.info(f'Load Zn -3.4V data ...')
            CO2_i, CO2_v, Ar_i, Ar_v = RawLSVLodaer_Zn(voltage=-3.4)

            lsv_dir = os.path.join(self.Dir, f'from_raw')
            create_directory(lsv_dir, path_is_directory=True)
            pickle_dump(CO2_i, os.path.join(lsv_dir, 'Zn_-3.4_CO2_i.pkl'))
            pickle_dump(CO2_v, os.path.join(lsv_dir, 'Zn_-3.4_CO2_v.pkl'))
            pickle_dump(Ar_i, os.path.join(lsv_dir, 'Zn_-3.4_Ar_i.pkl'))
            pickle_dump(Ar_v, os.path.join(lsv_dir, 'Zn_-3.4_Ar_v.pkl'))

    def preprocess_input(self, plot_for_check=False):
        lsv_dir = os.path.join(self.Dir, f'from_raw')
        CO2_i = pickle_load(os.path.join(lsv_dir, 'Zn_-3.4_CO2_i.pkl'))
        CO2_v = pickle_load(os.path.join(lsv_dir, 'Zn_-3.4_CO2_v.pkl'))
        Ar_i = pickle_load(os.path.join(lsv_dir, 'Zn_-3.4_Ar_i.pkl'))
        Ar_v = pickle_load(os.path.join(lsv_dir, 'Zn_-3.4_Ar_v.pkl'))

        logger.info(f'Curve Fitting: Zn -3.4V ...')
        CO2_fit, Ar_fit = curve_fitting('Zn', -3.4, CO2_i, CO2_v, Ar_i, Ar_v)

        current = np.stack((CO2_fit.T, Ar_fit.T), axis=2)
        mm1b_scaler = MinMaxScaler_1Bound(with_max=True)
        sc_input = mm1b_scaler.fit_transform(np.reshape(current, (-1, 1))).reshape(current.shape)

        CO2_fit, Ar_fit = CO2_fit[:, 1:], Ar_fit[:, 1:]  # exclude data w/ deactivated i_tot
        pickle_dump(CO2_fit, os.path.join(self.Dir, 'Zn_-3.4_CO2.pkl'))
        pickle_dump(Ar_fit, os.path.join(self.Dir, 'Zn_-3.4_Ar.pkl'))

        onset_30 = extract_onset_V(CO2_fit, onset_i=-30)
        onset_50 = extract_onset_V(CO2_fit, onset_i=-50)
        pickle_dump(onset_30, os.path.join(self.Dir, 'Zn_-3.4_onset_-30.pkl'))
        pickle_dump(onset_50, os.path.join(self.Dir, 'Zn_-3.4_onset_-50.pkl'))

        CO2_pad = zero_padding(CO2_fit)
        Ar_pad = zero_padding(Ar_fit)

        if plot_for_check:
            plot_scaled_result(current, sc_input)

        pickle_dump({'Zn_-3.4': CO2_pad}, os.path.join(self.Dir, 'CO2.pkl'))
        pickle_dump({'Zn_-3.4': Ar_pad}, os.path.join(self.Dir, 'Ar.pkl'))
        pickle_dump({'max': mm1b_scaler.max_}, os.path.join(self.Dir, 'LSV_input_info.pkl'))

    def preprocess_target(self):
        target = pd.read_csv(to_path('files/raw/Zn/-3.4V/targets.csv')).to_numpy()

        scaler = StandardScaler()
        sc_target = scaler.fit_transform(target)
        mean, std = scaler.mean_, np.sqrt(scaler.var_)

        target = target[1:] # exclude data w/ deactivated i_tot
        pickle_dump(target, os.path.join(self.Dir, 'Zn_-3.4_target.pkl'))

        pickle_dump({'mean': mean, 'std': std}, os.path.join(self.Dir, 'target_info.pkl'))


if __name__ == '__main__':
    V = np.arange(-0.0055, -3.4 - 0.001, -0.0025)
    pickle_dump(V, 'files/data/V_34.pkl')

    preprcs = LSVPreprocessor_AgNi(load=True)
    preprcs.preprocess_input(plot_for_check=True)
    preprcs.preprocess_target()

    preprcs = LSVPreprocessor_Zn(load=True)
    preprcs.preprocess_input()
    preprcs.preprocess_target()