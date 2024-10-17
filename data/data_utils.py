import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import gridspec

from utils import pickle_load, to_path


def get_exclude_idx(data, idx=None, catalyst=None, voltage=None):
    """
    Ignore first loop of every TRIAL of whole data due to pressure noise of GC
    """
    # last Ni AEM 3.0V LSV data(i.e. trial4 loop70) has only 9 data points
    #   -> exclude last loop
    if catalyst == 'Ni' and voltage == -3.0:
        len_data = len(data.T) + 1
    else:
        len_data = len(data.T)
    num_trial = len_data//70

    exclude_idx = []
    for i in range(num_trial):
        exclude_idx.append(i*70)

    if idx is not None:
        exclude_idx = exclude_idx + idx

    exclude_idx.sort()

    return exclude_idx


def curve_fitting(catalyst, voltage, CO2_i, CO2_v, N2_i, N2_v, **kwargs):
    V = np.arange(-0.0055, voltage-0.001, -0.0025)

    if catalyst == 'Ag' and voltage == -3.4:
        N2_i_a = N2_i
        N2_i_b = kwargs['Ag_34_N2_i_b']
        N2_v_a = N2_v
        N2_v_b = kwargs['Ag_34_N2_v_b']

        a = N2_v_a.shape[1]
        for j in range(CO2_v.shape[1]):
            if j < a:
                f_CO2 = interp1d(CO2_v[:, j], CO2_i[:, j])
                f_N2_a = interp1d(N2_v_a[:, j], N2_i_a[:, j])
                if j == 0:
                    CO2_fit = np.reshape(f_CO2(V), (len(V), 1))
                    N2_fit = np.reshape(f_N2_a(V), (len(V), 1))
                else:
                    CO2_fit_ = np.reshape(f_CO2(V), (len(V), 1))
                    N2_a_fit_ = np.reshape(f_N2_a(V), (len(V), 1))
                    CO2_fit = np.concatenate((CO2_fit, CO2_fit_), axis=1)
                    N2_fit = np.concatenate((N2_fit, N2_a_fit_), axis=1)
            else:
                f_CO2 = interp1d(CO2_v[:, j], CO2_i[:, j])
                f_N2_b = interp1d(N2_v_b[:, j - a], N2_i_b[:, j - a])
                CO2_fit_ = np.reshape(f_CO2(V), (len(V), 1))
                N2_b_fit_ = np.reshape(f_N2_b(V), (len(V), 1))
                CO2_fit = np.concatenate((CO2_fit, CO2_fit_), axis=1)
                N2_fit = np.concatenate((N2_fit, N2_b_fit_), axis=1)
    else:
        for j in range(CO2_v.shape[1]):
            f_CO2 = interp1d(CO2_v[:, j], CO2_i[:, j])
            f_N2 = interp1d(N2_v[:, j], N2_i[:, j])
            if j == 0:
                CO2_fit = np.reshape(f_CO2(V), (len(V), 1))
                N2_fit = np.reshape(f_N2(V), (len(V), 1))
            else:
                CO2_fit_ = np.reshape(f_CO2(V), (len(V), 1))
                N2_fit_ = np.reshape(f_N2(V), (len(V), 1))
                CO2_fit = np.concatenate((CO2_fit, CO2_fit_), axis=1)
                N2_fit = np.concatenate((N2_fit, N2_fit_), axis=1)

    return CO2_fit, N2_fit


def zero_padding(x, len_data=1424):
    x = x.T.tolist()
    for x_i in x:
        while len(x_i) < len_data:
            x_i.append(0)
    x_pad = np.array(x).T

    return x_pad


def plot_scaled_result(x, sc_x):
    fig = plt.figure(figsize=(13.5, 4.5))
    spec = gridspec.GridSpec(ncols=2, nrows=1)
    num = len(x)//50
    ax1 = plt.subplot(spec[0])
    ax2 = plt.subplot(spec[1])

    for i in range(num):
        ax1.plot(x[i*50, :, 1], color='green')
        ax1.plot(x[i*50, :, 0], color='orange')
        ax2.plot(sc_x[i*50, :, 1], color='green')
        ax2.plot(sc_x[i*50, :, 0], color='orange')
    ax1.plot(x[num*50, :, 1], color='green', label='N2')
    ax1.plot(x[num*50, :, 0], color='orange', label='CO2')
    ax2.plot(sc_x[num*50, :, 1], color='green', label='N2')
    ax2.plot(sc_x[num*50, :, 0], color='orange', label='CO2')

    ax1.set_ylabel('current [mA]')
    ax1.grid(True)
    ax1.set_title('LSV curve')
    ax1.legend()
    ax2.set_ylabel('scaled current')
    ax2.grid(True)
    ax2.set_title('scaled LSV curve')
    ax2.legend()

    plt.show()


def concatenate_data(x_list, axis=0):
    for i, x in enumerate(x_list):
        if i == 0:
            concat_x = x
        else:
            concat_x = np.concatenate((concat_x, x), axis=axis)

    return concat_x


def find_nearest_idx(x, value):
    return np.abs(x - value).argmin()


def load_LSV(catalyst, voltage):
    if catalyst in ['Ag', 'Ni']:
        CO2 = pickle_load(f'files/data/AgNi/{catalyst}/{catalyst}_{voltage}_CO2.pkl')
        ref = pickle_load(f'files/data/AgNi/{catalyst}/{catalyst}_{voltage}_N2.pkl')
    else:
        CO2 = pickle_load(f'files/data/{catalyst}/{catalyst}_{voltage}_CO2.pkl')
        ref = pickle_load(f'files/data/{catalyst}/{catalyst}_{voltage}_Ar.pkl')

    return CO2, ref

def load_feats(catalyst, voltage):
    if catalyst in ['Ag', 'Ni']:
        feats = pd.read_csv(to_path(f'files/data/AgNi/{catalyst}/features/{catalyst}_{voltage}_features.csv')).to_numpy()
    else:
        feats = pd.read_csv(to_path(f'files/data/{catalyst}/features/{catalyst}_{voltage}_features.csv')).to_numpy()

    return feats

def load_target(catalyst, voltage):
    if catalyst in ['Ag', 'Ni']:
        target = pickle_load(f'files/data/AgNi/{catalyst}/{catalyst}_{voltage}_target.pkl')
    else:
        target = pickle_load(f'files/data/{catalyst}/{catalyst}_{voltage}_target.pkl')

    return target


def load_onset(catalyst, voltage, onset_i=-50):
    if catalyst in ['Ag', 'Ni']:
        onset = pickle_load(f'files/data/AgNi/{catalyst}/{catalyst}_{voltage}_onset_{onset_i}.pkl')
    else:
        onset = pickle_load(f'files/data/{catalyst}/{catalyst}_{voltage}_onset_{onset_i}.pkl')

    return onset