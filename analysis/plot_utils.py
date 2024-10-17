import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler

from utils import pickle_load, to_path
from data.data_utils import find_nearest_idx


Ni30_exclude = [38, 39, 40, 46, 47, 48, 49, 54, 55, 56, 61, 62, 63, 64, 65, 66, 67, 68, 69]


def mm_scaling(x, use_scaler=False):
    if use_scaler:
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x)
    else:
        x_max = np.max(x, axis=0)
        x_scaled = x / x_max

    return x_scaled


def smooth(x, box_size):
    smooth_x = []
    for i in range(x.shape[0]):
        if i < box_size:
            smooth_x.append(np.mean(x[:i+box_size+1]))
        elif i >= x.shape[0]-box_size:
            smooth_x.append(np.mean(x[i-box_size:]))
        else:
            smooth_x.append(np.mean(x[i-box_size:i+box_size+1]))
    smooth_x = np.array(smooth_x)

    return smooth_x


def get_range_rand_idx(sort_x, num_rand=20, rand_seed=32):
    max_x, min_x = sort_x[-1], sort_x[0]
    q_size, h_size = (max_x-min_x)*0.25, (max_x-min_x)*0.5
    mid_x1_i = find_nearest_idx(sort_x, min_x+q_size)
    mid_x2_i = find_nearest_idx(sort_x, max_x-q_size)
    mid_mid_i = find_nearest_idx(sort_x, min_x+h_size)

    r1 = np.arange(0, mid_x1_i)
    r2 = np.arange(mid_x1_i, mid_mid_i)
    r3 = np.arange(mid_mid_i, mid_x2_i)
    r4 = np.arange(mid_x2_i, sort_x.shape[0])

    np.random.seed(rand_seed)
    r1_rand = np.random.choice(r1, num_rand)
    r2_rand = np.random.choice(r2, int(num_rand/2))
    r3_rand = np.random.choice(r3, int(num_rand/2))
    r4_rand = np.random.choice(r4, int(num_rand/2))

    return r1_rand, r2_rand, r3_rand, r4_rand


def get_V_ticks(voltage, len_data, start_V=None):
    V = pickle_load('files/data/V_34.pkl')
    std_red_idx = find_nearest_idx(V, -1.34)

    if start_V is None:
        start_V_idx = 0
        ticks_V = np.arange(voltage, 0.01, 0.2)
    else:
        start_V_idx = find_nearest_idx(V, start_V)-1
        ticks_V = np.arange(voltage, (start_V+0.01), 0.2)
    V = V[start_V_idx:std_red_idx+len_data]

    ticks_V_idx = []
    for v in ticks_V:
        ticks_V_idx.append(find_nearest_idx(V, v))
    ticks_V_ = ['{:.1f}'.format(ticks_V[i]) for i in range(len(ticks_V))]

    return ticks_V_idx, ticks_V_