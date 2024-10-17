import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import umap
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib as mpl

from data.data_utils import load_LSV, load_target, concatenate_data, load_onset, zero_padding
from utils import pickle_load, pickle_dump, to_path
from analysis.plot_utils import *


class PlotUMAP:
    def __init__(self, load=False):
        self.load = load
        N = 256
        c1 = np.array([250/N, 82/N, 112/N]) # pink
        c2 = np.array([251/N, 215/N, 30/N]) # yellow
        c3 = np.array([231/N, 117/N, 63/N]) # orange
        c4 = np.array([67 / N, 215 / N, 190 / N])  # green
        c5 = np.array([0 / N, 83 / N, 215 / N])  # blue
        c6 = np.array([174/N, 112/N, 255/N]) # purple
        self.custom_cmap_op = ListedColormap(np.vstack((c1,c2,c3,c4,c5,c6)))
        norm_op = mpl.colors.Normalize(vmin=0, vmax=5)
        self.cmapping_op = cm.ScalarMappable(norm=norm_op, cmap=self.custom_cmap_op)

        colors = ['#28308C', '#3A53A4', '#3ABEE9', '#76C7A4', '#AED03D', '#F9E001', '#F16121', '#821519']
        nodes = [0.0, 0.22, 0.36, 0.44, 0.55, 0.66, 0.8, 1.0]
        self.custom_cmap_fe = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))
        norm_fe = mpl.colors.Normalize(vmin=0.0, vmax=1.0)
        self.cmapping_fe = cm.ScalarMappable(norm=norm_fe, cmap=self.custom_cmap_fe)
        
        lsv_list = []
        target_list = []
        class_list = []
        catalyst = ['Ni', 'Ag']
        voltage = [-3.4, -3.2, -3.0]
        i = 0
        for cat in catalyst:
            for v in voltage:
                CO2_lsv = load_LSV(cat, v)[0]
                CO2_lsv = zero_padding(CO2_lsv)
                target = load_target(cat, v)
                lsv_list.append(CO2_lsv)
                target_list.append(target)
                class_list.append([i]*len(CO2_lsv.T))
                i += 1
        self.CO2_lsv = concatenate_data(lsv_list, axis=1)
        self.target = concatenate_data(target_list)
        self.operating_condition = concatenate_data(class_list)

    def plot_umap(self):
        if self.load:
            mapper = pickle_load('saved/AgNi/UMAP_mapper.pkl')
        else:            
            mapper = umap.UMAP(metric='canberra', n_neighbors=30, set_op_mix_ratio=0.4, negative_sample_rate=5, min_dist=2.99, spread=3,  random_state=42, verbose=True).fit_transform(self.CO2_lsv.T, y=self.target[:,0])
            pickle_dump(mapper, 'saved/AgNi/UMAP_mapper.pkl')

        figure, ax = plt.subplots(figsize=(16, 7))
        plt.subplot(1,2,1)
        plt.scatter(mapper[:,0], mapper[:,1], c=self.target[:,1], s=4, cmap=self.custom_cmap_fe)
        cbar_fe = figure.colorbar(self.cmapping_fe, ax=plt.gca())
        plt.gca().set_title('Faradaic efficiency')

        plt.subplot(1,2,2)
        plt.scatter(mapper[:,0], mapper[:,1], c=self.operating_condition, s=4, cmap=self.custom_cmap_op)
        cbar_op = figure.colorbar(self.cmapping_op, ax=plt.gca())
        cbar_op.set_ticks([(5/6)*i+(5/12) for i in range(6)])
        cbar_op.set_ticklabels([
            'Ni-N/C -3.4 V', 'Ni-N/C -3.2 V', 'Ni-N/C -3.0 V', 'Ag -3.4 V', 'Ag -3.2 V', 'Ag -3.0 V'
        ])
        plt.gca().set_title('Operating condition')
        ax.axis('off')

        plt.savefig(to_path('files/results/AgNi/figs/LSV_UMAP.png'))

        # plt.show()
        plt.close()


class PlotTafel:
    def __init__(self, start_V=None):
        self.v = pickle_load('files/data/V_34.pkl')
        self.std_red_idx = find_nearest_idx(self.v, -1.34) # standard reduction voltage = -1.34
        self.start_V = start_V

        # len_data: LSV from std_red_idx to corresponding voltage idx (padding removed)
        self.format = {-3.0: (665, 4), -3.2: (745, 5), -3.4: (825, 6)}

    def plot_tafel(self, catalyst, voltage, rand_seed=72):
        if catalyst == 'Ag' and voltage == -3.2:
            first_trial = 310 # degradation started after trial 310
        elif catalyst == 'Ag' and voltage == -3.4:
            first_trial = 323 # degradation started after trial 323
        else:
            first_trial = 0

        len_data, plot_h = self.format[voltage]

        LSV = load_LSV(catalyst, voltage)[0].T[first_trial:, self.std_red_idx:]
        J = LSV/10 # current density
        FE_CO = load_target(catalyst, voltage)[first_trial:, 1]

        if catalyst == 'Ni' and voltage == -3.0:
            J = np.delete(J, Ni30_exclude, 0)
            FE_CO = np.delete(FE_CO, Ni30_exclude, 0)

        v = self.v[self.std_red_idx:self.std_red_idx+len_data]
        if self.start_V is not None:
            start_V_idx = find_nearest_idx(v, self.start_V)
        else:
            start_V_idx = 1

        sort_idx = np.argsort(FE_CO)
        J = J[sort_idx]
        FE_CO = FE_CO[sort_idx]
        FE_CO = np.squeeze(mm_scaling(np.expand_dims(FE_CO, axis=1), use_scaler=True))
        tafel = np.log10(np.abs(J))[:, start_V_idx-1:]

        if voltage == -3.0:
            rand_seed = 40
        r1, r2, r3, r4 = get_range_rand_idx(FE_CO, num_rand=30, rand_seed=rand_seed)

        figname = f'{catalyst}_{voltage}_tafel'
        figure, ax = plt.subplots(figsize=(3.8, plot_h))
        for r in [r1, r2, r3, r4]:
            for i in r:
                color = plt.cm.bwr(FE_CO[[i]])
                plt.plot(tafel[i], v[start_V_idx-1:], color=color, linewidth=1.5)
        plt.ylim(min(v[start_V_idx-1:]), max(v[start_V_idx-1:]))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('Log I (mA/cm2)')
        plt.ylabel('Potential (V)')
        plt.title(figname)

        if catalyst in ['Ag', 'Ni']:
            plt.savefig(to_path(f'files/results/AgNi/figs/{figname}.png'))
        else:
            plt.savefig(to_path(f'files/results/{catalyst}/figs/{figname}.png'))

        # plt.show()
        plt.close()


class PlotdLSV:
    def __init__(self, start_V=None):
        self.v = pickle_load('files/data/V_34.pkl')
        self.std_red_idx = find_nearest_idx(self.v, -1.34) # standard reduction voltage = -1.34
        self.start_V = start_V

        # len_data: LSV from std_red_idx to corresponding voltage idx (padding removed)
        self.format = {-3.0: (665, 4), -3.2: (745, 5), -3.4: (825, 6)}

    def plot_dlsv(self, catalyst, voltage, smooth_box=20, rand_seed=72):
        if catalyst == 'Ag' and voltage == -3.2:
            first_trial = 310 # degradation started after trial 310
        elif catalyst == 'Ag' and voltage == -3.4:
            first_trial = 323 # degradation started after trial 323
        else:
            first_trial = 0

        len_data, plot_h = self.format[voltage]

        LSV = load_LSV(catalyst, voltage)[0][self.std_red_idx:, first_trial:]
        FE_CO = load_target(catalyst, voltage)[first_trial:, 1]

        if catalyst == 'Ni' and voltage == -3.0:
            LSV = np.delete(LSV, Ni30_exclude, 1)
            FE_CO = np.delete(FE_CO, Ni30_exclude, 0)

        v = self.v[self.std_red_idx:self.std_red_idx+len_data]
        dLSV = np.array([np.diff(LSV[:, i])/np.diff(v) for i in range(LSV.shape[1])])
        if self.start_V is not None:
            start_V_idx = find_nearest_idx(v, self.start_V)
        else:
            start_V_idx = 1

        sort_idx = np.argsort(FE_CO)
        dLSV = dLSV[sort_idx]
        FE_CO = FE_CO[sort_idx]
        FE_CO = np.squeeze(mm_scaling(np.expand_dims(FE_CO, axis=1), use_scaler=True))

        if voltage == -3.0:
            rand_seed = 40
        r1, r2, r3, r4 = get_range_rand_idx(FE_CO, num_rand=30, rand_seed=rand_seed)

        v = np.array([(v[i]+v[i+1])/2 for i in range(len(v)-1)])
        v = v[start_V_idx-1:]

        figname = f'{catalyst}_{voltage}_dLSV'
        figure, ax = plt.subplots(figsize=(3.8, plot_h))
        for r in [r1, r2, r3, r4]:
            for i in r:
                color = plt.cm.bwr(FE_CO[[i]])
                smooth_dLSV = smooth(dLSV[i], smooth_box)[start_V_idx-1:]
                ax.scatter(smooth_dLSV, v, s=4, color=color, edgecolors='none')
        plt.ylim(min(v), max(v))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('dI/dV')
        plt.ylabel('Potential (V)')
        plt.title(figname)

        if catalyst in ['Ag', 'Ni']:
            plt.savefig(to_path(f'files/results/AgNi/figs/{figname}.png'))
        else:
            plt.savefig(to_path(f'files/results/{catalyst}/figs/{figname}.png'))

        # plt.show()
        plt.close()


class PlotAttentionMap:
    def __init__(self, model_name, load=True, start_V=None):
        self.model_name = model_name
        self.load = load

        self.v = pickle_load('files/data/V_34.pkl')
        self.std_red_idx = find_nearest_idx(self.v, -1.34) # standard reduction voltage = -1.34
        if start_V is None:
            self.start_V_idx = self.std_red_idx
        else:
            self.start_V_idx = find_nearest_idx(self.v, start_V)
        self.targets = ['i_tot', 'FE_CO', 'FE_H2', 'FE_HCOOH']
        self.format = {-3.0: (665, 4), -3.2: (745, 5), -3.4: (825, 6)}

        colors = [
            '#000083', '#0000FE', '#0063FF', '#00CDFF', '#4BFFAB', '#AAFF4C', '#FFE500', '#FF7D00', '#FF1D00', '#880000'
        ]
        nodes = [0.0, 0.17, 0.28, 0.35, 0.44, 0.55, 0.66, 0.77, 0.88, 1.0]
        self.custom_cmap = LinearSegmentedColormap.from_list('mycmap', list(zip(nodes, colors)))

    @staticmethod
    def get_meshgrid(x, y, z):
        X, Y = np.meshgrid(x, y)
        Z = z[Y, np.where(x == X[0])]

        return X, Y, Z

    @staticmethod
    def format_func(value, tick_number):
        return int(value*100)

    def plot_attention_map(self, catalyst, voltage, target_node=1, extrap=False, gradcam=None):
        if catalyst == 'Ag' and voltage == -3.2:
            first_trial = 310 # degradation started after trial 310
        elif catalyst == 'Ag' and voltage == -3.4:
            first_trial = 323 # degradation started after trial 323
        else:
            first_trial = 0

        len_data, plot_h = self.format[voltage]

        if target_node == 0:
            target = self.targets[target_node].split('_')[0]
        else:
            target = self.targets[target_node].split('_')[1]

        if catalyst in ['Ag', 'Ni']:
            results_dir = 'files/results/AgNi'
        else:
            results_dir = 'files/results/Zn'

        if self.load:
            if extrap:
                gradcam = pickle_load(
                    os.path.join(results_dir, f'{self.model_name}_{catalyst}_{voltage}_gradcam_{target}_extrap.pkl')
                )
            else:
                gradcam = pickle_load(
                    os.path.join(results_dir, f'{self.model_name}_{catalyst}_{voltage}_gradcam_{target}.pkl')
                )
        else:
            assert gradcam is not None

        gradcam = gradcam[first_trial:, self.start_V_idx:self.std_red_idx + len_data]
        FE_CO = load_target(catalyst, voltage)[first_trial:, 1]
        if extrap:
            FE_CO = FE_CO[:9]
        onset_30 = load_onset(catalyst, voltage, onset_i=-30)[first_trial:]
        onset_50 = load_onset(catalyst, voltage, onset_i=-50)[first_trial:]

        if catalyst == 'Ni' and voltage == -3.0:
            gradcam = np.delete(gradcam, Ni30_exclude, 0)
            FE_CO = np.delete(FE_CO, Ni30_exclude, 0)
            onset_30 = np.delete(onset_30, Ni30_exclude, 0)
            onset_50 = np.delete(onset_50, Ni30_exclude, 0)

        sort_idx = np.argsort(FE_CO)
        gradcam = gradcam[sort_idx]
        gradcam = mm_scaling(np.abs(gradcam.T))
        FE_CO = FE_CO[sort_idx]
        onset_30 = onset_30[sort_idx]
        onset_50 = onset_50[sort_idx]

        y = np.arange(self.std_red_idx+len_data-self.start_V_idx)
        x = FE_CO
        X, Y, Z = self.get_meshgrid(x, y, gradcam)

        v = self.v[self.start_V_idx:self.std_red_idx+len_data]
        V = v[Y]

        mark = np.linspace(FE_CO[0], FE_CO[-1], 55, endpoint=True)
        mark_idx = [find_nearest_idx(FE_CO, mark[i]) for i in range(mark.shape[0])]
        mark_FE_CO_30, mark_onset_30 = [], []
        mark_FE_CO_50, mark_onset_50 = [], []
        for idx in mark_idx:
            if onset_30[idx] <= -1.8:
                mark_FE_CO_30.append(FE_CO[idx])
                mark_onset_30.append(onset_30[idx])
            if onset_50[idx] <= -1.8:
                mark_FE_CO_50.append(FE_CO[idx])
                mark_onset_50.append(onset_50[idx])

        figname = f'{catalyst}_{voltage}_gradcam_{target}'
        if extrap:
            figname += '_extrap'
        figure, ax = plt.subplots(figsize=(10, plot_h))
        cntr = ax.contourf(X, V, Z, levels=100, cmap=self.custom_cmap)
        ax.plot(mark_FE_CO_30, mark_onset_30, '--', color='white', linewidth=2)
        ax.plot(mark_FE_CO_50, mark_onset_50, '--', color='white', linewidth=3)
        plt.colorbar(cntr)
        plt.xlabel(f'{self.targets[target_node]} (%)')
        plt.ylabel('Potential (V)')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(self.format_func))
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(figname)

        plt.savefig(to_path(os.path.join(results_dir, f'figs/{self.model_name}_{figname}.png')))

        # plt.show()
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--plot_type', type=str, default='attn_map', choices=['umap', 'tafel', 'dlsv', 'attn_map'])
    parser.add_argument('--catalyst', type=str, default='Ag', choices=['Ag', 'Ni', 'Zn'])
    parser.add_argument('--voltage', type=float, default=-3.2, choices=[-3.0, -3.2, -3.4])
    parser.add_argument('--model_name', type=str, default='ResCBAM')
    parser.add_argument('--extrap', default=False, dest='extrap', action='store_true')

    args = parser.parse_args()

    return args

        
if __name__ == '__main__':
    args = parse_args()

    if args.plot_type == 'umap':
        plotter = PlotUMAP(load=True)
        plotter.plot_umap()
    elif args.plot_type == 'tafel':
        plotter = PlotTafel(start_V=-1.8)
        plotter.plot_tafel(args.catalyst, args.voltage)
    elif args.plot_type == 'dlsv':
        plotter = PlotdLSV(start_V=-1.8)
        plotter.plot_dlsv(args.catalyst, args.voltage)
    elif args.plot_type == 'attn_map':
        plotter = PlotAttentionMap(model_name=args.model_name, start_V=-1.8)
        plotter.plot_attention_map(args.catalyst, args.voltage, target_node=1, extrap=args.extrap)
    else:
        print(f'There is no plot_type names: {args.plot_type}')