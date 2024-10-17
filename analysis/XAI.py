import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import cv2
import torch
import time
import logging
import warnings
import argparse

from utils import pickle_load, pickle_dump, to_path, create_directory
from data.data_utils import load_target, concatenate_data
from data.scaler import scale_input
from analysis.performance import model_load
from analysis.plot import PlotAttentionMap

warnings.filterwarnings(action='ignore')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class GradCAM:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name

        for name, module in self.model._modules.items():
            if name == self.layer_name:
                self.target_layer = module

        self.target_output = None
        self.target_output_grad = None

        def forward_hook(_, __, output):
            self.target_output = output.clone()

        def backward_hook( _, grad_in, grad_out):
            assert len(grad_out) == 1
            self.target_output_grad = grad_out[0].clone()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def get_target_layer_output(self, x):
        for name, module in self.model._modules.items():
            x = module(x)
            if name == self.layer_name:
                break

        return x

    def forward_pass(self, x):
        self.model.eval()
        self.model.zero_grad()
        return self.get_target_layer_output(x), self.model(x)

    def get_gradcam(self, input, target_node):
        layer_out, model_out = self.forward_pass(input)
        model_out = model_out[:, target_node]
        model_out.backward(retain_graph=True)

        layer_out = layer_out.detach().cpu().numpy()

        grad = self.target_output_grad.detach().cpu().numpy()
        grad = np.maximum(grad, 0)
        weights = np.mean(grad, axis=(0, 2))

        gradcam = np.zeros(layer_out[0].shape[-1], dtype=np.float32)
        for k, w in enumerate(weights):
            gradcam += w*layer_out[0, k, :]

        gradcam = np.maximum(gradcam, 0)
        gradcam = gradcam / np.max(gradcam)
        gradcam = cv2.resize(gradcam, (1, input.shape[-1]))

        return gradcam


class GetGradCAM:
    def __init__(self, ckpt_filename, train_system='AgNi'):
        self.device = 'cuda' if torch.cuda.is_available() else None
        self.train_system = train_system

        self.ref = 'N2' if train_system == 'AgNi' else 'Ar'
        model = model_load(ckpt_filename, 'ResCBAM', train_system, self.device)
        layer_name = 'residual_cbams'

        self.explainer = GradCAM(model, layer_name)
        self.model_name = ckpt_filename.replace('_best_checkpoint.pkl', '')
        self.plotter = PlotAttentionMap(self.model_name, load=False, start_V=-1.8)
        self.targets = ['i_tot', 'FE_CO', 'FE_H2', 'FE_HCOOH']

    def _calculate(self, catalyst, voltage, input, target_node=1, extrap=False):
        logger.info(
            f'{self.model_name} {catalyst} {voltage} GradCAM calculation: {input.shape[0]} inputs ================'
        )

        input = torch.tensor(input).type(torch.float32)
        input = input.to(self.device)

        ti = time.time()
        gradcam_list = []
        for i in range(len(input)):
            try:
                if i % int(len(input)/10) == 0:
                    logger.info(
                        f'{self.model_name} {catalyst} {voltage} GradCAM {i} / {len(input)}: {round(i/len(input)*100)}% input calculated ...'
                    )
            except:
                logger.info(f'{self.model_name} {catalyst} {voltage} GradCAM_extrap: {i} input calculated...')

            gradcam_list.append(
                self.explainer.get_gradcam(torch.unsqueeze(input[i], 0), target_node).T
            )
        gradcam = concatenate_data(gradcam_list, axis=0)
        tt = time.time() - ti
        print('{} {} {} GradCAM total elapsed time: {:.2f}sec'.format(self.model_name, catalyst, voltage, tt))

        if target_node == 0:
            target = self.targets[target_node].split('_')[0]
        else:
            target = self.targets[target_node].split('_')[1]

        if catalyst in ['Ag', 'Ni']:
            results_dir = 'files/results/AgNi'
        else:
            results_dir = 'files/results/Zn'
        create_directory(results_dir)

        if extrap:
            assert target_node == 1 and catalyst == 'Zn'
            fname = f'{self.model_name}_{catalyst}_{voltage}_gradcam_{target}_extrap.pkl'
        else:
            fname = f'{self.model_name}_{catalyst}_{voltage}_gradcam_{target}.pkl'

        pickle_dump(
            gradcam, os.path.join(results_dir, f'{fname}')
        )

        return gradcam

    def __call__(self, catalyst, voltage, target_node=1, extrap=False, plot_attention_map=True):
        if extrap:
            assert catalyst == 'Zn' and voltage == -3.4
            assert self.train_system == 'AgNi'
            sc_input = pickle_load('files/data/Zn/LSV_input_extrap.pkl')
        else:
            CO2_dict = pickle_load(f'files/data/{self.train_system}/CO2.pkl')
            ref_dict = pickle_load(f'files/data/{self.train_system}/{self.ref}.pkl')
            CO2_input, ref_input = CO2_dict[f'{catalyst}_{voltage}'], ref_dict[f'{catalyst}_{voltage}']
            input_info = pickle_load(f'files/data/{self.train_system}/LSV_input_info.pkl')
            sc_input = scale_input(CO2_input, ref_input, input_info['max'])

        gradcam = self._calculate(catalyst, voltage, sc_input, target_node, extrap)

        if plot_attention_map:
            self.plotter.plot_attention_map(catalyst, voltage, target_node, extrap, gradcam)


class GetGradCAM_ensemble:
    def __init__(self, train_system='AgNi'):
        self.device = 'cuda' if torch.cuda.is_available() else None
        self.train_system = train_system

        self.ref = 'N2' if train_system == 'AgNi' else 'Ar'
        flist = os.listdir(to_path(f'saved/{train_system}/ensemble'))
        ckpt_filenames = [f'ensemble/{f}' for f in flist if f.endswith('best_checkpoint.pkl')]
        self.model_name = f'{flist[0].split("_")[0]}_seeds'
        self.models = []
        for ckpt in ckpt_filenames:
            model = model_load(ckpt, self.model_name, train_system, self.device)
            self.models.append(model)
        self.plotter = PlotAttentionMap(self.model_name, load=False, start_V=-1.8)
        self.targets = ['i_tot', 'FE_CO', 'FE_H2', 'FE_HCOOH']

    def _calculate(self, catalyst, voltage, input, target_node=1):
        logger.info(
            f'{self.model_name} {catalyst} {voltage} GradCAM calculation: {input.shape[0]} inputs ================'
        )

        input = torch.tensor(input).type(torch.float32)
        input = input.to(self.device)

        ti = time.time()
        gradcam_list = []
        for i in range(len(input)):
            try:
                if i % int(len(input)/10) == 0:
                    logger.info(
                        f'{self.model_name} {catalyst} {voltage} GradCAM {i} / {len(input)}: {round(i/len(input)*100)}% input calculated ...'
                    )
            except:
                logger.info(f'{self.model_name} {catalyst} {voltage} GradCAM_extrap: {i} input calculated...')

            gradcam_list.append(
                self.explainer.get_gradcam(torch.unsqueeze(input[i], 0), target_node).T
            )
        gradcam = concatenate_data(gradcam_list, axis=0)
        tt = time.time() - ti
        print('{} {} {} GradCAM total elapsed time: {:.2f}sec'.format(self.model_name, catalyst, voltage, tt))

        return gradcam

    def __call__(self, catalyst, voltage, target_node=1, extrap=False, plot_attention_map=True):
        if extrap:
            assert catalyst == 'Zn' and voltage == -3.4
            assert self.train_system == 'AgNi'
            sc_input = pickle_load('files/data/Zn/LSV_input_extrap.pkl')
        else:
            CO2_dict = pickle_load(f'files/data/{self.train_system}/CO2.pkl')
            ref_dict = pickle_load(f'files/data/{self.train_system}/{self.ref}.pkl')
            CO2_input, ref_input = CO2_dict[f'{catalyst}_{voltage}'], ref_dict[f'{catalyst}_{voltage}']
            input_info = pickle_load(f'files/data/{self.train_system}/LSV_input_info.pkl')
            sc_input = scale_input(CO2_input, ref_input, input_info['max'])

        gradcam = np.zeros(sc_input[:, 0, :].shape)
        for model in self.models:
            layer_name = 'residual_cbams'
            self.explainer = GradCAM(model, layer_name)

            gradcam += self._calculate(catalyst, voltage, sc_input, target_node)
        gradcam /= len(self.models)

        if catalyst in ['Ag', 'Ni']:
            results_dir = 'files/results/AgNi'
        else:
            results_dir = 'files/results/Zn'
        create_directory(results_dir)
        target = self.targets[target_node].split('_')[1]
        if extrap:
            assert target_node == 1 and catalyst == 'Zn'
            fname = f'{self.model_name}_{catalyst}_{voltage}_gradcam_{target}_extrap.pkl'
        else:
            fname = f'{self.model_name}_{catalyst}_{voltage}_gradcam_{target}.pkl'
        pickle_dump(gradcam, os.path.join(results_dir, fname))

        if plot_attention_map:
            self.plotter.plot_attention_map(catalyst, voltage, target_node, extrap, gradcam)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_filename', type=str, default='ResCBAM_best_checkpoint.pkl')
    parser.add_argument('--train_system', type=str, default='AgNi')
    parser.add_argument('--ensemble', default=False, dest='ensemble', action='store_true')
    parser.add_argument('--extrap', default=False, dest='extrap', action='store_true')
    parser.add_argument('--catalyst', type=str, default='Ag', choices=['Ag', 'Ni', 'Zn'])
    parser.add_argument('--voltage', type=float, default=-3.2, choices=[-3.0, -3.2, -3.4])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if not args.ensemble:
        get_gradcam = GetGradCAM(args.ckpt_filename, train_system=args.train_system)
    else:
        get_gradcam = GetGradCAM_ensemble(train_system=args.train_system)
    get_gradcam(catalyst=args.catalyst, voltage=args.voltage, target_node=1, extrap=args.extrap)