import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from matplotlib import gridspec
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import argparse

from train_utils import *
from utils import pickle_load, to_path, pickle_dump, create_directory
from data.LSV_dataset import LSVDataset
from data.feats_dataset import FeatsDataset
from model.ResCBAM import ResCBAM
from model.VGG16 import VGG16


def model_load(ckpt_filename, model_name, train_system='AgNi', device=None):
    if device is None:
        device = torch.device('cpu')
    else:
        device = 'cuda:0'

    num_out = 3 if train_system == 'AgNi' else 4
    if 'ResCBAM' in model_name:
        model = ResCBAM(num_out=num_out)
    else: # model_name == 'VGG16'
        model = VGG16(num_out=num_out)
    state_dict_objs = {'model': model}

    model_dir = to_path(f'saved/{train_system}/{ckpt_filename}')
    checkpoints = torch.load(model_dir, map_location=device)
    for k, obj in state_dict_objs.items():
        state_dict = checkpoints.pop(k)
        obj.load_state_dict(state_dict)

    if device != torch.device('cpu'):
        model.cuda(device)

    return model


class Performance:
    def __init__(self, ckpt_filename, train_system='AgNi'):
        self.device = 'cuda' if torch.cuda.is_available() else None
        self.train_system = train_system
        self.figs_dir = f'files/results/{train_system}/figs'
        create_directory(self.figs_dir, path_is_directory=True)

        if ckpt_filename.endswith('.sav'):
            self.model_name = ckpt_filename.replace('.sav', '')
            self.model = joblib.load(to_path(f'saved/{train_system}/{ckpt_filename}'))
        else:
            self.model_name = ckpt_filename.replace('_best_checkpoint.pkl', '')
            self.model = model_load(ckpt_filename, self.model_name, train_system, self.device)

        if train_system == 'AgNi':
            self.target_type = ['i_tot', 'FE_CO', 'FE_H2']
        else:
            self.target_type = ['i_tot', 'FE_CO', 'FE_H2', 'FE_HCOOH']
        self.target_info = pickle_load(f'files/data/{train_system}/target_info.pkl')

    def predict(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            for it, (x, ys) in enumerate(test_loader):
                x = x.type(torch.float32)
                if self.device is not None:
                    x = x.to(self.device)

                pred = self.model(x)
                if pred.dim() == 3:
                    pred = torch.squeeze(pred).cpu().numpy()
                else:
                    pred = self.model(x).cpu().numpy()
                ys = torch.squeeze(ys, dim=1).cpu().numpy()
        pred_list = np.array([pred[:, i] for i in range(pred.shape[1])])
        actual_list = np.array([ys[:, i] for i in range(ys.shape[1])])

        return pred_list, actual_list

    def plot_parity(self, predictions_dict, actual_dict, catalyst=None, voltage=None):
        if self.train_system == 'AgNi':
            max_val_list = [3.8, 100, 100]
            text_x = [0.25, 5, 5]
            text_y = [3.1, 82, 82]
            colors = ['blue', 'red', 'green']
            fig = plt.figure(figsize=(15, 5))
            spec = gridspec.GridSpec(ncols=3, nrows=1)
        else:
            max_val_list = [1.0, 100, 100, 100]
            text_x = [0.05, 5, 5, 5]
            text_y = [0.82, 82, 82, 82]
            colors = ['blue', 'red', 'green', 'darkorange']
            fig = plt.figure(figsize=(20, 5))
            spec = gridspec.GridSpec(ncols=4, nrows=1)

        for i, t in enumerate(self.target_type):
            t_pred, t_actual = predictions_dict[t], actual_dict[t]
            t_mae = mean_absolute_error(t_actual, t_pred)
            t_mse = mean_squared_error(t_actual, t_pred)
            val_range = [0.0, max_val_list[i]]

            ax = plt.subplot(spec[i])
            ax.set_xlabel('Observed', fontsize=15)
            ax.set_ylabel('Predicted', fontsize=15)
            ax.scatter(x=t_actual, y=t_pred, s=60, color=colors[i], alpha=0.5)
            ax.plot(val_range, val_range, color='black', linewidth=1.1)
            ax.set_xlim(val_range)
            ax.set_ylim(val_range)
            ax.text(
                text_x[i], text_y[i], 'MAE: {:.3f} \nMSE: {:.4f}'.format(t_mae, t_mse),
                fontsize=20, bbox={'facecolor': 'w', 'edgecolor': 'black'}
            )
            ax.set_title(f'{t}')

        if catalyst is not None and voltage is not None:
            plt.savefig(to_path(os.path.join(self.figs_dir, f'{self.model_name}_{catalyst}_{voltage}_parity.png')))
        else:
            plt.savefig(to_path(os.path.join(self.figs_dir, f'{self.model_name}_parity.png')))

        # plt.show()
        plt.close()

    def performance(self, extrap=False, parity_plot=True):
        if extrap:
            assert self.train_system == 'AgNi'

        if self.model_name in ['ELN', 'GPR']:
            if extrap:
                x, actual = FeatsDataset(subset='extrap', system='Zn')
            else:
                x, actual = FeatsDataset(subset='test', system=self.train_system)
            actual_list = actual.T
            pred_list = self.model.predict(x).T
        else:
            if extrap:
                data = LSVDataset(subset='extrap', system='Zn')
            else:
                data = LSVDataset(subset='test', system=self.train_system)
            dataloader = DataLoader(data, batch_size=len(data))

            pred_list, actual_list = self.predict(dataloader)

        std = np.expand_dims(self.target_info['std'], axis=-1)
        mean = np.expand_dims(self.target_info['mean'], axis=-1)
        predictions = pred_list * std + mean  # reverse scaling
        actuals = actual_list * std + mean  # reverse scaling

        predictions_dict, actual_dict = {}, {}
        for i, t in enumerate(self.target_type):
            t_pred, t_actual = predictions[i], actuals[i]
            if t != 'i_tot':
                t_pred *= 100
                t_actual *= 100
            predictions_dict[t] = t_pred
            actual_dict[t] = t_actual

            t_mae = mean_absolute_error(t_actual, t_pred)
            t_mse = mean_squared_error(t_actual, t_pred)
            print(f'{t} MAE: {t_mae}  |  MSE: {t_mse}')

        if parity_plot:
            if not extrap: # not recommended due to small number of data
                self.plot_parity(predictions_dict, actual_dict)

    def performance_condition(self, catalyst, voltage, parity_plot=True):
        data = LSVDataset(subset='test', system=self.train_system, catalyst=catalyst, voltage=voltage)
        dataloader = DataLoader(data, batch_size=len(data))

        pred_list, actual_list = self.predict(dataloader)

        std = np.expand_dims(self.target_info['std'], axis=-1)
        mean = np.expand_dims(self.target_info['mean'], axis=-1)
        predictions = pred_list*std+mean # reverse scaling
        actuals = actual_list*std+mean # reverse scaling

        predictions_dict, actual_dict = {}, {}
        print(f'{catalyst}_{voltage} ---------------')
        for i, t in enumerate(self.target_type):
            if t == 'i_tot':
                t_pred, t_actual = predictions[i], actuals[i]
            else:
                t_pred, t_actual = predictions[i] * 100, actuals[i] * 100
            predictions_dict[t] = t_pred
            actual_dict[t] = t_actual

            t_mae = mean_absolute_error(t_actual, t_pred)
            t_mse = mean_squared_error(t_actual, t_pred)
            print(f'{t} MAE: {t_mae}  |  MSE: {t_mse}')

        if parity_plot:
            self.plot_parity(predictions_dict, actual_dict, catalyst, voltage)


class Performance_ensemble:
    def __init__(self, train_system='AgNi'):
        self.device = 'cuda' if torch.cuda.is_available() else None
        self.train_system = train_system
        self.figs_dir = f'files/results/{train_system}/figs'
        create_directory(self.figs_dir, path_is_directory=True)

        flist = os.listdir(to_path(f'saved/{train_system}/ensemble'))
        ckpt_filenames = [f'ensemble/{f}' for f in flist if f.endswith('best_checkpoint.pkl')]
        self.model_name = f'{flist[0].split("_")[0]}_seeds'
        self.models = []
        for ckpt in ckpt_filenames:
            model = model_load(ckpt, self.model_name, train_system, self.device)
            self.models.append(model)

        if train_system == 'AgNi':
            self.target_type = ['i_tot', 'FE_CO', 'FE_H2']
        else:
            self.target_type = ['i_tot', 'FE_CO', 'FE_H2', 'FE_HCOOH']
        self.target_info = pickle_load(f'files/data/{train_system}/target_info.pkl')

    def predict(self, model, test_loader):
        model.eval()
        with torch.no_grad():
            for it, (x, ys) in enumerate(test_loader):
                x = x.type(torch.float32)
                if self.device is not None:
                    x = x.to(self.device)

                pred = model(x)
                if pred.dim() == 3:
                    pred = torch.squeeze(pred).cpu().numpy()
                else:
                    pred = model(x).cpu().numpy()
                ys = torch.squeeze(ys, dim=1).cpu().numpy()
        pred_list = np.array([pred[:, i] for i in range(pred.shape[1])])
        actual_list = np.array([ys[:, i] for i in range(ys.shape[1])])

        return pred_list, actual_list

    def plot_parity(self, predictions_dict, actual_dict, catalyst=None, voltage=None):
        if self.train_system == 'AgNi':
            max_val_list = [3.8, 100, 100]
            text_x = [0.25, 5, 5]
            text_y = [3.1, 82, 82]
            colors = ['blue', 'red', 'green']
            fig = plt.figure(figsize=(15, 5))
            spec = gridspec.GridSpec(ncols=3, nrows=1)
        else:
            max_val_list = [1.0, 100, 100, 100]
            text_x = [0.05, 5, 5, 5]
            text_y = [0.82, 82, 82, 82]
            colors = ['blue', 'red', 'green', 'darkorange']
            fig = plt.figure(figsize=(20, 5))
            spec = gridspec.GridSpec(ncols=4, nrows=1)

        for i, t in enumerate(self.target_type):
            t_pred, t_actual = predictions_dict[t], actual_dict[t]
            t_mae = mean_absolute_error(t_actual, t_pred)
            t_mse = mean_squared_error(t_actual, t_pred)
            val_range = [0.0, max_val_list[i]]

            ax = plt.subplot(spec[i])
            ax.set_xlabel('Observed', fontsize=15)
            ax.set_ylabel('Predicted', fontsize=15)
            ax.scatter(x=t_actual, y=t_pred, s=60, color=colors[i], alpha=0.5)
            ax.plot(val_range, val_range, color='black', linewidth=1.1)
            ax.set_xlim(val_range)
            ax.set_ylim(val_range)
            ax.text(
                text_x[i], text_y[i], 'MAE: {:.3f} \nMSE: {:.4f}'.format(t_mae, t_mse),
                fontsize=20, bbox={'facecolor': 'w', 'edgecolor': 'black'}
            )
            ax.set_title(f'{t}')

        if catalyst is not None and voltage is not None:
            plt.savefig(to_path(os.path.join(self.figs_dir, f'{self.model_name}_{catalyst}_{voltage}_parity.png')))
        else:
            plt.savefig(to_path(os.path.join(self.figs_dir, f'{self.model_name}_parity.png')))

        # plt.show()
        plt.close()

    def performance(self, extrap=False, parity_plot=True):
        if extrap:
            assert self.train_system == 'AgNi'
            data = LSVDataset(subset='extrap', system='Zn')
        else:
            data = LSVDataset(subset='test', system=self.train_system)
        dataloader = DataLoader(data, batch_size=len(data))

        num_out = 3 if self.train_system == 'AgNi' else 4
        pred_list = np.zeros((num_out, len(data)))
        for model in self.models:
            pred_, actual_list = self.predict(model, dataloader)
            pred_list += pred_
        pred_list /= len(self.models)

        std = np.expand_dims(self.target_info['std'], axis=-1)
        mean = np.expand_dims(self.target_info['mean'], axis=-1)
        predictions = pred_list * std + mean  # reverse scaling
        actuals = actual_list * std + mean  # reverse scaling

        predictions_dict, actual_dict = {}, {}
        for i, t in enumerate(self.target_type):
            t_pred, t_actual = predictions[i], actuals[i]
            if t != 'i_tot':
                t_pred *= 100
                t_actual *= 100
            predictions_dict[t] = t_pred
            actual_dict[t] = t_actual

            t_mae = mean_absolute_error(t_actual, t_pred)
            t_mse = mean_squared_error(t_actual, t_pred)
            print(f'{t} MAE: {t_mae}  |  MSE: {t_mse}')

        if parity_plot:
            if not extrap: # not recommended due to small number of data
                self.plot_parity(predictions_dict, actual_dict)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_filename', type=str, default='ResCBAM_best_checkpoint.pkl')
    parser.add_argument('--train_system', type=str, default='AgNi')
    parser.add_argument('--ensemble', default=False, dest='ensemble', action='store_true')
    parser.add_argument('--extrap', default=False, dest='extrap', action='store_true')

    args = parser.parse_args()

    return args



if __name__ == '__main__':
    args = parse_args()

    if not args.ensemble:
        perf = Performance(args.ckpt_filename, args.train_system)
    else:
        perf = Performance_ensemble(args.train_system)
    perf.performance(extrap=args.extrap)