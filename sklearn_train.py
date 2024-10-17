import os
import joblib
from sklearn.metrics import mean_absolute_error
import argparse

from model.sklearn_regressor import Regressor
from data.feats_dataset import FeatsDataset
from utils import pickle_load, create_directory


def main_worker(name, train_system='AgNi'):
    saved_dir = f'saved/{train_system}'
    create_directory(saved_dir, path_is_directory=True)

    print(f'Start {name} regression ...')
    x_train, y_train = FeatsDataset('train', train_system)
    x_test, y_test = FeatsDataset('test', train_system)

    regressor = Regressor()
    model = regressor(name)
    model.fit(x_train, y_train)

    joblib.dump(model, os.path.join(saved_dir, f'{name}.sav'))

    target_info = pickle_load(f'files/data/{train_system}/target_info.pkl')
    pred = model.predict(x_test)
    mae_list = []
    for i in range(pred.shape[-1]):
        mae = mean_absolute_error(y_test[:, i], pred[:, i])
        mae *= target_info['std'][i]
        mae_list.append(mae)

    print('TestMAE_i_tot: {:12.6e}'.format(mae_list[0]))
    print('TestMAE_FE_CO: {:12.6e}'.format(mae_list[1]))
    print('TestMAE_FE_H2: {:12.6e}'.format(mae_list[2]))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_system', type=str, default='AgNi', choices=['AgNi', 'Zn'])
    parser.add_argument('--regressor', type=str, default='GPR', choices=['ELN', 'GPR'])

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)

    main_worker(name=args.regressor, train_system=args.train_system)