import time
import copy
import shutil
import argparse
import os

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.LSV_dataset import LSVDataset
from train_utils import *
from utils import pickle_load, to_path, create_directory
from model.ResCBAM import ResCBAM


def main_worker(args):
    global best
    best = np.finfo(np.float32).max

    device = 'cuda' if torch.cuda.is_available() else None
    print(f'Use {device} for training')

    train_data = LSVDataset(subset='train', system=args.train_system)
    valid_data = LSVDataset(subset='valid', system=args.train_system)
    test_data = LSVDataset(subset='test', system=args.train_system)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=int(len(valid_data) / 2))
    test_loader = DataLoader(test_data, batch_size=int(len(test_data) / 2))

    num_out = 3 if args.train_system == 'AgNi' else 4
    model = ResCBAM(num_out=num_out)
    if torch.cuda.is_available():
        model.to(device)

    optimizer = Adam(model.parameters(), lr=args.init_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.85, patience=50, verbose=True)
    stopper = EarlyStopping(patience=275)
    state_dict_objs = {'model': model, 'optimizer': optimizer, 'scheduler': scheduler}

    loss_fn = MSELoss()
    metric_fn = AELoss()
    target_info = pickle_load(f'files/data/{args.train_system}/target_info.pkl')

    saved_dir = f'saved/{args.train_system}'
    create_directory(saved_dir, path_is_directory=True)
    ckpt = to_path(os.path.join(saved_dir, 'ResCBAM_checkpoint.pkl'))
    best_ckpt = to_path(os.path.join(saved_dir, 'ResCBAM_best_checkpoint.pkl'))
    epoch = args.epoch

    target_names = ['i_tot', 'FE_CO', 'FE_H2', 'FE_HCOOH']
    log_header = get_log_string(num_out, header=True, target_names=target_names)
    log_string = get_log_string(num_out, header=False)
    print(log_header)

    for e in range(epoch):
        ti = time.time()

        train_loss, train_mae = train(
            model, train_loader, optimizer, loss_fn, metric_fn, target_info, device=device
        )
        valid_loss, valid_mae = evaluate(
            model, valid_loader, loss_fn, metric_fn, target_info, device=device
        )

        if stopper.step(valid_loss):
            break
        scheduler.step(valid_loss)

        is_best = valid_loss < best
        if is_best:
            best = valid_loss

        save_checkpoint(state_dict_objs, ckpt, best_ckpt, best, e, is_best)
        tt = time.time() - ti

        print(log_string.format(e, train_loss, *train_mae, valid_loss, *valid_mae, tt))

    loc = 'cuda:0'
    checkpoint = torch.load(best_ckpt, loc)
    for k, obj in state_dict_objs.items():
        state_dict = checkpoint.pop(k)
        obj.load_state_dict(state_dict)

    _, test_mae = evaluate(
        model, test_loader, loss_fn, metric_fn, target_info, device=device
    )
    for i in range(num_out):
        print('TestMAE_{}: {:12.6e}'.format(target_names[i], test_mae[i]))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_system', type=str, default='AgNi', choices=['AgNi', 'Zn'])
    parser.add_argument('--batch_size', type=int, default=256) # 27 for Zn retrain
    parser.add_argument('--init_lr', type=float, default=0.00075) # 0.00025 for Zn retrain
    parser.add_argument('--epoch', type=int, default=3000)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)

    main_worker(args)