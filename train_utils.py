import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import shutil

import torch
from torch import nn
from torch.nn.functional import softplus


def train(model, dataloader, optimizer, loss_fn, metric_fn, target_info, device=None):
    model.train()
    num_out = len(target_info['mean'])

    epoch_loss = 0.0
    count = 0.0
    ae_list = [0.0]*num_out
    for it, (x, ys) in enumerate(dataloader):
        x = x.type(torch.float32)
        ys = ys.type(torch.float32)

        if device is not None:
            x = x.to(device)
            ys = ys.to(device)

        pred = model(x)
        if pred.dim() == 3:
            pred = torch.squeeze(pred)
        else:
            pred = model(x)
        pred = [pred[:, i] for i in range(pred.shape[1])]
        ys = torch.squeeze(ys, dim=1)
        ys = [ys[:, i] for i in range(ys.shape[1])]
        loss = loss_fn(pred, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()

        ae = metric_fn(pred, ys)
        for i in range(len(ae_list)):
            ae_list[i] += ae[i]

        count += ys[0].shape[0]

    epoch_loss /= (it + 1)

    std = target_info['std']
    mae_list = []
    for i in range(len(ae_list)):
        mae = (ae_list[i]/count)*std[i]
        mae_list.append(mae)

    return epoch_loss, mae_list


def evaluate(model, dataloader, loss_fn, metric_fn, target_info, device=None):
    model.eval()
    num_out = len(target_info['mean'])

    with torch.no_grad():
        test_loss = 0.0
        count = 0.0
        ae_list = [0.0] * num_out
        for it, (x, ys) in enumerate(dataloader):
            x = x.type(torch.float32)
            ys = ys.type(torch.float32)

            if device is not None:
                x = x.to(device)
                ys = ys.to(device)

            pred = model(x)
            if pred.dim() == 3:
                pred = torch.squeeze(pred)
            else:
                pred = model(x)
            pred = [pred[:, i] for i in range(pred.shape[1])]
            ys = torch.squeeze(ys, dim=1)
            ys = [ys[:, i] for i in range(ys.shape[1])]
            test_loss += loss_fn(pred, ys).detach().item()

            ae = metric_fn(pred, ys)
            for i in range(len(ae_list)):
                ae_list[i] += ae[i]

            count += ys[0].shape[0]

        test_loss /= (it + 1)

        std = target_info['std']
        mae_list = []
        for i in range(len(ae_list)):
            mae = (ae_list[i] / count) * std[i]
            mae_list.append(mae)

    return test_loss, mae_list


class EarlyStopping:
    def __init__(self, patience=200, silent=True):
        self.patience = patience
        self.silent = silent
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if not self.silent:
                print("EarlyStopping counter: {}/{}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        criterion = nn.MSELoss(reduction='mean')
        mse = 0
        for pred, target in zip(pred, target):
            mse += criterion(pred, target)

        return mse


class AELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        error = []
        for pred, target in zip(pred, target):
            abs_error = torch.abs(pred - target)
            error.append(abs_error)
        ae_list = [torch.sum(error[i]) for i in range(len(error))]

        return ae_list


def save_checkpoint(state_dict_objs, ckpt, best_ckpt, best, epoch, is_best):
    misc_objs = {'best': best, 'epoch': epoch}
    objects = copy.copy(misc_objs)
    for k, obj in state_dict_objs.items():
        objects[k] = obj.state_dict()
    torch.save(objects, ckpt)

    if is_best:
        shutil.copy(ckpt, best_ckpt)
        print('best model saved!')


def get_log_string(num_out, header, target_names=None):
    if header:
        log_str = '# Epoch   |   TrainLoss    '
        for i in range(num_out):
            log_str += f'TrainMAE_{target_names[i]}'
            if i < num_out - 1:
                log_str += '    '
        log_str += '   |   ValLoss    '
        for i in range(num_out):
            log_str += f'ValMAE_{target_names[i]}'
            if i < num_out - 1:
                log_str += '    '
        log_str += '   |   Time (sec)'
    else:
        log_str = '{:5d}      {:12.6e}    '
        for i in range(num_out):
            log_str += '{:12.6e}'
            if i < num_out - 1:
                log_str += '    '
        log_str += '      {:12.6e}    '
        for i in range(num_out):
            log_str += '{:12.6e}'
            if i < num_out - 1:
                log_str += '    '
        log_str += '      {:.2f}'

    return log_str