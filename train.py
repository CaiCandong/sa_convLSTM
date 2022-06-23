import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Subset

from skimage.metrics import structural_similarity as ssim
from data_loader.MovingMNIST import MovingMNIST
from model.Encode2Decode import Encode2Decode
from torch import optim
from pathlib import Path

from utils import save_images


def split_train_val(dataset):
    idx = [i for i in range(len(dataset))]

    random.seed(1234)
    random.shuffle(idx)

    num_train = int(0.8 * len(idx))

    train_idx = idx[:num_train]
    val_idx = idx[num_train:]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    print(f'train index: {len(train_idx)}')
    print(f'val index: {len(val_idx)}')

    return train_dataset, val_dataset


def train(model, train_dataloader, valid_dataloader, criterion, optimizer, device="cpu", epochs=10,
          save_dir=Path("outputs")):
    n_total_steps = len(train_dataloader)
    train_losses = []

    avg_train_losses = []
    avg_valid_losses = []
    for i in range(epochs):
        losses, val_losses = [], []
        model.train()
        epoch_loss = 0.0
        val_epoch_loss = 0.0

        for ite, (x, y) in enumerate(train_dataloader):
            x_train, y_train = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred_train = model(x_train, y_train)
            loss = criterion(pred_train, y_train)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            print(
                f'epoch {i + 1} / {epochs}, step {ite + 1}/{n_total_steps}, loss = {loss.item():.4f}')

            epoch_loss += loss.item()
            if ite and ite%200==0:
                save_images(y_train,pred_train,"outputs",i,ite)

        with torch.no_grad():
            model.eval()
            for _, (x, y) in enumerate(valid_dataloader):
                x_val, y_val = x.to(device), y.to(device)
                pred_val = model(x_val, y_val, teacher_forcing_rate=0)
                loss = criterion(pred_val, y_val)
                val_losses.append(loss.item())
                val_epoch_loss += loss.item()
        train_loss = np.average(train_losses)
        valid_loss = np.average(val_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        torch.save(model.state_dict(), save_dir / f"checkpoint_epoch_{i}.pt")
        print('{}th epochs train loss {}, valid loss {}'.format(i, np.mean(train_loss), np.mean(valid_loss)))

        plt.plot(avg_train_losses, '-o')
        plt.plot(avg_valid_losses, '-o')
        plt.xlabel('epoch')
        plt.ylabel('losses')
        plt.legend(['Train', 'Validation'])
        plt.title('(MSE) Avg Train vs Validation Losses')
        plt.savefig(save_dir / f"train_val_loss_curve_epoch_{i}.png")
        plt.clf()


def tset(model, test_dataloader, criterion, ckp_path, device):
    model.load_state_dict(torch.load(ckp_path, map_location=device))
    test_pred, test_gt = [], []
    with torch.no_grad():
        model.eval()
        for i, (x, y) in enumerate(test_dataloader):
            x, y = x.to(device), y.to(device)
            pred = model(x, y, teacher_forcing_rate=0)
            test_pred.append(pred.cpu().data.numpy())
            test_gt.append(y.cpu().data.numpy())
    test_pred = np.concatenate(test_pred)
    test_gt = np.concatenate(test_gt)
    mse = criterion(test_gt, test_pred)
    print('TEST Data loader - MSE = {:.6f}'.format(mse))

    # Frame-wise comparison in MSE and SSIM

    overall_mse = 0
    overall_ssim = 0
    frame_mse = np.zeros(test_gt.shape[1])
    frame_ssim = np.zeros(test_gt.shape[1])

    for i in range(test_gt.shape[1]):
        for j in range(test_gt.shape[0]):
            mse_ = np.square(test_gt[j, i] - test_pred[j, i]).sum()
            test_gt_img = np.squeeze(test_gt[j, i])
            test_pred_img = np.squeeze(test_pred[j, i])
            ssim_ = ssim(test_gt_img, test_pred_img)

            overall_mse += mse_
            overall_ssim += ssim_
            frame_mse[i] += mse_
            frame_ssim[i] += ssim_

    overall_mse /= 10
    overall_ssim /= 10
    frame_mse /= 1000
    frame_ssim /= 1000
    print(f'overall_mse.shape {overall_mse}')
    print(f'overall_ssim.shape {overall_ssim}')
    print(f'frame_mse.shape {frame_mse}')
    print(f'frame_ssim.shape {frame_ssim}')

    path_pred = './results/npy_file_save/saconvlstm_test_pred_speedpt5.npy'
    path_gt = './results/npy_file_save/saconvlstm_test_gt_speedpt5.npy'

    np.save(path_pred, test_pred)
    np.save(path_gt, test_gt)


def get_config():
    # TODO: get config from yaml file
    config = {
        "epoch": 1,
        'input_dim': 1,
        'batch_size': 8,
        'padding': 1,
        'lr': 0.001,
        'device': "cuda:0" if torch.cuda.is_available() else "cup",
        'attn_hidden_dim': 64,
        'kernel_size': (3, 3),
        'img_size': (16, 16),
        'hidden_dim': 64,
        'num_layers': 4,
        'output_dim': 10,
        'input_window_size': 10,
        'loss': "L2",
        'model_cell': 'sa_convlstm',
        'bias': True,
        "batch_first": True,
        "root": 'data_loader/.data/mnist'
    }
    return config


def main():
    config = get_config()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = MovingMNIST(config["root"], train=True)
    train_dataset, val_dataset = split_train_val(dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                                   num_workers=0)
    valid_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                                                   num_workers=0)

    # ***************** loss function  *********************#
    criterion = nn.MSELoss()
    # ***************** model ******************************#
    model = Encode2Decode(config['input_dim'],
                          config['hidden_dim'],
                          attn_hidden_dim=config["attn_hidden_dim"],
                          kernel_size=config['kernel_size'],
                          img_size=config["img_size"],
                          num_layers=config['num_layers'],
                          batch_first=config['batch_first'],
                          bias=config['bias']
                          )
    model= model.to(device)
    # ******************  optimizer ***********************#
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    # ****************** start training ************************#
    train(model, train_dataloader, valid_dataloader, criterion, optimizer,device)
    # print(next(iter(train_dataloader))[0].shape)


if __name__ == '__main__':
    main()
