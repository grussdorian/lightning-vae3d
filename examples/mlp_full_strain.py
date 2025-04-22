import os

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as Func
from torchinfo import summary
import numpy as np
import pandas as pd
import wandb
import time


class MLP(nn.Module):
    def __init__(self, latent_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 21)
        self.activation = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':

    torch.manual_seed(42)
    np.random.seed(42)

    # device = torch.device('cuda')
    # device = torch.device('mps')
    device = torch.device('cpu')

    latent_dim = 256
    hidden_dim = 8192
    loss_functions = 'bce_fft_fz'
    batch_size = 128
    n_epochs = 1_000
    learning_rate = 1e-5
    weight_decay = 1e-5
    dropout = 0.0
    n_components = 256
    ckpt_dir = f'./vae3d_datasets/vae_checkpoints/resnet18_latent256_hidden8192/'
    out_dir = './out/mlp_yield_prediction/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    n_train = 40_000

    x = np.load('./vae3d_datasets/mlp_data/uniaxial_load_fz_mg_fingerprint.npz')['state']
    y = np.load('./vae3d_datasets/mlp_data/uniaxial_load_fz_mg_stresses.npy')
    y_min = np.min(y)
    y = y - y_min
    y_max = np.max(y)
    y = y / y_max
    x = torch.from_numpy(x)
    y = torch.from_numpy(y).to(torch.float)

    shuffled_args = np.random.choice(x.shape[0], x.shape[0], replace=False)
    train_args = shuffled_args[:n_train]
    test_args = shuffled_args[n_train:]
    xtrain = x[train_args]
    xtest = x[test_args]
    ytrain = y[train_args]
    ytest = y[test_args]

    x_train_loader = DataLoader(xtrain, batch_size=batch_size)
    x_test_loader = DataLoader(xtest, batch_size=batch_size)
    y_train_loader = DataLoader(ytrain, batch_size=batch_size)
    y_test_loader = DataLoader(ytest, batch_size=batch_size)

    model = MLP(latent_dim=latent_dim, dropout=dropout).to(device)
    summary(model, (1, latent_dim))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, amsgrad=True)

    train_loss_list = []
    test_loss_list = []
    epoch_list = []
    for n in range(n_epochs):

        # Train model
        model.train()
        model.zero_grad(set_to_none=True)
        train_loss = 0.0
        n_batches = len(x_train_loader)
        for _, (x_batch, y_batch) in enumerate(zip(x_train_loader, y_train_loader)):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            Y_hat = model(x_batch)
            loss = nn.MSELoss()(Y_hat, y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= n_batches
        wandb.log({"train_loss": train_loss}, step=n)

        # Evaluate model
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            n_batches = len(x_test_loader)
            for _, (x_batch, y_batch) in enumerate(zip(x_test_loader, y_test_loader)):
                Y_t_hat = model(x_batch.to(device))
                loss_t = Func.mse_loss(Y_t_hat, y_batch.to(device))
                test_loss += loss_t.item()

            test_loss /= n_batches
            wandb.log({"test_loss": test_loss}, step=n)

        if (n+1) % 10 == 0 or n == 0:
            epoch_list += [n+1]
            train_loss_list += [train_loss]
            test_loss_list += [test_loss]
            print(f'epoch {n+1}/{n_epochs}: train loss = {train_loss}, test loss = {test_loss}')

    checkpoint = {'epoch': n_epochs,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'train_loss': train_loss,
                  'test_loss': test_loss,
                  }
    torch.save(checkpoint, f'{out_dir}mlp_checkpoint_epoch{n_epochs}.pt')
    print(f'Test args: {test_args}')

    # Post-processing
    df_train_pred = pd.DataFrame()
    df_test_pred = pd.DataFrame()
    x_train_loader = DataLoader(xtrain, batch_size=len(xtrain))
    x_test_loader = DataLoader(xtest, batch_size=len(xtest))
    y_train_loader = DataLoader(ytrain, batch_size=len(ytrain))
    y_test_loader = DataLoader(ytest, batch_size=len(ytest))
    model.eval()
    with torch.no_grad():
        for _, (x_batch, y_batch) in enumerate(zip(x_train_loader, y_train_loader)):
            y_hat = torch.squeeze(model(x_batch))
            y_batch = torch.squeeze(y_batch)
            print(y_batch.shape)
            print(y_hat.shape)
            np.save(f'{out_dir}train_stresses.npy', (np.copy(y_batch) * y_max + y_min) * 1e-6)
            np.save(f'{out_dir}train_pred_stresses.npy', (y_hat * y_max + y_min) * 1e-6)
        for _, (x_batch, y_batch) in enumerate(zip(x_test_loader, y_test_loader)):
            y_hat = torch.squeeze(model(x_batch))
            y_batch = torch.squeeze(y_batch)
            np.save(f'{out_dir}test_stresses.npy', (np.copy(y_batch) * y_max + y_min) * 1e-6)
            np.save(f'{out_dir}test_pred_stresses.npy', (y_hat * y_max + y_min) * 1e-6)
            print(y_batch.shape)
            print(y_hat.shape)

    df = pd.DataFrame({'epoch': epoch_list,
                       'train_loss': train_loss_list,
                       'test_loss': test_loss_list})
    df.to_csv(f'{out_dir}loss_plots.csv')
