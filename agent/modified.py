# modified.py
"""
Training script of the model shipped with this repository
(model_final.pth, x_scaler.save, y_scaler.save).

The shipped model is the result of run 3 in logs/target_mse (23,400 training
samples). AIDE's script for that run trained five seeds and reached a validation
MSE of 0.001955; modifier.py turned it into this script, which trains seed 42
and saves the model and both scalers.

Four changes were made by hand so that the script works with the current code:
  1. The training part is under `if __name__ == "__main__":`, so that
     `from modified import forward_model` (inverse_NA.py) does not retrain.
  2. The model and the scalers are saved in this folder. The original wrote to
     an absolute path on the machine of the run.
  3. The device is "cuda". The original named "cuda:2".
  4. The predictions for ./input/test_g.csv (./working/submission.csv) were
     removed: the current code does not put a test set in ./input.
The network, the preprocessing and the training settings are unchanged.

A run of the agent system replaces this file: the inverse stage writes
modified.py as the refactored training script (see modifier.py), together with
a new model_final.pth and new scalers. Running this file does the same: it
retrains on the data in ./input and overwrites the shipped model.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.optim.lr_scheduler import OneCycleLR
import joblib


class ResidualBlock(nn.Module):
    def __init__(self, dim, p_dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(p_dropout)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
    def forward(self, x):
        out = self.act(self.bn1(self.fc1(x)))
        out = self.drop(out)
        out = self.bn2(self.fc2(out))
        out = self.drop(out)
        return self.act(out + x)

class forward_model(nn.Module):
    def __init__(self, in_dim=14, hid_dim=512, out_dim=2001, n_blocks=3, p_dropout=0.1):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hid_dim)
        self.bn_in = nn.BatchNorm1d(hid_dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(p_dropout)
        self.blocks = nn.Sequential(*[ResidualBlock(hid_dim, p_dropout) for _ in range(n_blocks)])
        self.fc_out = nn.Linear(hid_dim, out_dim)
        self.fc_skip = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        h = self.act(self.bn_in(self.fc_in(x)))
        h = self.drop(h)
        h = self.blocks(h)
        return self.fc_out(h) + self.fc_skip(x)


if __name__ == "__main__":
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    g_train = pd.read_csv("./input/g_training.csv").values.astype(np.float32)
    s_train = pd.read_csv("./input/s_training.csv").values.astype(np.float32)
    g_val = pd.read_csv("./input/g_validation.csv").values.astype(np.float32)
    s_val = pd.read_csv("./input/s_validation.csv").values.astype(np.float32)
    s_val_orig = s_val.copy()

    # Standardize
    scaler_x = StandardScaler().fit(g_train)
    scaler_y = StandardScaler().fit(s_train)
    g_train_s = scaler_x.transform(g_train)
    g_val_s = scaler_x.transform(g_val)
    s_train_s = scaler_y.transform(s_train)
    s_val_s = scaler_y.transform(s_val)

    # Ensure matching lengths
    n_train = min(g_train_s.shape[0], s_train_s.shape[0])
    g_train_s = g_train_s[:n_train]
    s_train_s = s_train_s[:n_train]
    n_val = min(g_val_s.shape[0], s_val_s.shape[0])
    g_val_s = g_val_s[:n_val]
    s_val_s = s_val_s[:n_val]
    s_val_orig = s_val_orig[:n_val]

    # DataLoaders
    batch_size = 64
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(g_train_s), torch.from_numpy(s_train_s)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(g_val_s), torch.from_numpy(s_val_s)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Training + SWA ensemble over 5 seeds
    epochs = 100
    swa_start = 50
    seeds = [42]
    val_preds_list = []
    last_swa = None

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        model = forward_model().to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-2,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
        )
        swa_model = AveragedModel(model)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            if epoch >= swa_start:
                swa_model.update_parameters(model)

        update_bn(train_loader, swa_model, device=device)
        swa_model.eval()
        last_swa = swa_model

        with torch.no_grad():
            Xv = torch.from_numpy(g_val_s).to(device)
            preds_s = swa_model(Xv).cpu().numpy()
        preds = scaler_y.inverse_transform(preds_s)
        val_preds_list.append(preds)

    val_stack = np.stack(val_preds_list, axis=0)
    val_mean = val_stack.mean(axis=0)
    val_mse = ((val_mean - s_val_orig) ** 2).mean()
    print(f"Final Validation MSE: {val_mse:.6f}")

    model_path = "./model_final.pth"
    torch.save(last_swa.module.state_dict(), model_path)

    x_path = "./x_scaler.save"
    y_path = "./y_scaler.save"
    try:
        joblib.dump(scaler_x, x_path)
    except Exception:
        class _SimpleScaler:
            def __init__(self, mean_, scale_):
                self.mean_ = mean_
                self.scale_ = scale_
            def transform(self, X):
                return (X - self.mean_) / self.scale_
            def inverse_transform(self, X):
                return X * self.scale_ + self.mean_
        xs = _SimpleScaler(scaler_x.mean_, scaler_x.scale_)
        joblib.dump(xs, x_path)
    try:
        joblib.dump(scaler_y, y_path)
    except Exception:
        class _SimpleScalerY:
            def __init__(self, mean_, scale_):
                self.mean_ = mean_
                self.scale_ = scale_
            def transform(self, X):
                return (X - self.mean_) / self.scale_
            def inverse_transform(self, X):
                return X * self.scale_ + self.mean_
        ys = _SimpleScalerY(scaler_y.mean_, scaler_y.scale_)
        joblib.dump(ys, y_path)
