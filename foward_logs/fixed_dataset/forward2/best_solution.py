import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load data
g_train = pd.read_csv("./input/g_training.csv").values
s_train = pd.read_csv("./input/s_training.csv").values
g_val = pd.read_csv("./input/g_validation.csv").values
s_val = pd.read_csv("./input/s_validation.csv").values
g_test = pd.read_csv("./input/test_g.csv").values

# Combine train+val
X_all = np.vstack([g_train, g_val])
y_all = np.vstack([s_train, s_val])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.SiLU())
        self.fc2 = nn.Sequential(
            nn.Linear(256, 512), nn.LayerNorm(512), nn.SiLU(), nn.Dropout(0.3)
        )
        self.fc3 = nn.Sequential(nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU())
        self.shortcut = nn.Linear(in_dim, 256)
        self.out = nn.Linear(256, out_dim)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        x3 = self.fc3(x2) + self.shortcut(x)
        return self.out(x3)


class TabTransformerReg(nn.Module):
    def __init__(self, num_features, out_dim, emb_dim=256, num_heads=16, num_layers=4):
        super().__init__()
        self.emb = nn.Linear(1, emb_dim)
        self.pos_emb = nn.Parameter(torch.randn(num_features, emb_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            activation="gelu",
            batch_first=False,
        )
        self.trans = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = ResMLP(num_features * emb_dim, out_dim)

    def forward(self, x):
        B, F = x.size()
        e = self.emb(x.unsqueeze(-1)) + self.pos_emb.unsqueeze(0)
        t = e.permute(1, 0, 2)
        out = self.trans(t).permute(1, 0, 2).reshape(B, -1)
        return self.head(out)


def smoothness_penalty(y):
    diff2 = y[:, 2:] - 2 * y[:, 1:-1] + y[:, :-2]
    return torch.mean(diff2**2)


# Cross-validation setup
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
oof_preds = np.zeros_like(y_all, dtype=np.float32)
test_preds = np.zeros((g_test.shape[0], y_all.shape[1]), dtype=np.float32)

# Loss and hyperparameters
criterion = nn.SmoothL1Loss()
lambda_smooth = 1e-4
num_tta = 100
tta_noise = 0.002
pca_dims = 200

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_all), 1):
    X_tr, X_va = X_all[tr_idx], X_all[va_idx]
    y_tr, y_va = y_all[tr_idx], y_all[va_idx]

    scaler_x = StandardScaler().fit(X_tr)
    X_tr_s = scaler_x.transform(X_tr)
    X_va_s = scaler_x.transform(X_va)
    X_te_s = scaler_x.transform(g_test)

    scaler_y = StandardScaler().fit(y_tr)
    y_tr_s = scaler_y.transform(y_tr)

    pca = PCA(n_components=pca_dims, random_state=seed).fit(y_tr_s)
    y_tr_pca = pca.transform(y_tr_s)

    ds_tr = TensorDataset(
        torch.tensor(X_tr_s, dtype=torch.float32),
        torch.tensor(y_tr_pca, dtype=torch.float32),
    )
    loader = DataLoader(ds_tr, batch_size=128, shuffle=True, num_workers=0)

    model = TabTransformerReg(
        num_features=14, out_dim=pca_dims, emb_dim=256, num_heads=16, num_layers=4
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    swa_start, epochs = 25, 50
    swa_lr = 5e-4
    cos_scheduler = CosineAnnealingLR(optimizer, T_max=swa_start, eta_min=swa_lr)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(
        optimizer,
        swa_lr=swa_lr,
        anneal_strategy="cos",
        anneal_epochs=epochs - swa_start,
    )

    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            xb_noisy = xb + torch.randn_like(xb) * 0.01
            pred = model(xb_noisy)
            loss = criterion(pred, yb) + lambda_smooth * smoothness_penalty(pred)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if ep < swa_start:
            cos_scheduler.step()
        else:
            swa_model.update_parameters(model)
            swa_scheduler.step()

    swa_model.to(device)
    update_bn(loader, swa_model, device=device)

    # Validation TTA
    swa_model.train()
    X_va_t = torch.tensor(X_va_s, dtype=torch.float32).to(device)
    sum_pred = np.zeros((X_va_s.shape[0], pca_dims), np.float32)
    with torch.no_grad():
        for _ in range(num_tta):
            xb_noisy = X_va_t + torch.randn_like(X_va_t) * tta_noise
            sum_pred += swa_model(xb_noisy).cpu().numpy()
    avg_pca = sum_pred / num_tta
    pred_s = pca.inverse_transform(avg_pca)
    pred = scaler_y.inverse_transform(pred_s)
    mse = np.mean((pred - y_va) ** 2)
    print(f"Fold {fold} Val MSE: {mse:.6f}")
    oof_preds[va_idx] = pred

    # Test TTA
    swa_model.train()
    X_te_t = torch.tensor(X_te_s, dtype=torch.float32).to(device)
    sum_test = np.zeros((g_test.shape[0], pca_dims), np.float32)
    with torch.no_grad():
        for _ in range(num_tta):
            xb_noisy = X_te_t + torch.randn_like(X_te_t) * tta_noise
            sum_test += swa_model(xb_noisy).cpu().numpy()
    avg_test = sum_test / num_tta
    pred_te_s = pca.inverse_transform(avg_test)
    test_preds += scaler_y.inverse_transform(pred_te_s)

# Overall OOF
oof_mse = np.mean((oof_preds - y_all) ** 2)
print(f"OOF MSE: {oof_mse:.6f}")

os.makedirs("./working", exist_ok=True)
pd.DataFrame(test_preds / n_splits).to_csv(
    "./working/submission.csv", index=False, header=False
)
print("Saved test predictions to ./working/submission.csv")
