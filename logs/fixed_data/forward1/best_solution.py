import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import random

# -------------------------
# Hyperparameters
# -------------------------
MAX_EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EMA_DECAY = 0.999
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Ensemble with 10 seeds
SEEDS = list(range(10))

# -------------------------
# Data Loading
# -------------------------
g_train = pd.read_csv("./input/g_training.csv")
s_train = pd.read_csv("./input/s_training.csv")
g_val = pd.read_csv("./input/g_validation.csv")
s_val = pd.read_csv("./input/s_validation.csv")
g_test = pd.read_csv("./input/test_g.csv")

X_train = g_train.values.astype(np.float32)
y_train = s_train.values.astype(np.float32)
X_val = g_val.values.astype(np.float32)
y_val = s_val.values.astype(np.float32)
X_test = g_test.values.astype(np.float32)

scaler_x = StandardScaler().fit(X_train)
X_train = scaler_x.transform(X_train)
X_val = scaler_x.transform(X_val)
X_test = scaler_x.transform(X_test)

train_data = torch.utils.data.TensorDataset(
    torch.tensor(X_train), torch.tensor(y_train)
)
val_data = torch.utils.data.TensorDataset(torch.tensor(X_val), torch.tensor(y_val))

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)


# -------------------------
# Swish Activation
# -------------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# -------------------------
# EMA Class
# -------------------------
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in self.model.named_parameters():
            self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[
                    name
                ]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

    def store_shadow(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_shadow(self, shadow_dict):
        for name in self.shadow:
            self.shadow[name] = shadow_dict[name].clone()


# -------------------------
# Residual Block with SE
# -------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = Swish()

        # Squeeze-and-Excitation sub-block
        self.se_down = nn.Linear(1, dim // reduction)
        self.se_up = nn.Linear(dim // reduction, dim)

    def forward(self, x):
        identity = x
        out = self.ln1(x)
        out = self.act(self.fc1(out))
        out = self.ln2(out)
        out = self.fc2(out)

        # Squeeze-and-Excitation
        se = out.mean(dim=1, keepdim=True)  # Nx1
        se = self.se_down(se)  # Nx(dim//reduction)
        se = self.act(se)
        se = self.se_up(se)  # Nx(dim)
        se = torch.sigmoid(se)
        out = out * se  # Nx(dim)

        return self.act(out + identity)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim=14, hidden_dim=256, out_dim=2001, n_blocks=4):
        super().__init__()
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.act = Swish()
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(n_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.act(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_get_predictions(seed):
    set_seed(seed)

    model = ResidualMLP().to(DEVICE)
    ema = EMA(model, decay=EMA_DECAY)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=False
    )

    best_val_mse = float("inf")
    best_shadow = ema.store_shadow()
    epochs_without_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ema.update()

        model.eval()
        ema.apply_shadow()
        val_preds_list = []
        val_targets_list = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                out = model(batch_X).cpu().numpy()
                val_preds_list.append(out)
                val_targets_list.append(batch_y.numpy())
        ema.restore()

        val_preds_cat = np.concatenate(val_preds_list, axis=0)
        val_targets_cat = np.concatenate(val_targets_list, axis=0)
        val_mse_ema = mean_squared_error(val_targets_cat, val_preds_cat)

        scheduler.step(val_mse_ema)

        if val_mse_ema < best_val_mse:
            best_val_mse = val_mse_ema
            best_shadow = ema.store_shadow()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    ema.load_shadow(best_shadow)
    ema.apply_shadow()

    model.eval()
    val_preds_list = []
    with torch.no_grad():
        for batch_X, _ in val_loader:
            batch_X = batch_X.to(DEVICE)
            out = model(batch_X).cpu().numpy()
            val_preds_list.append(out)
    val_preds_cat = np.concatenate(val_preds_list, axis=0)

    test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        test_preds = model(test_tensor).cpu().numpy()

    ema.restore()
    return val_preds_cat, test_preds, best_val_mse


all_val_preds = []
all_test_preds = []
all_val_mses = []

for sd in SEEDS:
    val_preds_seed, test_preds_seed, val_mse_seed = train_and_get_predictions(sd)
    all_val_preds.append(val_preds_seed)
    all_test_preds.append(test_preds_seed)
    all_val_mses.append(val_mse_seed)

val_preds_ensemble = np.mean(np.stack(all_val_preds, axis=0), axis=0)
val_targets_concat = s_val.values.astype(np.float32)
ensemble_val_mse = mean_squared_error(val_targets_concat, val_preds_ensemble)
print(f"Ensemble Validation MSE: {ensemble_val_mse:.6f}")

test_preds_ensemble = np.mean(np.stack(all_test_preds, axis=0), axis=0)
submission = pd.DataFrame(test_preds_ensemble)
submission.to_csv("./working/submission.csv", index=False)
