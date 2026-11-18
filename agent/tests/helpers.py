# tests/helpers.py
"""
Shared pieces of the small tests: the simple training script that stands in
for the AIDE search, the test data, and scripted replacements for the LLM.
"""

import os
import shutil
import tempfile

import numpy as np

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_TRAIN, N_VAL = 2000, 400
DIM_G, DIM_S = 14, 2001

# ----------------------------------------------------------------------------
# Stand-in for the AIDE search: the kind of flat script AIDE writes.
# ----------------------------------------------------------------------------
_TRAINING_BODY = '''
torch.manual_seed(0)
np.random.seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
g_train = pd.read_csv("./input/g_training.csv").values.astype("float32")
s_train = pd.read_csv("./input/s_training.csv").values.astype("float32")
g_val = pd.read_csv("./input/g_validation.csv").values.astype("float32")
s_val = pd.read_csv("./input/s_validation.csv").values.astype("float32")
x_scaler = StandardScaler().fit(g_train)
y_scaler = StandardScaler().fit(s_train)
X_tr = torch.tensor(x_scaler.transform(g_train), dtype=torch.float32, device=device)
Y_tr = torch.tensor(y_scaler.transform(s_train), dtype=torch.float32, device=device)
X_va = torch.tensor(x_scaler.transform(g_val), dtype=torch.float32, device=device)
model = MODEL_CLASS(g_train.shape[1], s_train.shape[1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(X_tr.shape[0], device=device)
    for i in range(0, X_tr.shape[0], 128):
        idx = perm[i:i + 128]
        if idx.numel() < 2:
            continue
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(X_tr[idx]), Y_tr[idx])
        loss.backward()
        opt.step()
model.eval()
with torch.no_grad():
    pred = y_scaler.inverse_transform(model(X_va).cpu().numpy())
val_mse = float(np.mean((pred - s_val) ** 2))
'''

_MODEL_DEF = '''
class MODEL_CLASS(nn.Module):
    def __init__(self, d_in=14, d_out=2001, width=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, d_out),
        )

    def forward(self, x):
        return self.net(x)
'''

_IMPORTS = '''import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
'''


def simple_aide_script(epochs: int = 15) -> str:
    """A flat training script: no main guard, its own class name, writes ./working/submission.csv."""
    code = _IMPORTS + _MODEL_DEF + _TRAINING_BODY + '''
print(f"Validation MSE: {val_mse:.6f}")
pd.DataFrame(pred).to_csv("./working/submission.csv", index=False)
'''
    return code.replace("MODEL_CLASS", "ResNetLike").replace("EPOCHS", str(epochs))


def refactored_script(epochs: int = 150) -> str:
    """What the CodeModifier has to return for the script above."""
    body = "\n".join("    " + line if line else line for line in _TRAINING_BODY.splitlines())
    code = _IMPORTS + _MODEL_DEF + '\nif __name__ == "__main__":' + body + '''
    torch.save(model.state_dict(), "./model_final.pth")
    joblib.dump(x_scaler, "./x_scaler.save")
    joblib.dump(y_scaler, "./y_scaler.save")
    print(f"Final Validation MSE: {val_mse}")
'''
    return code.replace("MODEL_CLASS", "forward_model").replace("EPOCHS", str(epochs))


# ----------------------------------------------------------------------------
# Test data
# ----------------------------------------------------------------------------
def _copy_head(src: str, dst: str, rows: int) -> None:
    with open(src, "r") as fin, open(dst, "w") as fout:
        for i, line in enumerate(fin):
            if i >= rows:
                break
            fout.write(line)


def _synthetic(n: int, rng) -> tuple:
    """Smooth geometry -> spectrum map with the shape of the ADM data (14 -> 2001)."""
    lo = np.array([0.3, 1.0] + [0.05] * 8 + [-0.8] * 4, dtype="float32")
    hi = np.array([0.6, 1.5] + [0.25] * 8 + [0.8] * 4, dtype="float32")
    g = rng.uniform(lo, hi, size=(n, DIM_G)).astype("float32")
    u = (g - lo) / (hi - lo)
    f = np.linspace(0.0, 1.0, DIM_S, dtype="float32")[None, :]
    s = np.zeros((n, DIM_S), dtype="float32")
    for k in range(4):
        center = 0.15 + 0.7 * (0.5 * u[:, k:k + 1] + 0.5 * u[:, 10 + k:11 + k])
        width = 0.02 + 0.06 * u[:, 4 + k:5 + k]
        height = 0.2 + 0.6 * u[:, 6 + k:7 + k]
        s += height * width ** 2 / ((f - center) ** 2 + width ** 2)
    return g, np.clip(s, 0.0, 1.0)


def make_workdir() -> tuple:
    work = tempfile.mkdtemp(prefix="design_agent_test_")
    os.makedirs(os.path.join(work, "dataset"))
    src = os.path.join(AGENT_DIR, "dataset")
    names = ["g_training.csv", "s_training.csv", "g_validation.csv", "s_validation.csv"]
    rows = [N_TRAIN, N_TRAIN, N_VAL, N_VAL]
    target = os.path.join(work, "dataset", "target_s.csv")
    if all(os.path.exists(os.path.join(src, n)) for n in names):
        for name, n in zip(names, rows):
            _copy_head(os.path.join(src, name), os.path.join(work, "dataset", name), n)
        shutil.copy(os.path.join(src, "test_s_inverse_test_0.csv"), target)
        source = "real ADM data (subset)"
    else:
        rng = np.random.default_rng(0)
        g, s = _synthetic(N_TRAIN + N_VAL + 1, rng)
        parts = [g[:N_TRAIN], s[:N_TRAIN], g[N_TRAIN:N_TRAIN + N_VAL], s[N_TRAIN:N_TRAIN + N_VAL]]
        for name, array in zip(names, parts):
            np.savetxt(os.path.join(work, "dataset", name), array, delimiter=",", fmt="%.8e")
        np.savetxt(target, s[-1:], delimiter=",", fmt="%.8e")
        source = "synthetic data (run prepare_dataset.py to test on the ADM data)"
    return work, source, target


# ----------------------------------------------------------------------------
# Scripted replacements for the LLM
# ----------------------------------------------------------------------------
class FakeResult:
    def __init__(self, text):
        self.final_output = text


class FakeRunner:
    """Replaces agents.Runner inside one module: returns scripted answers and records the prompts."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []
        self.inputs = []

    async def run(self, agent, user_input, **kwargs):
        self.prompts.append(agent.instructions)
        self.inputs.append(user_input)
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return FakeResult(answer)
