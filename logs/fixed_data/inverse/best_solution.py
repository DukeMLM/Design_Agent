import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import joblib


class forward_model(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, output_dim=2001):
        super(forward_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        # Skip connection
        x2 = x2 + x1
        out = self.fc3(x2)
        return out


def train_model(
    modelf,
    optimizer,
    criterion,
    train_x,
    train_y,
    val_x,
    val_y,
    device,
    epochs=5,
    batch_size=64,
):
    dataset_size = train_x.shape[0]
    indices = np.arange(dataset_size)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        np.random.shuffle(indices)
        modelf.train()

        for start_idx in range(0, dataset_size, batch_size):
            end_idx = start_idx + batch_size
            batch_idx = indices[start_idx:end_idx]
            x_batch = torch.tensor(
                train_x[batch_idx], dtype=torch.float32, device=device
            )
            y_batch = torch.tensor(
                train_y[batch_idx], dtype=torch.float32, device=device
            )

            optimizer.zero_grad()
            outputs = modelf(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        modelf.eval()
        with torch.no_grad():
            x_val_t = torch.tensor(val_x, dtype=torch.float32, device=device)
            y_val_t = torch.tensor(val_y, dtype=torch.float32, device=device)
            val_preds = modelf(x_val_t)
            val_loss = criterion(val_preds, y_val_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(modelf.state_dict(), "best_model.pth")

    return best_val_loss


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Read data
    g_train = pd.read_csv("./input/g_training.csv")
    s_train = pd.read_csv("./input/s_training.csv")
    g_val = pd.read_csv("./input/g_validation.csv")
    s_val = pd.read_csv("./input/s_validation.csv")
    g_test = pd.read_csv("./input/test_g.csv")
    s_test = pd.read_csv("./input/test_s.csv")

    # 2. Scale only geometry inputs
    scaler = StandardScaler()
    g_train_scaled = scaler.fit_transform(g_train.values)
    g_val_scaled = scaler.transform(g_val.values)
    g_test_scaled = scaler.transform(g_test.values)
    joblib.dump(scaler, "scaler.save")

    # 3. Convert spectral outputs to NumPy
    s_train_np = s_train.values
    s_val_np = s_val.values
    s_test_np = s_test.values

    # 4. Define and move model to device
    modelf = forward_model(input_dim=14, hidden_dim=128, output_dim=2001).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(modelf.parameters(), lr=1e-3)

    # 5. Train model
    best_val_loss = train_model(
        modelf,
        optimizer,
        criterion,
        g_train_scaled,
        s_train_np,
        g_val_scaled,
        s_val_np,
        device,
        epochs=5,
        batch_size=64,
    )

    # 6. Load best model and evaluate on validation set
    modelf.load_state_dict(torch.load("best_model.pth", map_location=device))
    modelf.eval()
    with torch.no_grad():
        val_x_tensor = torch.tensor(g_val_scaled, dtype=torch.float32, device=device)
        val_y_tensor = torch.tensor(s_val_np, dtype=torch.float32, device=device)
        val_preds = modelf(val_x_tensor)
        val_loss = criterion(val_preds, val_y_tensor).item()
    print("Validation MSE:", val_loss)

    # 7. Inference on test set
    with torch.no_grad():
        test_x_tensor = torch.tensor(g_test_scaled, dtype=torch.float32, device=device)
        test_preds = modelf(test_x_tensor).cpu().numpy()

    # 8. Calculate test MSE
    with torch.no_grad():
        test_y_tensor = torch.tensor(s_test_np, dtype=torch.float32, device=device)
        test_outputs = modelf(test_x_tensor)
        test_loss = criterion(test_outputs, test_y_tensor).item()
    print("Test MSE:", test_loss)

    # 9. Save predictions
    submission = pd.DataFrame(test_preds)
    submission.to_csv("./working/submission.csv", index=False)
