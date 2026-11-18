# inverse_NA.py
"""

This module mainly contains the implementation of the Neural Adjoint (NA). The code is adapted from https://github.com/BensonRen/AEM_DIM_Bench

"""

import os
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")  # the figure is only saved to a file
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset

import config as cfg
from common import read_table

DEVICE = torch.device(cfg.DEVICE or ("cuda:0" if torch.cuda.is_available() else "cpu"))


def load_forward_model_class():
    """Return the `forward_model` class defined in modified.py (the verified training script)."""
    try:
        from modified import forward_model
    except ModuleNotFoundError as e:
        if e.name != "modified":
            raise
        raise FileNotFoundError(
            "modified.py was not found. Run the code-verify step first (modifier.py): it writes "
            "modified.py, model_final.pth, x_scaler.save and y_scaler.save."
        ) from e
    return forward_model


def geometry_bounds(x_scaler) -> Tuple[np.ndarray, np.ndarray]:
    """
    Lower and upper bound of every geometry parameter, in the space of `x_scaler`.

    They are taken from the training set of the forward model, so that the inverse
    design stays in the region where the surrogate has seen data.
    """
    candidates = (
        os.path.join(cfg.INPUT_DIR, "g_training.csv"),
        os.path.join(cfg.DATASET_DIR, "g_training.csv"),
    )
    for path in candidates:
        if os.path.exists(path):
            g = np.asarray(x_scaler.transform(read_table(path)), dtype="float32")
            return g.min(axis=0), g.max(axis=0)
    raise FileNotFoundError(
        "The training geometry is needed for the bounds of the design space, but neither "
        f"{candidates[0]} nor {candidates[1]} exists."
    )


def build_tensor(nparray, requires_grad=False):
    return torch.tensor(np.asarray(nparray), requires_grad=requires_grad, device=DEVICE, dtype=torch.float)


def initialize_geometry_eval(config, dim=14, init_from_Xpred=None):
    """Random start points, uniform in [0, 1]; they are mapped to the design space below."""
    if init_from_Xpred is not None:
        return build_tensor(init_from_Xpred, requires_grad=True)
    return torch.rand([config['batch_size'], dim], requires_grad=True, device=DEVICE)


def initialize_from_uniform_to_dataset_distrib(geometry_eval, lower, upper):
    """Map [0, 1] to [lower, upper] for every geometry parameter."""
    return geometry_eval * (upper - lower) + lower


def make_loss(logit=None, labels=None, G=None, lower=None, upper=None):
    if logit is None:
        return None
    mse = F.mse_loss(logit, labels)
    if G is None or lower is None or upper is None:
        return mse
    # Boundary loss of the Neural Adjoint method: zero inside [lower, upper],
    # growing linearly outside, so that the designs stay in the training range.
    mean = (lower + upper) / 2
    half_range = (upper - lower) / 2
    bdy = 0.1 * torch.sum(F.relu(torch.abs(G - mean) - half_range))
    return mse + bdy


class ADMDataSet(Dataset):
    def __init__(self, g, s):
        self.g = g
        self.s = s
    def __len__(self):
        return len(self.g)
    def __getitem__(self, ind):
        return self.g[ind], self.s[ind]


class NeuralAdjointEvaluator:
    def __init__(self, model, simulator,  loader, model_path, config, xscaler, yscaler):
        self.loader = loader
        self.model = model
        self.simulator = simulator
        self.model_path = model_path
        self.config = config
        self.xscaler = xscaler
        self.yscaler = yscaler
        self.lower = build_tensor(config['lower_bound'])
        self.upper = build_tensor(config['upper_bound'])
        self._loaded = False

    def load_models(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
        self.simulator.load_state_dict(torch.load(cfg.MODEL_FILE, map_location=DEVICE))
        for net in (self.model, self.simulator):
            net.eval()  # BatchNorm and Dropout in inference mode
            for p in net.parameters():
                p.requires_grad_(False)  # only the geometry is optimized
        self._loaded = True

    def evaluate_one(self, spectra):
        """
        Inverse design for one target spectrum (in the space of the y scaler).
        Returns (best geometry in the x-scaler space, its predicted spectrum in
        physical units, its MSE against the target in physical units).
        """
        if not self._loaded:
            self.load_models()
        dim = self.config['dim_g']
         # --- normalize input type & shape ---
        if isinstance(spectra, np.ndarray):
            spectra = torch.from_numpy(spectra)
        spectra = spectra.to(DEVICE, dtype=torch.float)
        if spectra.dim() == 1:
            spectra = spectra.unsqueeze(0)  # (1, F)
        if spectra.shape[0] != 1:
            raise ValueError(f"evaluate_one takes one target spectrum, got {spectra.shape[0]}")
        guess = initialize_geometry_eval(self.config, dim)
        optimizer = optim.Adam([guess], lr=self.config['lr'])
        # duplicate target spectra across the batch
        spectra_expand = spectra.repeat(self.config['batch_size'], 1)  # (B, F)
        for step in range(self.config['backprop_step']):
            input_eval = initialize_from_uniform_to_dataset_distrib(guess, self.lower, self.upper)
            optimizer.zero_grad()
            logit = self.model(input_eval)
            loss = make_loss(logit, spectra_expand, input_eval, self.lower, self.upper)
            loss.backward()
            optimizer.step()  # Move one step the optimizer

        with torch.no_grad():
            # the designs after the last step, kept inside the bounds
            input_eval = initialize_from_uniform_to_dataset_distrib(guess, self.lower, self.upper)
            input_eval = torch.max(torch.min(input_eval, self.upper), self.lower)
            output = self.simulator(input_eval).cpu().numpy()
        spec_true = np.asarray(self.yscaler.inverse_transform(spectra.cpu().numpy()), dtype='float32')
        spec_pred = np.asarray(self.yscaler.inverse_transform(output), dtype='float32')
        mse = np.mean((spec_pred - spec_true)**2, axis=1)
        best = int(np.argmin(mse))
        return input_eval[best], spec_pred[best], float(mse[best])

    def evaluate(self):
        self.load_models()

        results = []
        it = 0
        for g, s in self.loader:
            it += 1
            print("Iteration:", it)
            g, s = g.to(DEVICE), s.to(DEVICE)
            xpred, ypred, mse = self.evaluate_one(s)
            print(mse)
            xpred = self.xscaler.inverse_transform([xpred.detach().cpu().numpy()]).astype('float32')
            g = self.xscaler.inverse_transform(g.cpu().numpy()).astype('float32')
            s = self.yscaler.inverse_transform(s.cpu().numpy()).astype('float32')
            results.append((xpred, g, ypred, s, mse))

        mses = [r[4] for r in results]
        output_filename_npz = 'evaluation_results.npz'

        # Extract data from the results list
        xpreds = np.array([r[0] for r in results])
        gs = np.array([r[1] for r in results])
        ypreds_np = np.array([r[2] for r in results])
        ss = np.array([r[3] for r in results])
        mean_mse = np.mean(mses)
        # Save the arrays to a .npz file
        np.savez(output_filename_npz,
                 mean_mse=mean_mse,
                 xpreds=xpreds,
                 gs=gs,
                 ypreds=ypreds_np,
                 ss=ss,
                 mses=np.array(mses))

        print(f"Results saved to {output_filename_npz}")
        return np.mean(mses),results


def _model_config(lower: np.ndarray, upper: np.ndarray) -> Dict[str, Any]:
    return {
        "batch_size": cfg.NA_BATCH_SIZE,
        "backprop_step": cfg.NA_BACKPROP_STEPS,
        "lr": cfg.NA_LR,
        "dim_g": int(lower.shape[0]),
        "lower_bound": lower,
        "upper_bound": upper,
    }


# for all tests
def train_inverse_model(
    g_test_path: str,
    s_test_path: str,
) -> Tuple[float, List[Any]]:
    forward_model = load_forward_model_class()

    x_scaler = joblib.load(cfg.X_SCALER_FILE)
    y_scaler = joblib.load(cfg.Y_SCALER_FILE)
    g_test = np.asarray(x_scaler.transform(read_table(g_test_path)), dtype='float32')
    s_test = np.asarray(y_scaler.transform(read_table(s_test_path)), dtype='float32')
    g_test = g_test[:s_test.shape[0], :]
    print(g_test.shape)
    print("load")

    testset = ADMDataSet(g_test, s_test)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    lower, upper = geometry_bounds(x_scaler)
    model_config = _model_config(lower, upper)

    trained_model = forward_model().to(DEVICE)
    accurate_model = forward_model().to(DEVICE)

    evaluator = NeuralAdjointEvaluator(trained_model, accurate_model, testloader, cfg.MODEL_FILE, model_config, x_scaler, y_scaler)
    final_mse, res = evaluator.evaluate()


    return  final_mse, res


# simulate function;
'''
They following function is used to call the local simulation server to get the real simulation results based on the predicted geometry parameters.
This is very specific to the ADM problem setup and the local server implementation.
default disabled (set RUN_REAL_SIMULATION=1, SIM_SERVER_URL and SIM_CST_FILE to use it).
'''
def trans_radians_to_degrees(radians):
    degrees = radians * (180 / np.pi)
    return degrees


def simulte(xpred):
    url = f"{cfg.SIM_SERVER_URL.rstrip('/')}/simulate"
    xpred = [float(v) for v in xpred]
    params = {
        "h": xpred[0], "p": xpred[1],
        "rma1": xpred[2], "rmi1": xpred[6],
        "rma2": xpred[3], "rmi2": xpred[7],
        "rma3": xpred[4], "rmi3": xpred[8],
        "rma4": xpred[5], "rmi4": xpred[9],
        "theta1": trans_radians_to_degrees(xpred[10]), "theta2": trans_radians_to_degrees(xpred[11]),
        "theta3": trans_radians_to_degrees(xpred[12]), "theta4": trans_radians_to_degrees(xpred[13]),
        "filepath": cfg.SIM_CST_FILE,
    }

    response = requests.post(url, json=params)
    real_simulation = response.json()['absorptance']
    return real_simulation

# for one tests
def inverse_one(
    s_test_path: str,
)-> Dict[str, Any]:
    """
    Inverse design for one target spectrum (a CSV file with one row, physical units).
    Returns {"Surrogate MSE": float, "geometry": [...]} and, when the real
    simulation is enabled, "Real Simulation MSE". Saves the figure real_inverse.png.
    """
    forward_model = load_forward_model_class()

    x_scaler = joblib.load(cfg.X_SCALER_FILE)
    y_scaler = joblib.load(cfg.Y_SCALER_FILE)

    s_raw = read_table(s_test_path)  # target spectrum in physical units
    if s_raw.shape[0] != 1:
        raise ValueError(
            f"inverse_one expects one target spectrum (one row), but {s_test_path} has "
            f"{s_raw.shape[0]} rows. Use train_inverse_model for several targets."
        )
    s_test = np.asarray(y_scaler.transform(s_raw), dtype='float32')
    print("load")

    lower, upper = geometry_bounds(x_scaler)
    model_config = _model_config(lower, upper)

    trained_model = forward_model().to(DEVICE)
    accurate_model = forward_model().to(DEVICE)

    evaluator = NeuralAdjointEvaluator(trained_model, accurate_model, None, cfg.MODEL_FILE, model_config, x_scaler, y_scaler)

    # evaluate_one loads the trained weights into both models before optimizing
    xpred, ypred, mse = evaluator.evaluate_one(s_test)
    print("MSE:", mse)
    xpred = np.asarray(x_scaler.inverse_transform([xpred.cpu().numpy()]))[0]
    result = {"Surrogate MSE": mse, "geometry": [float(v) for v in xpred]}

    plt.figure()
    if cfg.RUN_REAL_SIMULATION: # default False
        real_simulation = np.asarray(simulte(xpred), dtype='float32')
        result["Real Simulation MSE"] = float(mean_squared_error(real_simulation, s_raw[0]))
        plt.plot(real_simulation, label='Real Simulation')

    # both curves in physical units
    plt.plot(ypred, label='Prediction')
    plt.plot(s_raw[0], label='Real Values')
    plt.legend()
    plt.xlabel('Frequency')
    plt.title('Absorption Spectrum')
    plt.grid()
    plt.ylabel('Absorption')
    plt.savefig('./real_inverse.png')
    plt.close('all')
    return result

if __name__ == "__main__":
    test_data_path = './dataset/test_s_inverse_test_0.csv'
    result = inverse_one(test_data_path)
    print(result)

# if __name__ == "__main__":
#     g_test_path = "./dataset/test_g_inverse_test.csv"
#     s_test_path = "./dataset/test_s_inverse_test.csv"

#     final_mse, res = train_inverse_model(g_test_path, s_test_path)
#     print(f"Final MSE: {final_mse}")
#     for xpred, g, ypred, s, mse in res:
#         print(f"xpred: {xpred}, g: {g}, ypred: {ypred}, s: {s}, mse: {mse}")
