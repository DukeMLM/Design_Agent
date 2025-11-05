
# test model
from modified import forward_model
from typing import Dict, Any
# from pydantic import BaseModel
import asyncio
import torch
import pandas as pd
import joblib
from typing import Dict, Any
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import pandas as pd
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import csv
# import openai
import os
import torch.optim.lr_scheduler as lr_scheduler

import openai
import os
import requests
from sklearn.metrics import mean_squared_error
from modified import forward_model
# Load from env or set default
api_key =""
openai.api_key = api_key
os.environ['OPENAI_API_KEY'] = ""

"""

This module mainly contains the implementation of the Neural Adjoint (NA). The code is adapted from https://github.com/BensonRen/AEM_DIM_Bench

"""



def get_boundary_lower_bound_upper_bound(dim=14):
    return np.array([2 for _ in range(dim)]), np.array([-1 for _ in range(dim)]), np.array([1 for _ in range(dim)])

def build_tensor(nparray, requires_grad=False):
    return torch.tensor(nparray, requires_grad=requires_grad, device='cuda:0', dtype=torch.float)

def initialize_geometry_eval(config, dim=14, init_from_Xpred=None):
    if init_from_Xpred is not None:
        return build_tensor(init_from_Xpred, requires_grad=True)
    return torch.rand([config['batch_size'], dim], requires_grad=True, device='cuda:0')

def initialize_from_uniform_to_dataset_distrib(geometry_eval, dim=14):
    X_range, X_lower_bound, _ = get_boundary_lower_bound_upper_bound(dim=dim)
    return geometry_eval * build_tensor(X_range) + build_tensor(X_lower_bound)

def make_loss(logit=None, labels=None, G=None):
    if logit is None:
        return None
    # ignore boundary penalty
    mse = F.mse_loss(logit, labels)
    return mse

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

    def load_models(self):
        self.model.load_state_dict(torch.load(self.model_path))
        self.simulator.load_state_dict(torch.load("./model_final.pth"))

    def evaluate_one(self, spectra):
        dim = self.config['dim_g']
         # --- normalize input type & shape ---
        if isinstance(spectra, np.ndarray):
            spectra = torch.from_numpy(spectra)
        spectra = spectra.to('cuda:0', dtype=torch.float)
        if spectra.dim() == 1:
            spectra = spectra.unsqueeze(0)  # (1, F)
        guess = initialize_geometry_eval(self.config, dim)
        optimizer = optim.Adam([guess], lr=self.config['lr'])
        # self.lr_scheduler = self.make_lr_scheduler(optimizer)
        # duplicate target spectra across the batch
        spectra_expand = spectra.repeat(self.config['batch_size'], 1)  # (B, F)
        for step in range(self.config['backprop_step']):
            input_eval = initialize_from_uniform_to_dataset_distrib(guess,dim)
            # input_eval = self.to_scaled(input_eval_phys)
            optimizer.zero_grad()
            logit = self.model(input_eval)
            # print(logit.shape, spectra_expand.shape)
            loss = make_loss(logit, spectra_expand, input_eval)
            loss.backward()
            optimizer.step()  # Move one step the optimizer
            # self.lr_scheduler.step(loss.item())

        output = self.simulator(input_eval).detach()
        output = output.cpu().numpy()
        spec_true = self.yscaler.inverse_transform(spectra_expand.cpu()).astype('float32')
        spec_pred = self.yscaler.inverse_transform(output).astype('float32')
        mse = np.mean((spec_pred - spec_true)**2, axis=1)
        best = np.argmin(mse)
        # best_phys = self.to_phys(input_eval[best]).cpu().detach().numpy()
        return input_eval[best], spec_pred[best], mse[best]

    def evaluate(self):
        self.load_models()
        self.model.eval()
        self.simulator.eval()

        results = []
        it = 0
        for g, s in self.loader:
            it += 1
            print("Iteration:", it)
            g, s = g.cuda(), s.cuda()
            print(s)
            xpred, ypred, mse = self.evaluate_one(s)
            print(mse)
            xpred = self.xscaler.inverse_transform([xpred.detach().cpu().numpy()]).astype('float32')
            g = self.xscaler.inverse_transform(g.cpu().numpy()).astype('float32')
            # ypred = self.yscaler.inverse_transform([ypred])[0]
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


# for all tests
def train_inverse_model(
    g_test_path: str,
    s_test_path: str,
) -> Dict[str, str]:
    model1 = forward_model().cuda()
    model1.load_state_dict(torch.load("./model_final.pth"))
    model1.eval()
    
    backprop_steps=400
    batch_size = 3000
    learning_rate=0.001


    g_test = pd.read_csv(g_test_path, header=None).values.astype('float32')
    s_test = pd.read_csv(s_test_path, header=None).values.astype('float32')

    x_scaler = joblib.load("./x_scaler.save")
    g_test = x_scaler.transform(g_test).astype('float32')
    dim_g = g_test.shape[1]
    g_test = g_test[:s_test.shape[0], :]
    print(g_test.shape)
    y_scaler = joblib.load("./y_scaler.save")
    s_test = y_scaler.transform(s_test).astype('float32')
    print("load")

    testset = ADMDataSet(g_test, s_test)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    model_config = {
        "batch_size": batch_size,
        "backprop_step": backprop_steps,
        "lr": learning_rate,
        "dim_g": dim_g,
    }

    trained_model = forward_model().cuda()
    accurate_model = forward_model().cuda()
    model_path = f"./model_final.pth"


    evaluator = NeuralAdjointEvaluator(trained_model, accurate_model, testloader, model_path, model_config, x_scaler, y_scaler)
    final_mse, res = evaluator.evaluate()


    return  final_mse, res


# simulate function;
'''
They following function is used to call the local simulation server to get the real simulation results based on the predicted geometry parameters.
This is very specific to the ADM problem setup and the local server implementation.
default disabled.
'''
def trans_radians_to_degrees(radians):
    degrees = radians * (180 / np.pi)
    return degrees

link = "http://10.197.59.167:8000/" # use the local IP


def simulte(xpred):
    url = f"{link}/simulate" 
    filepath = r"D:\Study\agent\simulation_test\example1\dl_infra_noSub_random1.cst"
    params = {
        "h": xpred[0], "p": xpred[1],
        "rma1": xpred[2], "rmi1": xpred[6],
        "rma2": xpred[3], "rmi2": xpred[7],
        "rma3": xpred[4], "rmi3": xpred[8],
        "rma4": xpred[5], "rmi4": xpred[9],
        "theta1": trans_radians_to_degrees(xpred[10]), "theta2": trans_radians_to_degrees(xpred[11]), 
        "theta3": trans_radians_to_degrees(xpred[12]), "theta4": trans_radians_to_degrees(xpred[13]),
        "filepath": filepath,
    }

    response = requests.post(url, json=params)
    real_simulation = response.json()['absorptance']
    return real_simulation

# for one tests
def inverse_one(
    s_test_path: str,
)-> Dict[str, str]:
    model1 = forward_model().cuda()
    model1.load_state_dict(torch.load("./model_final.pth"))
    model1.eval()
    
    backprop_steps=400
    batch_size = 3000
    learning_rate=0.001

    g_test_path = "./dataset/g_training.csv"
    g_test = pd.read_csv(g_test_path, header=None).values.astype('float32')
    s_test = pd.read_csv(s_test_path, header=None).values.astype('float32')

    x_scaler = joblib.load("./x_scaler.save")
    g_test = x_scaler.transform(g_test).astype('float32')
    dim_g = g_test.shape[1]
    g_test = g_test[:s_test.shape[0], :]
    # print(g_test.shape)
    y_scaler = joblib.load("./y_scaler.save")
    s_test = y_scaler.transform(s_test).astype('float32')
    print("load")

    testset = ADMDataSet(g_test, s_test)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    model_config = {
        "batch_size": batch_size,
        "backprop_step": backprop_steps,
        "lr": learning_rate,
        "dim_g": dim_g,
    }

    trained_model = forward_model().cuda()
    accurate_model = forward_model().cuda()
    model_path = f"./model_final.pth"


    evaluator = NeuralAdjointEvaluator(trained_model, accurate_model, testloader, model_path, model_config, x_scaler, y_scaler)

    xpred, ypred, mse = evaluator.evaluate_one(s_test)
    print("MSE:", mse)
    xpred = x_scaler.inverse_transform([xpred.cpu().detach().numpy()])[0]
    if False: # default False
        real_simulation = simulte(xpred)
        mse_real = mean_squared_error(real_simulation, s_test[0])
        plt.plot(real_simulation, label='Real Simulation')

    # plt.plot(real_simulation, label='Real Simulation')
    plt.plot(ypred, label='Prediction')
    plt.plot(s_test[0], label='Real Values')
    plt.legend()
    plt.xlabel('Frequency')
    plt.title('Absorption Spectrum')
    plt.grid()
    plt.ylabel('Absorption')
    plt.savefig('./real_inverse.png')
    plt.close('all')
    if False:
        return {"Surrogate MSE": mse, "Real Simulation MSE": mse_real}
    return {"Surrogate MSE": mse}

if __name__ == "__main__":
    test_data_path = './dataset/test_s_inverse_test_0.csv'
    result = inverse_one(test_data_path)
    print(result)

# if __name__ == "__main__":
#     g_test_path = "./dataset/g_training.csv"
#     s_test_path = "./dataset/test_s_inverse_test.csv"

#     final_mse, res = train_inverse_model(g_test_path, s_test_path)
#     print(f"Final MSE: {final_mse}")
#     for xpred, g, ypred, s, mse in res:
#         print(f"xpred: {xpred}, g: {g}, ypred: {ypred}, s: {s}, mse: {mse}")