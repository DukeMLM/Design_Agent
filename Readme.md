# Metamaterials Agent: Forward Modeling & Inverse Design

LLM-driven agents that **train a forward surrogate model** (geometry → spectrum) and perform **Neural-Adjoint inverse design** (target spectrum → geometry). 

- 🔁 End-to-end workflow: data prep → forward training → inverse design
- 🧠 iterative code generation & evaluation
- 🧪 Neural-Adjoint backpropagation for inverse design

---

## Quick Start

All commands below are run from the `agent` folder.

### 1) Requirements
- simple requirement in requirements.txt (`pip install -r requirements.txt`)
- full requirement in environment_full.yml. This is the environment of the paper runs; it does not list AIDE, so install it afterwards with `pip install --no-deps aideml==0.2.0`

To dive into the details of forward modeling, please check the used AIDE (https://github.com/WecoAI/aideml).

### 2) OpenAI API key
Set the key in your shell, or put it in a file named `.env` in the `agent` folder (the file is ignored by git):

```
export OPENAI_API_KEY=sk-...        # shell
OPENAI_API_KEY=sk-...               # or one line in agent/.env
```

### 3) Datasets
- Download the data (ADM.zip) in: https://research.repository.duke.edu/record/176?ln=en&v=zip and extract it into ./agent/dataset
- Create the training/validation split the agent reads (`g_training.csv`, `s_training.csv`, `g_validation.csv`, `s_validation.csv`): `python prepare_dataset.py`
- Inverse dataset split in: ./agent/dataset

### 4) Run
- python agent_system.py
- Provide the target (one spectrum) to agent (it will ask the path) in a .csv file (for example: ./dataset/test_s_inverse_test_0.csv)
- It will take more than 20 hours

### 5) Settings
The defaults are in `agent/config.py`. Each one can be changed with an environment variable of the same name (or a line in `agent/.env`):

| Setting | Default | Meaning |
| --- | --- | --- |
| `MSE_TARGET` | 0.002 | Target validation MSE of the forward loop. A target stated in the task description takes precedence. |
| `MAX_DATASET_SIZE` | 41000 | Largest training set the loop may use. |
| `AIDE_STEPS` | 50 | AIDE search steps per code generation. |
| `AIDE_K_FOLD` | 1 | 1 = one hold-out validation on the validation set (no k-fold cross-validation). |
| `AIDE_CODE_MODEL`, `AIDE_FEEDBACK_MODEL` | AIDE defaults | Models AIDE uses to write and to review code. |
| `AIDE_EXEC_TIMEOUT` | 3600 | Time limit (s) for each script AIDE runs during its search. |
| `RUN_TIMEOUT_S` | 36000 | Time limit (s) for one run of the selected script on a larger dataset. |
| `AGENT_MODEL` | o4-mini | Model of the planner and the sub-agents. |
| `NA_BATCH_SIZE`, `NA_BACKPROP_STEPS`, `NA_LR` | 3000, 400, 0.001 | Neural-Adjoint inverse design. |
| `DEVICE` | cuda:0 if available, else cpu | Device of the inverse design. |

---
## Typical Workflow
👤 Hi there! I'm here to assist you with metamaterials deep learning tasks. Briefly: do you want to run Forward training, Inverse design, or Both? (reply: forward / inverse / both)

User: both

👤 Please describe your task and the dataset you have, in one or two paragraphs.

User: The task involves designing and optimizing a deep learning regression model to predict the electromagnetic spectrum from geometry parameters. The goal is to achieve a mean squared error (MSE) of 2e-3.

👤 Could you clarify the model’s input/output dimensions?

User: input 14; output: 2001

Then, agent will perform the forward modeling.

The forward stage writes `best_solution.py` (the selected training script) and `run_history.json`. The inverse stage refactors that script into `modified.py`, retrains it (`model_final.pth`, `x_scaler.save`, `y_scaler.save`) and runs the Neural-Adjoint design; the designed geometry and its surrogate MSE are returned and the spectra are plotted in `real_inverse.png`.

To run the inverse stage alone, `best_solution.py` from a forward run (or a `modified.py` that matches `model_final.pth`) must be in the `agent` folder.

---

We will continue to improve it!

---
## Citation
If you use this repo in your research, please cite it as follows:

```python
@article{Lu2025,
  title = {An Agentic Framework for Autonomous Metamaterial Modeling and Inverse Design},
  ISSN = {2330-4022},
  url = {http://dx.doi.org/10.1021/acsphotonics.5c01514},
  DOI = {10.1021/acsphotonics.5c01514},
  journal = {ACS Photonics},
  publisher = {American Chemical Society (ACS)},
  author = {Lu,  Darui and Malof,  Jordan M. and Padilla,  Willie J.},
  year = {2025},
  month = oct 
}
```
