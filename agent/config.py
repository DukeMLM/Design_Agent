# config.py
"""
Shared settings for the agent system.

Every value below can be changed without editing the code: set the environment
variable of the same name, or put it in a `.env` file next to this script.

The OpenAI API key is read from the OPENAI_API_KEY environment variable (or from
`.env`). No module overwrites it.
"""

import os

try:  # python-dotenv is installed together with aideml
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:  # the shell environment still works without it
    pass


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name) or default)


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name) or default)


def require_api_key() -> None:
    """Stop early with a clear message when no OpenAI API key is available."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Export it in your shell "
            "(export OPENAI_API_KEY=sk-...) or put OPENAI_API_KEY=sk-... in agent/.env"
        )


# ---------------- LLM used by the planner and the sub-agents ----------------
AGENT_MODEL = os.environ.get("AGENT_MODEL") or "o4-mini"

# ---------------- forward loop ----------------
# Default target for the validation MSE. A target stated by the user in the task
# description (saved by the spec writer in TASK_CONFIG_FILE) takes precedence.
MSE_TARGET = _float("MSE_TARGET", 0.002)
# Largest training set the loop may use.
MAX_DATASET_SIZE = _int("MAX_DATASET_SIZE", 41000)
# Upper limit on the number of LoopDirector decisions in one run.
MAX_LOOP_DECISIONS = _int("MAX_LOOP_DECISIONS", 60)
# Wall-clock limit (seconds) for one run of a generated training script.
RUN_TIMEOUT_S = _int("RUN_TIMEOUT_S", 36000)

# ---------------- AIDE ----------------
AIDE_STEPS = _int("AIDE_STEPS", 50)
# 1 = a single hold-out validation on the provided validation set (no k-fold).
AIDE_K_FOLD = _int("AIDE_K_FOLD", 1)
# Leave empty to use AIDE's own default models.
AIDE_CODE_MODEL = os.environ.get("AIDE_CODE_MODEL") or None
AIDE_FEEDBACK_MODEL = os.environ.get("AIDE_FEEDBACK_MODEL") or None
# Time limit (seconds) AIDE gives each script it runs while searching.
AIDE_EXEC_TIMEOUT = _int("AIDE_EXEC_TIMEOUT", 3600)

# ---------------- inverse design (Neural Adjoint) ----------------
NA_BATCH_SIZE = _int("NA_BATCH_SIZE", 3000)
NA_BACKPROP_STEPS = _int("NA_BACKPROP_STEPS", 400)
NA_LR = _float("NA_LR", 0.001)
# Empty = "cuda:0" when a GPU is available, otherwise "cpu".
DEVICE = os.environ.get("DEVICE") or None

# Real CST simulation of the designed geometry (off by default; needs the lab's
# simulation server).
RUN_REAL_SIMULATION = (os.environ.get("RUN_REAL_SIMULATION") or "0") == "1"
SIM_SERVER_URL = os.environ.get("SIM_SERVER_URL") or "http://10.197.59.167:8000"
SIM_CST_FILE = os.environ.get("SIM_CST_FILE") or (
    r"D:\Study\agent\simulation_test\example1\dl_infra_noSub_random1.cst"
)

# ---------------- files (relative to the agent/ folder) ----------------
TASK_DESC_FILE = "./desp_flex.txt"
TASK_CONFIG_FILE = "./task_config.json"
HISTORY_FILE = "./run_history.json"
BEST_CODE_FILE = "./best_solution.py"
MODIFIED_CODE_FILE = "./modified.py"
MODEL_FILE = "./model_final.pth"
X_SCALER_FILE = "./x_scaler.save"
Y_SCALER_FILE = "./y_scaler.save"
DATASET_DIR = "./dataset"
INPUT_DIR = "./input"
