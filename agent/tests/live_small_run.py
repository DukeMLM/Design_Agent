# tests/live_small_run.py
"""
Small live run of the whole agent system. It needs OPENAI_API_KEY and makes
real LLM calls (a few cents, a few minutes on a GPU).

Only the AIDE search is replaced: generate_solution() returns a simple MLP
training script together with the validation MSE that script prints. Everything
else is the real system: planner, spec writer, path verifier, LoopDirector,
the code-data loop, the CodeModifier and the Neural-Adjoint inverse design.
The questions the agents ask are answered by the script below (ANSWERS).

Run from the agent folder:

    python tests/live_small_run.py

Set AGENT_MODEL to choose the model of the planner and the sub-agents.
"""

import asyncio
import builtins
import json
import os
import re
import shutil
import sys

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, AGENT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# a small problem: at most 1,200 training samples and 8 LoopDirector decisions
os.environ.setdefault("MAX_DATASET_SIZE", "1200")
os.environ.setdefault("MAX_LOOP_DECISIONS", "8")
os.environ.setdefault("RUN_TIMEOUT_S", "900")
os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")

import config

config.require_api_key()

from helpers import make_workdir, simple_aide_script

TARGET_MSE = 0.004
ANSWERS = [
    ("forward / inverse / both", "both"),
    ("describe your task", (
        "The task is to design and optimize a deep learning regression model that predicts the "
        "electromagnetic absorption spectrum (2001 points) of an all-dielectric metasurface from "
        "its 14 geometry parameters. The goal is to achieve a validation mean squared error (MSE) "
        f"of {TARGET_MSE}."
    )),
    ("dimension", "input 14; output: 2001"),
    ("test spectra csv path", "./dataset/target_s.csv"),
    ("missing", "stop"),
]
asked = []


def scripted_input(prompt: str = "") -> str:
    """Answer the agents' questions the way a user would."""
    asked.append(prompt.strip())
    low = prompt.lower()
    answer = "No further information. Please continue with what you have."
    for key, text in ANSWERS:
        if key in low:
            answer = text
            break
    print(f"{prompt}{answer}")
    return answer


def simple_model_solution(dataset_size: int) -> dict:
    """Stand-in for the AIDE search: one simple script, run on the data in ./input."""
    from common import run_script

    code = simple_aide_script(epochs=40)
    result = run_script(code, timeout=config.RUN_TIMEOUT_S)
    if not result["ok"]:
        raise RuntimeError(result["stderr"])
    mse = float(re.findall(r"Validation MSE: ([0-9.eE+-]+)", result["stdout"])[-1])
    print(f"[stand-in for AIDE] simple MLP on {dataset_size} samples: validation MSE {mse}")
    return {"mse": mse, "code": code}


def main() -> int:
    work, source, _ = make_workdir()
    sys.path.insert(0, work)
    os.chdir(work)
    print(f"Run folder: {work}\nData: {source}\nAgent model: {config.AGENT_MODEL}\n")

    import agent_system
    import forward_loop

    forward_loop.generate_solution = simple_model_solution
    builtins.input = scripted_input

    asyncio.run(agent_system.main())

    # ---- what the run left behind ----
    print("\n=== Checks ===")
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    check("task description written", os.path.exists(config.TASK_DESC_FILE))
    target = None
    if os.path.exists(config.TASK_CONFIG_FILE):
        with open(config.TASK_CONFIG_FILE) as f:
            target = json.load(f).get("target_mse")
    check("user's target MSE saved", target == TARGET_MSE, f"target_mse = {target}")

    history = []
    if os.path.exists(config.HISTORY_FILE):
        with open(config.HISTORY_FILE) as f:
            history = json.load(f)
    sizes = [h["size"] for h in history]
    check("code-data loop ran", len(history) >= 2, f"{len(history)} entries, sizes {sizes}")
    check("dataset size never above the limit", bool(sizes) and max(sizes) <= config.MAX_DATASET_SIZE)
    check("MSEs stored as numbers", all(h["mse"] is None or isinstance(h["mse"], float) for h in history))
    for h in history:
        print(f"      size {h['size']:>5}  mse {h['mse']}  {h['action']}: {str(h.get('reason', ''))[:90]}")

    for path in (config.BEST_CODE_FILE, config.MODIFIED_CODE_FILE, config.MODEL_FILE,
                 config.X_SCALER_FILE, config.Y_SCALER_FILE, "./real_inverse.png"):
        check(f"{path} written", os.path.exists(path))
    if os.path.exists(config.MODIFIED_CODE_FILE):
        from common import check_script_structure
        with open(config.MODIFIED_CODE_FILE, encoding="utf-8") as f:
            problem = check_script_structure(f.read())
        check("modified.py has forward_model and a main guard", problem is None, problem or "")
    check("the user was asked for the target file", any("csv path" in q.lower() for q in asked))

    keep = os.environ.get("KEEP_RUN_FOLDER") == "1"
    os.chdir(AGENT_DIR)
    if keep:
        print(f"\nRun folder kept: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{sum(checks)} of {len(checks)} checks passed")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
