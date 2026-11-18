# forward_loop.py
import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.utils import shuffle

# OpenAI Agents SDK
from agents import Agent, Runner, ModelSettings

import config
from common import last_line, read_table, run_script

# How AIDE is asked to evaluate its solutions. The metric is measured once, on the
# provided validation set (no k-fold), and printed in a fixed format so that the
# value parsed by parse_mse_async() is the one AIDE reports.
AIDE_EVAL = (
    "Use Mean Squared Error (MSE) between predicted and true spectra, measured once on the "
    "provided validation set (g_validation.csv / s_validation.csv). Do not use k-fold "
    "cross-validation. Print the final value on its own line as 'Validation MSE: <value>'."
)


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _new_experiment(goal: str):
    """Create an AIDE experiment on ./input with the settings of config.py."""
    import aide  # imported here: only the code generation needs it

    exp = aide.Experiment(data_dir=config.INPUT_DIR, goal=goal, eval=AIDE_EVAL)
    # 1 = single hold-out validation: AIDE then does not ask for k-fold cross-validation
    exp.cfg.agent.k_fold_validation = config.AIDE_K_FOLD
    if config.AIDE_CODE_MODEL:
        exp.cfg.agent.code.model = config.AIDE_CODE_MODEL
    if config.AIDE_FEEDBACK_MODEL:
        exp.cfg.agent.feedback.model = config.AIDE_FEEDBACK_MODEL
    exp.cfg.exec.timeout = config.AIDE_EXEC_TIMEOUT
    exp.interpreter.timeout = config.AIDE_EXEC_TIMEOUT
    return exp


def generate_solution(dataset_size: int) -> Dict[str, Any]:
    """
    Spins up an AIDE experiment, reading the goal prompt from a text file
    and auto‑injecting the current dataset size.
    Returns {"mse": float or None, "code": str}.
    """
    # 1) Read the base goal description from a .txt file
    with open(config.TASK_DESC_FILE, "r", encoding="utf-8") as f:
        base_goal = f.read()

    # 2) Append a line about the current dataset size
    goal = (
        base_goal.strip()
        + f"\n\nCurrent dataset size: {dataset_size} samples"
    )

    exp = _new_experiment(goal)
    best = exp.run(steps=config.AIDE_STEPS)
    print(best.valid_metric)
    return {"mse": _to_float(best.valid_metric), "code": best.code}

def fine_tune_hp(dataset_size: int, code:str) -> Dict[str, Any]:
    """
    Not used currently
    """
    # 1) Read the base goal description from a .txt file
    with open(config.TASK_DESC_FILE, "r", encoding="utf-8") as f:
        base_goal = f.read()

    # 2) Append a line about the current dataset size
    goal = (
        base_goal.strip()
        + f"\n\nCurrent dataset size: {dataset_size} samples"
        + f"\n You should only use the following code to fine-tune the hyperparameters of the model."
        + f"\n\nCurrent code: {code}"
    )

    exp = _new_experiment(goal)
    best = exp.run(steps=config.AIDE_STEPS)
    print(best.valid_metric)
    return {"mse": _to_float(best.valid_metric), "code": best.code}

def generate_dataset(dataset_size: int) -> int:
    """
    Regenerate the training and validation CSVs in ./input/
    with the given dataset_size (by subsampling from the full dataset).
    This is to simulate generating new data as mentioned in the paper.
    Returns the number of training samples actually written.
    """
    src = config.DATASET_DIR
    dst = config.INPUT_DIR
    needed = ["g_training.csv", "s_training.csv", "g_validation.csv", "s_validation.csv"]
    missing = [name for name in needed if not os.path.exists(os.path.join(src, name))]
    if missing:
        raise FileNotFoundError(
            f"Missing in {src}: {', '.join(missing)}. Download ADM.zip, extract it into {src} "
            "and run `python prepare_dataset.py` to create the training/validation split."
        )
    os.makedirs(dst, exist_ok=True)

    # --- 1) Regenerate the smaller training split ---
    # Here using the down sampling approach to simulate the real generation process
    X_full = read_table(os.path.join(src, "g_training.csv"))
    y_full = read_table(os.path.join(src, "s_training.csv"))
    X_shuf, y_shuf = shuffle(X_full, y_full, random_state=1)
    n_train = min(int(dataset_size), X_shuf.shape[0])
    X_small, y_small = X_shuf[:n_train], y_shuf[:n_train]
    # the files in ./input have a header row so the generated code can read them
    pd.DataFrame(X_small).to_csv(os.path.join(dst, "g_training.csv"), index=False)
    pd.DataFrame(y_small).to_csv(os.path.join(dst, "s_training.csv"), index=False)

    Xv_full = read_table(os.path.join(src, "g_validation.csv"))
    yv_full = read_table(os.path.join(src, "s_validation.csv"))
    Xv_shuf, yv_shuf = shuffle(Xv_full, yv_full, random_state=1)
    n_val = max(1, min(int(0.1 * n_train), Xv_shuf.shape[0]))
    Xv_small, yv_small = Xv_shuf[:n_val], yv_shuf[:n_val]
    pd.DataFrame(Xv_small).to_csv(os.path.join(dst, "g_validation.csv"), index=False)
    pd.DataFrame(yv_small).to_csv(os.path.join(dst, "s_validation.csv"), index=False)
    print(f"Generated {n_train} training and {n_val} validation samples in {dst}/")
    return n_train

def evaluate_solution(code: str, dataset_size: int) -> Dict[str, Any]:
    """
    1) Subsample with `dataset_size` samples,
       writing CSVs with a header row so the generated code can read them.
    2) Dump the AIDE-generated `code` to a temp .py file and run it.
    Returns {"ok", "returncode", "output"}: "output" is stdout (plus stderr, if
    any) for downstream parsing; "ok" is False when the script crashed or ran
    longer than config.RUN_TIMEOUT_S.
    """

    # --- 1) Regenerate the training split with the new size ---
    generate_dataset(dataset_size)

    # --- 2) Run the AIDE code, capturing everything ---
    result = run_script(code, timeout=config.RUN_TIMEOUT_S)

    # --- 3) Combined stdout+stderr for downstream parsing ---
    output = result["stdout"]
    if result["stderr"]:
        output += "\n=== STDERR ===\n" + result["stderr"]
    return {"ok": result["ok"], "returncode": result["returncode"], "output": output}


_VALIDATION_MSE_RE = re.compile(
    r"(?i)\bval(?:idation)?\s*mse\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

_NUMERIC_RE = re.compile(r"^\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*$")

# the LLM fallback only needs the end of a long training log
_MAX_LOG_CHARS = 8000


async def parse_mse_async(raw_output: str) -> float:
    """
    Return the (last) Validation MSE printed by a training script as a float.
    1) Regex first (robust to case and sci notation).
    2) LLM fallback that must return only a numeric string.
    Raises ValueError if not found or not a valid float.
    """
    matches = _VALIDATION_MSE_RE.findall(raw_output)
    if matches:
        if float(matches[-1]) > 0:
            return float(matches[-1])

    # LLM fallback via async Runner.run
    log_tail = raw_output[-_MAX_LOG_CHARS:]
    prompt = (
    "I ran a training script and got the following output:\n\n"
    f"{log_tail}\n\n"
    "Extract the **single numeric Validation MSE** value from the text. The text may not be perfectly formatted, so be robust to minor typos (e.g., 'va\\lidation MSE').\n"
    "If it’s not present, return only `null`.\n"
    "Respond with the number **only**, parsable as a float. Do **not** include any text or explanation."
    )
    agent = Agent(name="MSEExtractor", instructions=prompt, tools=[])
    resp = await Runner.run(agent, "Return the Validation MSE.")
    val_str = resp.final_output.strip()
    if val_str.lower() == "null":
        raise ValueError("Validation MSE not found in output (LLM fallback returned null).")
    if not _NUMERIC_RE.fullmatch(val_str):
        raise ValueError(f"LLM fallback returned non-numeric text: {val_str!r}")
    return float(val_str)


def load_target_mse() -> float:
    """
    Target validation MSE of the loop: the value the user gave in the task
    description (saved by the spec writer), otherwise config.MSE_TARGET.
    """
    try:
        with open(config.TASK_CONFIG_FILE, "r", encoding="utf-8") as f:
            value = _to_float(json.load(f).get("target_mse"))
    except (OSError, ValueError, AttributeError):
        value = None
    return value if value and value > 0 else config.MSE_TARGET


async def decide_next_step(
    history: List[Dict[str, Any]],
    mse_threshold: float = config.MSE_TARGET,
    max_size: int = config.MAX_DATASET_SIZE,
) -> Dict[str, Any]:
    """
    Given history of {"size":…, "mse":…, "action":"generated" or "tested"},
    decide next action:
      • test       (on a larger dataset)
      • generate   (new code)
      • done       (low efficiency or threshold reached)
    """
    prompt =  f"""
You are an intelligent controller for an iterative ML loop that alternates between

  • **Generating** new model code via AIDE
  • **Testing** that code on progressively larger datasets

Objectives
1. Reach a validation MSE ≤ {mse_threshold} **before** the dataset exceeds {max_size} samples.
2. Minimize both (a) the number of extra samples and (b) the number of AIDE code-generation calls.

History
{json.dumps(history)}

Reading the history
• "generated" entries hold the MSE of newly generated code. With "kept_code": "previous" the new code was not better than the code in use, so it was discarded and the earlier code is still the current one.
• "test_failed" entries mean the current code crashed or printed no validation MSE at that dataset size ("mse" is null); the reason holds the error.

Guidelines for the next decision
────────────────────────────────

Performance-versus-cost reasoning
• For each pair of consecutive **test** runs, compute

 ΔMSE/Δsamples = |MSE₍t-1₎ − MSE₍t₎| ÷ (size₍t₎ − size₍t-1₎).

• **Reflect** on recent trends before acting.  Consider at least:

  1. How does the latest ΔMSE/Δsamples compare with the typical rate over the last few tests?
  2. Is the current improvement meaningful in the context of your overall progress?
  3. Could a small MSE spike be noise that warrants one confirmation *test*?
  4. Does the evidence suggest the current code has saturated (diminishing gains despite larger data)?
  5. Was a **generate** action performed recently at a similar dataset size, and if so did it give only a marginal or worse MSE?

Avoid redundant code generations
• Only consider **generate** after the dataset size has grown **substantially** since the last generate (e.g. a meaningful percentage increase), **or** when clear evidence shows the current code has saturated.
• Generating new code at nearly the same dataset size almost always yields a similar MSE, so skip it unless you have a strong rationale — for example, the most-recent generation produced a **clearly worse** MSE.

Stopping rules
• If the current MSE ≤ {mse_threshold}, run one confirmation test (**test**).
• If that confirmation also meets the goal, halt (**done**).
• If both extra data and a new code generation appear unhelpful, halt (**done**).

Action choices
• **test**     Evaluate current code on a larger dataset (≈ +10–25 % or at least +300 samples, but never beyond {max_size}).
• **generate** Regenerate code using the current dataset size.
• **done**      Stop; further work is unlikely to help; If mse meets the target, stop.

Respond with **only** a JSON object with no code fences, back-ticks, or extra text. If you include back-ticks, the program will crash. The object has exactly these keys:

{{
  "dataset_size": <int>,
  "action": "<test|generate|done>",
  "reason": "<optional brief justification>"
}}
"""

    agent = Agent(
        name="LoopDirector",
        instructions=prompt,
        model=config.AGENT_MODEL,
        model_settings=ModelSettings(),
        tools=[]
    )
    resp = await Runner.run(agent, "Decide the next step.")
    out = resp.final_output.strip()
    if out.startswith("```"):
        out = re.sub(r"^```(?:json)?\s*|\s*```$", "", out, flags=re.DOTALL).strip()

    try:
        decision = json.loads(out)
    except json.JSONDecodeError:
        raise RuntimeError(f"Expected JSON but got:\n{out!r}")
    if not isinstance(decision, dict) or decision.get("action") not in ("test", "generate", "done"):
        raise RuntimeError(f"Unknown decision from the LoopDirector:\n{out!r}")
    return decision

def save_history(history: List[Dict[str, Any]], path: str = config.HISTORY_FILE):
    """Atomically overwrite the JSON file with the current history."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    os.replace(tmp, path)

def save_code(code: str, path: str = config.BEST_CODE_FILE):
    """Write the code currently in use, so that it survives a later failure of the loop."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

async def code_data_loop(
    initial_size: int,
    maxsize: int = config.MAX_DATASET_SIZE,
    mse_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Run the AIDE code-data loop until done or maxsize reached.
        Returns {"code", "size", "mse", "mse_size", "history"}: the code in use
        at the end (also saved to best_solution.py), the size of the dataset now
        in ./input, the validation MSE of that code and the dataset size that
        MSE was measured at.
        """
    if mse_threshold is None:
        mse_threshold = load_target_mse()
    history: List[Dict[str, Any]] = []

    def record(entry: Dict[str, Any]):
        history.append(entry)
        save_history(history)

    # 1) Initial generation
    sol = generate_solution(initial_size)
    if sol["mse"] is None:
        raise RuntimeError(
            "AIDE did not produce a working solution (no validation metric). "
            "See the AIDE logs in ./logs for the errors of the generated scripts."
        )
    code = sol["code"]
    current_mse = sol["mse"]
    current_mse_size = initial_size  # dataset size current_mse was measured at
    current_size = initial_size
    save_code(code)
    record({
        "size":   initial_size,
        "mse":    current_mse,
        "action": "generated",
        "reason": "New Generated"
    })

    for _ in range(config.MAX_LOOP_DECISIONS):
        if current_size >= maxsize:
            print(f"🛑 Ending loop: dataset size {current_size} reached the maximum {maxsize}")
            break

        # 2) Ask the LoopDirector agent what to do next
        decision = await decide_next_step(history, mse_threshold, maxsize)
        print("Decision:", decision)

        if decision["action"] == "done":
            print(f"✅ Done: reached MSE {current_mse}")
            break

        # never go back to a smaller dataset, never beyond the maximum
        requested = _to_float(decision.get("dataset_size"))
        next_size = int(requested) if requested else current_size
        next_size = max(current_size, min(next_size, maxsize))
        reason = decision.get("reason", "")

        if decision["action"] == "generate":
            generate_dataset(next_size)
            sol = generate_solution(next_size)
            print(f"🔄 Regenerated at size={next_size}, mse={sol['mse']}")
            # keep the new code only when it is better than the code in use
            better = sol["mse"] is not None and (current_mse is None or sol["mse"] < current_mse)
            if better:
                code = sol["code"]
                current_mse = sol["mse"]
                current_mse_size = next_size
                save_code(code)
            current_size = next_size
            record({
                "size":   next_size,
                "mse":    sol["mse"],
                "action": "generated",
                "reason": reason,
                "kept_code": "new" if better else "previous"
            })

        else:  # "test"
            run = evaluate_solution(code, next_size)
            print("Raw output:", run["output"])
            current_size = next_size
            mse, error = None, None
            if not run["ok"]:
                error = f"exit code {run['returncode']}: {last_line(run['output'])}"
            else:
                try:
                    mse = await parse_mse_async(run["output"])
                except ValueError as e:
                    error = str(e)
            if error is not None:
                print(f"⚠️ Test failed at size={next_size}: {error}")
                record({
                    "size":   next_size,
                    "mse":    None,
                    "action": "test_failed",
                    "reason": error
                })
                continue
            print(f"🔍 Tested at size={next_size}, mse={mse}")
            current_mse = mse
            current_mse_size = next_size
            record({
                "size":   next_size,
                "mse":    mse,
                "action": "tested",
                "reason": reason
            })
    else:
        print(f"🛑 Ending loop: {config.MAX_LOOP_DECISIONS} decisions reached")

    return {
        "code": code,
        "size": current_size,
        "mse": current_mse,
        "mse_size": current_mse_size,
        "history": history,
    }

if __name__ == "__main__":
    print("=== AIDE Loop ===")
    config.require_api_key()
    dataset_size = 500

    # seed ./input with the initial training and validation subsets
    generate_dataset(dataset_size)

    result = asyncio.run(code_data_loop(dataset_size, maxsize=config.MAX_DATASET_SIZE))
    print(f"Final size: {result['size']}, final MSE: {result['mse']} (measured at size {result['mse_size']})")
