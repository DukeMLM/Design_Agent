# modifier.py
import asyncio
import os
import time
from typing import Any, Dict, Optional

from agents import Agent, Runner, ModelSettings

import config
from common import check_script_structure, last_line, run_script, strip_code_fences

_SYSTEM_PROMPT = f"""
Please refactor the entire AIDEML training script so that it meets **all** of the following requirements, **without** changing any of the existing behavior (model code, SWA logic, CSV export, console printing, etc.):

0. **Zero-arg model instantiation**
   - The `forward_model` class must support zero-argument construction (`model = forward_model()`) by giving all its `__init__` arguments sensible defaults as zero-arg call recreates **exactly** the architecture that produced
    model_final.pth during training.

1. **Class renaming**
   - Rename your network class to `forward_model` everywhere it appears so that in another file one can do:
     ```python
     from modified import forward_model
     model = forward_model().to(device)
     model.load_state_dict(torch.load("{config.MODEL_FILE}"))
     ```

2. **Input & output pipelines**
   - **Bundle every X-preprocessing step** (PolynomialFeatures, PCA, manual scaling, etc.) into a single scikit-learn `Pipeline` or scaler object named `x_scaler`.
   - **Bundle every Y-preprocessing step** (StandardScaler, PCA, normalization, etc.) into a single `Pipeline` or scaler object named `y_scaler`.
   - Replace any manual `mean/std` code with the equivalent `StandardScaler` step within these pipelines.
   - Both objects must provide `transform()` and `inverse_transform()`. The network takes `x_scaler.transform(geometry)` as input and its output is in the space of `y_scaler`.
   - If the original script applies no preprocessing to X (or to Y), set that object to the identity transform `sklearn.preprocessing.FunctionTransformer()` and fit it on the training data. Do not add any other preprocessing that was not in the original script.

3. **Saving artifacts**
   - **After** the main training loop (and after any SWA finalization), save all three files:
     - Model weights → `{config.MODEL_FILE}`
     - `x_scaler` → `{config.X_SCALER_FILE}`
     - `y_scaler` → `{config.Y_SCALER_FILE}`
   - Use this two-step pattern for each scaler:
     ```python
     import joblib
     try:
         joblib.dump(x_scaler, x_path)
     except Exception:
         # build an equivalent simple scaler/PCA wrapper
         joblib.dump(serializable_x_scaler, x_path)
     ```

4. **Final validation printout**
   - At the very end, after everything, console-print exactly:
     ```
     Final Validation MSE: <numeric_value>
     ```

5. **Importable module**
   - Keep the imports and the class and function definitions at module level.
   - Put everything else (reading data, training, saving, printing) inside `if __name__ == "__main__":`, so that `from modified import forward_model` only defines the class and does not start a training run.

**Keep** all existing imports, prints, CSV exports, and SWA code intact. Just insert/replace the preprocessing with your two pipelines, hook in the save logic, and add the final print.
────────────────────────────────────────────────────────────
Return only the modified runnable Python script — no explanations, no markdown, no extra text.
VERY IMPORTANT: Your entire response must be valid Python source code — directly executable with no editing.
• Do NOT include any explanations, headers, markdown fences, or extra commentary.
• Do NOT wrap your output inside ```python or ``` marks.
• Output ONLY the modified runnable Python script, starting directly from the first line of Python code.
"""

_ARTIFACTS = (config.MODEL_FILE, config.X_SCALER_FILE, config.Y_SCALER_FILE)


def _make_agent() -> Agent:
    ''' Create and return a CodeModifier agent.     '''
    return Agent(
        name="CodeModifier",
        instructions=_SYSTEM_PROMPT,
        model=config.AGENT_MODEL,
        model_settings=ModelSettings(),
        tools=[],
    )

def _run_code(code: str) -> dict:
    ''' Run the given code in a temporary file and capture output. '''
    return run_script(code, timeout=config.RUN_TIMEOUT_S)

def _check_run(result: Dict[str, Any], started_at: float) -> Optional[str]:
    '''
    Decide whether a run of the refactored script is usable by the inverse design.
    Returns a description of the problem (fed back to the agent), or None.
    '''
    if not result["ok"]:
        return result["stderr"] or f"The script exited with code {result['returncode']} (no stderr)."
    stale = [
        path for path in _ARTIFACTS
        if not os.path.exists(path) or os.path.getmtime(path) < started_at - 2
    ]
    if stale:
        return (
            "The script ran without error but did not save: " + ", ".join(stale)
            + ". All three files must be written after training (requirement 3)."
        )
    if "Final Validation MSE:" not in result["stdout"]:
        return "The script ran without error but did not print the line 'Final Validation MSE: <numeric_value>'."
    return None

async def modify_and_verify(py_file_path: str, max_rounds: int = 4) -> Dict[str, Any]:
    '''
    Modify the code in `py_file_path` until it runs without error and saves the
    model and both scalers.
    Returns {"ok", "code", "rounds", "error", "stdout"}; "ok" is False when no
    round produced a usable script ("code" is then the last attempt).
    '''
    agent = _make_agent()
    with open(py_file_path, 'r', encoding="utf-8") as f:
        raw_code = f.read()
    user_input = raw_code
    patched = None
    problem = "no round was run"
    stdout = ""

    for round_idx in range(1, max_rounds + 1):
        resp  = await Runner.run(agent, user_input)
        patched = strip_code_fences(resp.final_output)

        # cheap checks first: a script that cannot be imported is not worth a training run
        problem = check_script_structure(patched)
        if problem is None:
            print("RUN")
            started_at = time.time()
            result = _run_code(patched)
            stdout = result["stdout"]
            problem = _check_run(result, started_at)

        if problem is None:
            print(f"✅ Round {round_idx}: script ran successfully.")
            return {"ok": True, "code": patched, "rounds": round_idx, "error": None, "stdout": stdout}

        print(f"⚠️  Round {round_idx} failed – feeding the problem back to agent.")
        print(f"{problem}\n")
        user_input = (
            "The previous patch still fails. Here is the code followed by "
            "the full traceback. Please fix all errors and return an updated "
            "runnable script.\n\n"
            "```python\n" + patched + "\n```\n\n"
            "Traceback:\n```\n" + problem + "\n```"
        )
        agent = _make_agent()

    return {"ok": False, "code": patched, "rounds": max_rounds, "error": last_line(problem, problem), "stdout": stdout}

if __name__ == "__main__":
    config.require_api_key()
    input_path = config.BEST_CODE_FILE
    out_path = config.MODIFIED_CODE_FILE
    outcome = asyncio.run(modify_and_verify(input_path))
    if not outcome["ok"]:
        raise SystemExit(f"No working script after {outcome['rounds']} rounds: {outcome['error']}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(outcome["code"])
    print(f"Final file written to  {out_path}")
