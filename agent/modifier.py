import os, sys, subprocess, tempfile, pathlib
from agents import Agent, Runner, ModelSettings
# Load from env or set default
import openai
api_key =""
openai.api_key = api_key
os.environ['OPENAI_API_KEY'] = ""

file_path = "./"
_SYSTEM_PROMPT = f"""
Please refactor the entire AIDEML training script so that it meets **all** of the following requirements, **without** changing any of the existing behavior (model code, SWA logic, CSV export, console printing, etc.):

0. **Zero-arg model instantiation**  
   - The `forward_model` class must support zero-argument construction (`model = forward_model()`) by giving all its `__init__` arguments sensible defaults as zero-arg call recreates **exactly** the architecture that produced
    model_final.pth during training.

1. **Class renaming**  
   - Rename your network class to `forward_model` everywhere it appears so that in another file one can do:  
     ```python
     from test import forward_model
     model = forward_model().to(device)
     model.load_state_dict(torch.load("{file_path}/model_final.pth"))
     ```

2. **Input & output pipelines**  
   - **Bundle every X-preprocessing step** (PolynomialFeatures, PCA, manual scaling, etc.) into a single scikit-learn `Pipeline` or scaler object named `x_scaler`.  
   - **Bundle every Y-preprocessing step** (StandardScaler, PCA, normalization, etc.) into a single `Pipeline` or scaler object named `y_scaler`.  
   - Replace any manual `mean/std` code with the equivalent `StandardScaler` step within these pipelines.

3. **Saving artifacts**  
   - **After** the main training loop (and after any SWA finalization), save:  
     - Model weights → `{file_path}/model_final.pth`  
     - If `x_scaler` exists → `{file_path}/x_scaler.save`  
     - If `y_scaler` exists → `{file_path}/y_scaler.save`  
   - Use this two-step pattern for each scaler:
     ```python
     import joblib
     try:
         joblib.dump(x_scaler, x_path)
     except Exception:
         # build an equivalent simple scaler/PCA wrapper
         joblib.dump(serializable_x_scaler, x_path)
     ```
   - *Do not* invent new scalers if none were in the original script—only save the ones you build above.

4. **Final validation printout**  
   - At the very end, after everything, console-print exactly:
     ```
     Final Validation MSE: <numeric_value>
     ```
**Keep** all existing imports, prints, CSV exports, and SWA code intact. Just insert/replace the preprocessing with your two pipelines, hook in the save logic, and add the final print.
────────────────────────────────────────────────────────────
Return only the modified runnable Python script — no explanations, no markdown, no extra text.
VERY IMPORTANT: Your entire response must be valid Python source code — directly executable with no editing.  
• Do NOT include any explanations, headers, markdown fences, or extra commentary.  
• Do NOT wrap your output inside ```python or ``` marks.  
• Output ONLY the modified runnable Python script, starting directly from the first line of Python code.
"""

def _make_agent() -> Agent:
    ''' Create and return a CodeModifier agent.     '''
    return Agent(
        name="CodeModifier",
        instructions=_SYSTEM_PROMPT,
        model="o4-mini",
        model_settings=ModelSettings(),
        tools=[],
    )

def _run_code(code: str) -> dict:
    ''' Run the given code in a temporary file and capture output. '''
    tf = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tf.write(code); tf.close()
    proc = subprocess.run(
        [sys.executable, tf.name],
        capture_output=True,
        text=True,
    )
    return {
        "ok": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "path": tf.name,
    }

async def modify_and_verify(py_file_path: str, max_rounds: int = 4) -> str:
    ''' Modify the code in `py_file_path` until it runs without error. '''
    agent = _make_agent()
    with open(py_file_path, 'r') as f:
        raw_code = f.read()
    user_input = raw_code
    patched = None

    for round_idx in range(1, max_rounds + 1):
        resp  = await Runner.run(agent, user_input)
        patched = resp.final_output
        print("RUN")
        result = _run_code(patched)

        if result["ok"]:
            print(f"✅ Round {round_idx}: script ran successfully.")
            break

        traceback = result["stderr"] or "(no stderr)"
        print(f"⚠️  Round {round_idx} failed – feeding traceback back to agent.")
        print(f"{traceback}\n")
        user_input = (
            "The previous patch still fails. Here is the code followed by "
            "the full traceback. Please fix all errors and return an updated "
            "runnable script.\n\n"
            "```python\n" + patched + "\n```\n\n"
            "Traceback:\n```\n" + traceback + "\n```"
        )
        agent = _make_agent()

    return patched

if __name__ == "__main__":
    input_path = "./best_solution.py"
    out_path = "./modified.py"
    fixed_code = modify_and_verify(input_path)
    with open(out_path, "w") as f:
        f.write(fixed_code)
    print(f"Final file written to  {out_path}")
