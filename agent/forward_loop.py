import aide
from agents import tool, Agent, Runner, ModelSettings  
from typing import Any, Dict, List    
import os
import json
import tempfile
import subprocess
import re
import tempfile
import subprocess
import pandas as pd
import sys
from sklearn.utils import shuffle
import tempfile, subprocess, re, pandas as pd
from sklearn.utils import shuffle
          # OpenAI Agents SDK
from agents.tool import function_tool as tool
from agents import Agent, Runner, ModelSettings
import os
import openai
# Load from env or set default
api_key =""
openai.api_key = api_key
os.environ['OPENAI_API_KEY'] = ""

def generate_solution(dataset_size: int) -> Dict[str, Any]:
    """
    Spins up an AIDE experiment, reading the goal prompt from a text file
    and auto‑injecting the current dataset size.
    Returns {"mse": float, "code": str}.
    """
    # 1) Read the base goal description from a .txt file
    goal_txt_path = f"./desp_flex.txt"  # point to your prompt template
    with open(goal_txt_path, "r") as f:
        base_goal = f.read()

    # 2) Append a line about the current dataset size
    goal = (
        base_goal.strip()
        + f"\n\nCurrent dataset size: {dataset_size} samples"
    )

    exp = aide.Experiment(
        data_dir=f"./input",
        goal=goal,
        eval="Use Mean Squared Error (MSE) between predicted and true spectra"
    )

    best = exp.run(steps=50)
    print(best.valid_metric)
    return {"mse": best.valid_metric, "code": best.code}

def fine_tune_hp(dataset_size: int, code:str) -> Dict[str, Any]:
    """
    Not used currently
    """
    # 1) Read the base goal description from a .txt file
    goal_txt_path = f"./desp_flex.txt"  # point to your prompt template
    with open(goal_txt_path, "r") as f:
        base_goal = f.read()

    # 2) Append a line about the current dataset size
    goal = (
        base_goal.strip()
        + f"\n\nCurrent dataset size: {dataset_size} samples"
        + f"\n You should only use the following code to fine-tune the hyperparameters of the model."
        + f"\n\nCurrent code: {code}"
    )

    exp = aide.Experiment(
        data_dir=f"./input",
        goal=goal,
        eval="Use Mean Squared Error (MSE) between predicted and true spectra"
    )

    best = exp.run(steps=50)
    print(best.valid_metric)
    return {"mse": best.valid_metric, "code": best.code}

def generate_dataset(dataset_size: int) -> None:
    """
    Regenerate the training and validation CSVs in ./input/
    with the given dataset_size (by subsampling from the full dataset). 
    This is to simulate generating new data as mentioned in the paper
    """
    # --- 1) Regenerate the smaller training split in dataset2 ---
    # Here using the down sampling approach to simulate the real generation process
    X_full = pd.read_csv("./dataset/g_training.csv", header=0).values
    y_full = pd.read_csv("./dataset/s_training.csv", header=0).values
    X_shuf, y_shuf = shuffle(X_full, y_full, random_state=1)
    X_small, y_small = X_shuf[:dataset_size], y_shuf[:dataset_size]
    pd.DataFrame(X_small).to_csv(
        f"./input/g_training.csv", index=False
    )
    pd.DataFrame(y_small).to_csv(
        f"./input/s_training.csv", index=False
    )
    Xv_full = pd.read_csv("./dataset/g_validation.csv", header=0).values
    yv_full = pd.read_csv("./dataset/s_validation.csv", header=0).values
    Xv_shuf, yv_shuf = shuffle(Xv_full, yv_full, random_state=1)
    Xv_small, yv_small = Xv_shuf[:int(0.1*dataset_size)], yv_shuf[:int(0.1*dataset_size)]
    pd.DataFrame(Xv_small).to_csv(
        f"./input/g_validation.csv", index=False
    )
    pd.DataFrame(yv_small).to_csv(
        f"./input/s_validation.csv", index=False
    )
    print(f"Generated {dataset_size} samples in ./input/")

def evaluate_solution(code: str, dataset_size: int) -> float:
    """
    1) Subsample with `dataset_size` samples,
       writing CSVs with a header row so the generated code can read them.
    2) Dump the AIDE-generated `code` to a temp .py file.
    3) Run it and parse out the printed Validation MSE.
    Returns that MSE as a float, or raises with full logs if the script errors.
    """

    # --- 1) Regenerate the smaller training split in dataset2 ---
    generate_dataset(dataset_size)

    # --- 2) Write the AIDE code to a temp file ---
    tf = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    tf.write(code)
    tf.close()

    # --- 3) Execute without check so we capture everything ---
    proc = subprocess.run(
        [sys.executable, tf.name],
        capture_output=True,
        text=True,
        check=False
    )

    # --- 4) Return combined stdout+stderr for downstream parsing ---
    output = proc.stdout
    if proc.stderr:
        output += "\n=== STDERR ===\n" + proc.stderr
    return output


_VALIDATION_MSE_RE = re.compile(
    r"(?i)\bval(?:idation)?\s*mse\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

_NUMERIC_RE = re.compile(r"^\s*[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s*$")

# def parse_mse(
#     raw_output: str,
#     use_llm_fallback: bool = True,
# ) -> float:
#     """
#     Return the (last) Validation MSE as float.
#     1) Regex first (robust to case and sci notation).
#     2) Optional LLM fallback that must return only a numeric string.

#     Raises ValueError if not found or not a valid float.
#     """
#     # 1) regex path
#     matches = _VALIDATION_MSE_RE.findall(raw_output)
#     if matches:
#         val = float(matches[-1])  # last occurrence
#         return val

#     if not use_llm_fallback:
#         raise ValueError("Validation MSE not found in output.")

#     # 2) LLM fallback (force numeric-only response)
#     prompt = (
#     "I ran a training script and got the following output:\n\n"
#     f"{raw_output}\n\n"
#     "Extract the **single numeric Validation MSE** value from the text. The text may not be perfectly formatted, so be robust to minor typos (e.g., 'va\\lidation MSE').\n"
#     "If it’s not present, return only `null`.\n"
#     "Respond with the number **only**, parsable as a float. Do **not** include any text or explanation."
#     )

#     agent = Agent(name="MSEExtractor", instructions=prompt, tools=[])
#     resp = Runner.run_sync(agent, raw_output)
#     val_str = resp.final_output.strip()
#     # -------------------------------------------------------------------------------

#     if val_str.lower() == "null":
#         raise ValueError("Validation MSE not found in output (LLM fallback returned null).")

#     # Enforce numeric-only output
#     if not _NUMERIC_RE.fullmatch(val_str):
#         raise ValueError(f"LLM fallback returned non-numeric text: {val_str!r}")

#     val = float(val_str)
#     return val

async def parse_mse_async(raw_output: str) -> str:
    """
    Async version of parse_mse that returns the MSE as a string.
    1) Regex first (robust to case and sci notation).
    2) LLM fallback that must return only a numeric string.
    Raises ValueError if not found or not a valid float.
    """
    matches = _VALIDATION_MSE_RE.findall(raw_output)
    if matches:
        if float(matches[-1]) > 0:
            return str(matches[-1])

    # LLM fallback via async Runner.run
    prompt = (
    "I ran a training script and got the following output:\n\n"
    f"{raw_output}\n\n"
    "Extract the **single numeric Validation MSE** value from the text. The text may not be perfectly formatted, so be robust to minor typos (e.g., 'va\\lidation MSE').\n"
    "If it’s not present, return only `null`.\n"
    "Respond with the number **only**, parsable as a float. Do **not** include any text or explanation."
    )
    agent = Agent(name="MSEExtractor", instructions=prompt, tools=[])
    resp = await Runner.run(agent, raw_output)
    val_str = resp.final_output.strip()
    if val_str.lower() == "null":
        raise ValueError("Validation MSE not found in output (LLM fallback returned null).")
    if not _NUMERIC_RE.fullmatch(val_str):
        raise ValueError(f"LLM fallback returned non-numeric text: {val_str!r}")
    return str(val_str)

async def last_nonempty_line(text: str) -> str:
    """
    Return the last non-empty line from the given text.
    If none found, fall back to parsing MSE via LLM.
    """
    # Fast + robust to trailing newlines/spaces
    
    for line in reversed(text.splitlines()):
        if line.strip():
            return line
    return await parse_mse_async(text)

async def decide_next_step(
    history: List[Dict[str, Any]],
    mse_threshold: float = 0.002,
    max_size: int = 41000
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
1. Reach a validation MSE ≤ {mse_threshold}. **before** the dataset exceeds {max_size}. samples.  
2. Minimize both (a) the number of extra samples and (b) the number of AIDE code-generation calls.

History 
{json.dumps(history)}

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
• If the current MSE ≤ <<mse_threshold>>, run one confirmation test (**test**).  
• If that confirmation also meets the goal, halt (**done**).  
• If both extra data and a new code generation appear unhelpful, halt (**done**).

Action choices  
• **test**     Evaluate current code on a larger dataset (≈ +10–25 % or at least +300 samples, but never beyond <<max_size>>).  
• **generate** Regenerate code using the current dataset size.  
• **done**      Stop; further work is unlikely to help; If mse meets the target, stop.

Respond with **only** a JSON object with no code fences, back-ticks, or extra text. If you include back-ticks, the program will crash

```json
{{
  "dataset_size": <int>, 
  "action": "<test|generate|done>",
  "reason": "<optional brief justification>"
}}
"""

    agent = Agent(
        name="LoopDirector",
        instructions=prompt,
        model="o4-mini",
        model_settings=ModelSettings(),
        tools=[]
    )
    resp = await Runner.run(agent, "")
    out = resp.final_output.strip()
    if out.startswith("```"):
        out = re.sub(r"^```(?:json)?\s*|\s*```$", "", out, flags=re.DOTALL).strip()

    try:
        return json.loads(out)
    except json.JSONDecodeError:
        raise RuntimeError(f"Expected JSON but got:\n{out!r}")

def save_history(history: List[Dict[str, Any]], path: str = "run_history.json"):
    """Atomically overwrite the JSON file with the current history."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    os.replace(tmp, path)

async def code_data_loop(initial_size: int, maxsize: int) -> List[Dict[str, Any]]:
    """Run the AIDE code-data loop until done or maxsize reached.
        Returns the final code and dataset size used.
        """ 
    history: List[Dict[str, Any]] = []
    # 1) Initial generation
    sol = generate_solution(initial_size)
    code = sol["code"]
    mse = sol["mse"]
    history.append({
        "size":   initial_size,
        "mse":    mse,
        "action": "generated",
        "reason": "New Generated"
    })



    def record(entry: Dict[str, Any]):
        history.append(entry)
        save_history(history)

    while True:
        # 2) Ask the LoopDirector agent what to do next
        decision = await decide_next_step(history)
        print("Decision:", decision)
        data_temp = pd.read_csv(f"./input/g_training.csv", header=0).values
        save_history(history)

        if decision["action"] == "done":
            print(f"✅ Done: reached MSE {history[-1]['mse']}")
            break

        if data_temp.shape[0] >= maxsize:
            print(f"🛑 Ending loop: dataset size {data_temp.shape[0]} is too large")
            break
        previous_size = history[-1]["size"]
        previous_mse = history[-1]["mse"]
        next_size = decision["dataset_size"]
        reason = decision["reason"]

        if decision["action"] == "generate":
            generate_dataset(next_size)
            # print("dataset_size", data_temp.shape[0])
            sol = generate_solution(next_size)
            print(f"🔄 Regenerated at size={next_size}, mse={sol['mse']}")
            history.append({
                "size":   next_size,
                "mse":    sol["mse"],
                "action": "generated",
                "reason": reason
            })
            code = sol["code"]

        elif decision["action"] == "test":
            raw = evaluate_solution(code, next_size)
            # print("dataset_size", data_temp.shape[0])
            print("Raw output:", raw)
            mse = await parse_mse_async(raw)
            print(f"🔍 Tested at size={next_size}, mse={se}")
            history.append({
                "size":   next_size,
                "mse":     mse,
                "action": "tested",
                "reason": reason
            })

        else:
            raise RuntimeError(f"Unknown action: {decision['action']}")


    return code, next_size

if __name__ == "__main__":
    print("=== AIDE Loop ===")
    dataset_size = 500

    # seed ./input with the initial subset
    X_full = pd.read_csv("./dataset/g_training.csv", header=0).values
    y_full = pd.read_csv("./dataset/s_training.csv", header=0).values
    X_shuf, y_shuf = shuffle(X_full, y_full, random_state=1)
    X_small, y_small = X_shuf[:dataset_size], y_shuf[:dataset_size]
    pd.DataFrame(X_small).to_csv("./input/g_training.csv", index=False)
    pd.DataFrame(y_small).to_csv("./input/s_training.csv", index=False)

    import asyncio
    asyncio.run(code_data_loop(dataset_size, maxsize=41000))