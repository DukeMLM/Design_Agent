# subagent.py

from agents import Agent, ItemHelpers, Runner, ModelSettings
from agents.tool import function_tool
from typing import Dict, Any, List
import os
import asyncio

import config

''' Subagents for various tasks related to code modification and verification. '''


def fail_no_retry(ctx, error: Exception) -> str:
    """
    Error message for the long-running tools. The SDK default ends with
    "Please try again", which makes the calling agent restart hours of work.
    """
    return (
        f"FAILED: {type(error).__name__}: {error}\n"
        "This step was not completed. Do not call this tool again; report the failure."
    )


def agent_tool(agent: Agent, tool_name: str, tool_description: str, max_turns: int = 20):
    """Like Agent.as_tool(), but a failure is reported once instead of being retried."""

    @function_tool(
        name_override=tool_name,
        description_override=tool_description,
        failure_error_function=fail_no_retry,
    )
    async def run_agent(input: str) -> str:
        result = await Runner.run(agent, input, max_turns=max_turns)
        return ItemHelpers.text_message_outputs(result.new_items)

    return run_agent


# ------------------ 1. Verifier ------------------
_SPLIT_FILES = ("g_training.csv", "s_training.csv", "g_validation.csv", "s_validation.csv")
_VERIFIED_FILES = (
    config.MODIFIED_CODE_FILE,
    config.MODEL_FILE,
    config.X_SCALER_FILE,
    config.Y_SCALER_FILE,
)


def verified_model_is_current() -> bool:
    """True when modified.py and its model files exist and are not older than best_solution.py."""
    if not all(os.path.exists(path) for path in _VERIFIED_FILES):
        return False
    if not os.path.exists(config.BEST_CODE_FILE):
        return True
    return os.path.getmtime(config.MODIFIED_CODE_FILE) >= os.path.getmtime(config.BEST_CODE_FILE)


def missing_files(purpose: str) -> List[str]:
    """Files the given stage ('forward' or 'inverse') needs and that do not exist."""
    purpose = purpose.strip().lower()
    if purpose == "forward":
        needed = [config.TASK_DESC_FILE]
        needed += [os.path.join(config.DATASET_DIR, name) for name in _SPLIT_FILES]
    elif purpose == "inverse":
        if verified_model_is_current():
            # only the training geometry is needed (bounds of the design space)
            geometry = [os.path.join(d, "g_training.csv") for d in (config.INPUT_DIR, config.DATASET_DIR)]
            needed = [] if any(os.path.exists(p) for p in geometry) else [geometry[0]]
        else:
            # the training script is refactored and retrained on the data in ./input
            needed = [config.BEST_CODE_FILE]
            needed += [os.path.join(config.INPUT_DIR, name) for name in _SPLIT_FILES]
    else:
        raise ValueError(f"purpose must be 'forward' or 'inverse', got {purpose!r}")
    return [path for path in needed if not os.path.exists(path)]


@function_tool
async def check_stage_files(purpose: str) -> Dict[str, Any]:
    """
    Check that every file a stage needs exists.

    Args:
        purpose: "forward" or "inverse".
    """
    missing = missing_files(purpose)
    result: Dict[str, Any] = {"ok": not missing, "missing": missing}
    if missing and purpose.strip().lower() == "inverse":
        result["hint"] = (
            "best_solution.py and the files in ./input are written by the forward stage. "
            "Run the forward stage first, or provide a modified.py that matches model_final.pth."
        )
    if missing and purpose.strip().lower() == "forward":
        result["hint"] = (
            "desp_flex.txt is written by generate_desp. The dataset files are created by "
            "extracting ADM.zip into ./dataset and running `python prepare_dataset.py`."
        )
    return result

@function_tool
async def verify_path(path: str) -> Dict[str, Any]:
    """
    Generic checker: verify a single filesystem path exists.
    """
    if not os.path.exists(path):
        return {"ok": False, "missing": path}
    return {"ok": True}
@function_tool
async def ask_human(prompt: str) -> Dict[str, str]:
    """Display *prompt* in the console and return the user’s reply."""
    return {"human_input": await asyncio.to_thread(input, f"\n👤 {prompt}\n> ")}

PathVerifier = Agent(
    name="PathVerifier",
    instructions="""
You are a helper that ensures the files a stage needs are in place.

1. You will be called with a single argument: {"purpose": "<forward|inverse>"}.
2. Call check_stage_files(purpose).
3. If it returns ok=True, return {"status":"ok"}.
4. If it returns ok=False, use ask_human(...) once to tell the user which files are
   missing (include the hint if there is one) and to reply "done" when the files are
   in place, or "stop" to give up.
   - If the user replies "done", call check_stage_files(purpose) again and repeat from step 3.
   - If the user replies "stop", return {"status":"missing","missing":[<the missing files>]}.
""",
    tools=[check_stage_files, verify_path, ask_human],
    model=config.AGENT_MODEL,
    model_settings=   ModelSettings(
    )
)
# ------------------ 2. Forward Trainer Subagent ------------------

@function_tool
async def generate_datasets(dataset_size: int) -> Dict[str, Any]:
    """Write the training and validation CSV files of the given size to ./input."""
    from forward_loop import generate_dataset
    written = generate_dataset(dataset_size)
    return {"status": f"datasets {written} generated"}

@function_tool(failure_error_function=fail_no_retry)
async def code_data_loops(dataset_size: int) -> dict:
    """
    Run the AIDE code generation and dataset-size loop, starting at dataset_size.
    The final code is saved to best_solution.py. This takes hours.
    """
    from forward_loop import code_data_loop, save_code
    result = await code_data_loop(dataset_size, maxsize=config.MAX_DATASET_SIZE)
    # save the code
    save_code(result["code"])
    return {
        "code_path": "best_solution.py",
        "final_size": result["size"],
        "final_mse": result["mse"],
        "conclusion": f"Completed loop at size={result['size']} with validation MSE={result['mse']}"
    }

ForwardTrainer = Agent(
    name="ForwardTrainer",
    instructions="""
You are the Forward Trainer agent. You will be given a single integer argument: `dataset_size`.
1. Call `generate_datasets(dataset_size)` to write out the training/validation CSV files with inital dataset size.
2. After it has returned, invoke `code_data_loops(dataset_size)` to run the AIDE and dataset size optimize loop.
   It saves the final Python source to best_solution.py and returns a dict with
   `code_path`, `final_size`, `final_mse` and `conclusion`.
3. If a tool result starts with "FAILED", do not call the tool again. Return {"status": "failed", "error": <the message>}.
4. Otherwise return exactly the dict returned by `code_data_loops`:
   {
     "code_path": "best_solution.py",
     "final_size": <the final dataset size integer>,
     "final_mse": <the final validation MSE>,
     "conclusion": <the summary of the process>,
   }
    """,
    tools=[code_data_loops, generate_datasets],
    model=config.AGENT_MODEL,
    model_settings=   ModelSettings(
    )
)

# ------------------ 3. Inverse Designer Subagent ------------------

async def run_code_verify() -> str:
    """
    Refactor best_solution.py into modified.py (class `forward_model`, saved model
    and scalers) and check that it runs. Raises RuntimeError when no round works.
    """
    from modifier import modify_and_verify

    if verified_model_is_current():
        return "already verified: modified.py and its model files are up to date"

    outcome = await modify_and_verify(config.BEST_CODE_FILE)
    if not outcome["ok"]:
        raise RuntimeError(
            f"the training script could not be verified in {outcome['rounds']} rounds: {outcome['error']}"
        )
    with open(config.MODIFIED_CODE_FILE, "w", encoding="utf-8") as f:
        f.write(outcome["code"])

    return f"fixed, Final file written to  modified.py"

@function_tool(failure_error_function=fail_no_retry)
async def code_verify() -> str:
    """Patch and verify the forward training script; writes modified.py, model_final.pth and both scalers."""
    return await run_code_verify()

@function_tool(failure_error_function=fail_no_retry)
async def inverse_design(
    test_data_path: str,
) -> dict:
    """
    Calls your Neural Adjoint inverse design pipeline and returns
    {"Surrogate MSE": float, "geometry": [float, ...]}.
    """
    from inverse_NA import inverse_one
    return inverse_one(
        test_data_path
    )
InverseDesigner = Agent(
    name="InverseDesigner",
    instructions="""
You are the Inverse Designer agent. You will be given one argument:
- `test_data_path`: the path to the target spectrum file (CSV).

Your steps:
1. Call `code_verify()` to patch and verify the Python source.
2. If its result starts with "FAILED", stop: return {"status": "failed", "error": <the message>}.
3. After verifying, call `inverse_design(test_data_path)` which runs the Neural Adjoint pipeline.
4. `inverse_design` returns a dict with the keys "Surrogate MSE" and "geometry"
   (and "Real Simulation MSE" only when the real simulation is enabled).
5. Return exactly the dict returned by `inverse_design`. Do not add keys or values that the tool did not return.
""",
    tools=[code_verify, inverse_design],
    model=config.AGENT_MODEL,
    model_settings=   ModelSettings(
    )
)
# ------------------ 4. test ------------------
if __name__ == "__main__":
    config.require_api_key()
    print(asyncio.run(run_code_verify()))
