# agent_system.py

import asyncio
import traceback
from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace, ModelSettings
from agents import run as agents_run

import config
from task_desp import generate_desp
from subagent import PathVerifier, ForwardTrainer, InverseDesigner, ask_human, agent_tool

# The OpenAI API key is read from the OPENAI_API_KEY environment variable, or from
# a `.env` file in this folder (see config.py). It is not stored in the code.

_orig_run_sync = agents_run.Runner.run_sync
def _guard_run_sync(*a, **k):
    try:
        asyncio.get_running_loop()  # raises if no running loop
    except RuntimeError:
        return _orig_run_sync(*a, **k)
    print("\n\n🚨 RUN_SYNC under running loop — stack below 🚨")
    traceback.print_stack(limit=500)
    raise RuntimeError("Nested event loop: Runner.run_sync called under running loop.")

agents_run.Runner.run_sync = _guard_run_sync

# ——— wrap each subagent and the spec‐writer as tools ———

verify_paths = PathVerifier.as_tool(
    tool_name="verify_paths",
    tool_description="Ensure required files exist for forward or inverse stages, prompting if missing"
)

# The two long-running stages report a failure once instead of being retried.
run_forward = agent_tool(
    ForwardTrainer,
    tool_name="run_forward",
    tool_description="Run the forward training pipeline given a dataset size"
)

run_inverse = agent_tool(
    InverseDesigner,
    tool_name="run_inverse",
    tool_description="Run the inverse design pipeline given the test data path"
)

# ——— the interactive Planner agent ———

planner_agent = Agent(
    name="planner_agent",
    instructions="""

You are the Planner agent for the metamaterials agents forward deep learning training and inverse design system.
Use only the provided tools to plan and execute the workflow.

ALWAYS get required info via ask_human(tool), never by plain text questions.

Steps:
1) Call ask_human("Hi there! I'm here to assist you with metamaterials deep learning tasks. Briefly: do you want to run Forward training, Inverse design, or Both? (reply: forward / inverse / both)") and capture the reply as `choice`.

2) If choice in {"forward","both"}:
   a) Call generate_desp() (no arguments). It interviews the user and saves the task description.
   b) Call verify_paths({"purpose":"forward"}). If it does not return status "ok", stop and report the missing files.
   c) Choose a small initial dataset size (e.g., 500). Then call run_forward with that integer.
   d) The forward code is saved to 'best_solution.py' by the tool. Do not print or copy the code.

3) If choice in {"inverse","both"}:
   a) Call ask_human("Please paste the TEST spectra CSV path for inverse design.")
   b) Call verify_paths({"purpose":"inverse"}). If it does not return status "ok", stop and report the missing files.
   c) Call run_inverse with that path.

4) If a tool result starts with "FAILED" or reports the status "failed", do not call that tool again: stop and report the failure.

5) Return a single merged dict of all tool outputs as your final answer.
""",
    tools=[ask_human, generate_desp, verify_paths, run_forward, run_inverse],
    model=config.AGENT_MODEL,
    model_settings=ModelSettings(
    )
)

# ——— optional synthesizer to pretty‐print ———

synthesizer_agent = Agent(
    name="synthesizer_agent",
    instructions="""
You receive the planner’s merged result. Produce a concise, human‐readable summary that includes:
- Which stages were run
- The dataset size used
- The final forward MSE (if any)
- The inverse‐design MSE (if any)
Report only values that appear in the result. If a stage failed, say so and give the error.
""",
)

# ——— entry point ———

async def main():
    # Kick off the planner with no initial info
    with trace("Planner execution"):
        planner_result = await Runner.run(planner_agent, "Start.", max_turns=30)

    # Print the planner's messages
    for item in planner_result.new_items:
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            if text:
                print(f"[Planner] {text}")

    # Hand off to the synthesizer for a friendly summary
    synth_input = planner_result.to_input_list()
    synth_result = await Runner.run(synthesizer_agent, synth_input)

    print("\n=== Final Summary ===")
    print(synth_result.final_output)

if __name__ == "__main__":
    config.require_api_key()
    asyncio.run(main())
    # Design and optimize a deep learning regression model to predict the electromagnetic spectrum from geometry parameters using supervised learning.
    # ./dataset/test_s_inverse_test_0.csv
