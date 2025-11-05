# main.py

import asyncio
from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace, ModelSettings
from task_desp import generate_desp
from subagent import PathVerifier, ForwardTrainer, InverseDesigner, ask_human
import os
import openai

# Load from env or set default
api_key =""
openai.api_key = api_key
os.environ['OPENAI_API_KEY'] = ""

import asyncio, traceback
from agents import run as agents_run

_orig_run_sync = agents_run.Runner.run_sync
def _guard_run_sync(*a, **k):
    try:
        asyncio.get_running_loop()  # raises if no running loop
        print("\n\n🚨 RUN_SYNC under running loop — stack below 🚨")
        traceback.print_stack(limit=500)
        raise RuntimeError("Nested event loop: Runner.run_sync called under running loop.")
    except RuntimeError:
        pass
    return _orig_run_sync(*a, **k)

agents_run.Runner.run_sync = _guard_run_sync

# ——— wrap each subagent and the spec‐writer as tools ———

# generate_desp = TASK_AGENT.as_tool(
#     tool_name="generate_desp",
#     tool_description="Generate and save the forward task description to 'desp_flex.txt'"
# )

verify_paths = PathVerifier.as_tool(
    tool_name="verify_paths",
    tool_description="Ensure required files exist for forward or inverse stages, prompting if missing"
)

run_forward = ForwardTrainer.as_tool(
    tool_name="run_forward",
    tool_description="Run the forward training pipeline given a dataset size"
)

run_inverse = InverseDesigner.as_tool(
    tool_name="run_inverse",
    tool_description="Run the inverse design pipeline given code path and test data path"
)

# ask = ask_human.as_tool(
#     tool_name="ask_human",
#     tool_description="Prompt the user with a question and return their response"
# )

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
   a) Call generate_desp with a short summary you infer from the user's earlier task.
   b) Call verify_paths({"purpose":"forward"}).
   c) Choose a small initial dataset size (e.g., 500). Then call run_forward with that integer.
   d) Save the returned "code" to 'code_aideml.py'.

3) If choice in {"inverse","both"}:
   a) Call ask_human("Please paste the TEST spectra CSV path for inverse design.")
   b) Call verify_paths({"purpose":"inverse"}).
   c) Call run_inverse with that path.

4) Return a single merged dict of all tool outputs as your final answer.
""",
    tools=[ask_human, generate_desp, verify_paths, run_forward, run_inverse],
    model="o4-mini",
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
""",
)

# ——— entry point ———

async def main():
    # Kick off the planner with no initial info
    with trace("Planner execution"):
        planner_result = await Runner.run(planner_agent, "")

    # Print each tool’s raw output as it arrives
    for item in planner_result.new_items:
        if isinstance(item, MessageOutputItem):
            text = ItemHelpers.text_message_output(item)
            if text:
                print(f"[Tool] {text}")

    # Hand off to the synthesizer for a friendly summary
    synth_input = planner_result.to_input_list()
    synth_result = await Runner.run(synthesizer_agent, synth_input)

    print("\n=== Final Summary ===")
    print(synth_result.final_output)

if __name__ == "__main__":
    asyncio.run(main())
    # Design and optimize a deep learning regression model to predict the electromagnetic spectrum from geometry parameters using supervised learning.
    # /home/dl370/agent/dataset/test_s_inverse_test_0.csv