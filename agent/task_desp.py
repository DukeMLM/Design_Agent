# task_desp.py
import asyncio
import json
from agents import Agent, Runner, function_tool, ModelSettings

import config

''' AIDEML Task Description Writer Agent'''



@function_tool
def ask_human(prompt: str) -> str:
    """Display *prompt* in the console and return the user’s reply."""
    return input(f"\n👤 {prompt}\n> ")

@function_tool
def write_txt(content: str, target_mse: float) -> str:
    """
    Write *content* into 'desp_flex.txt' (UTF-8) and save the target validation MSE.

    Args:
        content: the full Markdown task description.
        target_mse: the target validation MSE stated by the user, as a number
            (for example 0.002). Use 0 if the user did not state a target.
    """
    with open(config.TASK_DESC_FILE, "w", encoding="utf-8") as f:
      f.write(content)
    # read by the forward loop (forward_loop.load_target_mse)
    task_config = {"target_mse": target_mse if target_mse and target_mse > 0 else None}
    with open(config.TASK_CONFIG_FILE, "w", encoding="utf-8") as f:
      json.dump(task_config, f, indent=2)
    return f"📄  Saved to desp_flex.txt"

# ---------- the agent ---------------------------------------------------------
TASK_AGENT = Agent(
    name="SpecWriter",
    instructions="""
You will generate a four-section Markdown specification and save it via write_txt(). Follow these rules exactly:

1. **Always** use the function tool ask_human(...) to gather any user input.
   Never ask questions via plain text.

2. **Step 1**: Call ask_human("Please describe your task and the dataset you have, in one or two paragraphs.")
   Let that response be the raw user input.

3. **Step 2**: Inspect the user input.
   If it does not clearly include:
     - A concise task overview, and model input/output dimensions.
     - Anything else you need to know.
   then call ask_human again with something like:
     ask_human("Could you clarify the model’s input/output dimensions?")
   Repeat until you have both pieces.

4. **Step 3**: Compose the final Markdown with these four sections (in your own words, paraphrasing the user):

   ## Task Overview
   <refined paraphrase of the user’s task>

   ## Dataset
    - Training set
      - g_training.csv: Geometry
      - s_training.csv: Spectrum

   - Validation set
      - g_validation.csv: Geometry
      - s_validation.csv: Spectrum

   - Test set (if available)
      - test_g.csv: Geometry
      - test_s.csv: Spectrum

   ## Model Specifications
   - Input dimension: <based on the user’s input>
   - Output dimension: <based on the user’s input>
   - Problem type: <based on the user’s input>
   - Loss Function: MSE
   - Evaluation Metric: MSE
   - Objective: <based on the user’s input>

   ## Instructions to AIDE
   1. Automatically explore and train deep neural networks to predict the <output dimension>D spectrum from <input dimension>D geometry inputs (use the dimensions given by the user).

   2. Use only scikit-learn preprocessing methods for data transformation. Only include transformers that support inverse_transform() so that predictions can be consistently mapped back to the original space.
   For example: Allowed: StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, PCA, etc. Avoid: PolynomialFeatures, OneHotEncoder, or any custom function without an inverse_transform method.

   3. Conduct architecture search to optimize performance. While fully connected (MLP) models are expected, you
   may also explore alternative architectures such as:
      - 1D Convolutional Neural Networks (CNNs)
      - Residual MLPs or ResNet-style architectures
      - Transformer-based blocks
      …if you determine they may improve performance for this task. Try not to introduce ensemble learning, bagging, stochastic weight averaging,
   EMA, or any multi-model averaging techniques.

   4. Perform hyperparameter tuning (e.g., learning rate, batch size, training steps) after architecture tuning.

   5. Report:
      - Final architecture and model type
      - Validation and test MSE
      - Any intermediate tuning results

5. **Step 4**: When your draft is complete, call the tool write_txt() exactly (Important!, you must call this tool):
   write_txt(<the full markdown string you generated>, <target_mse>)
   where <target_mse> is the target validation MSE the user stated, as a number (for example 0.002),
   or 0 if the user did not state one.
""", model_settings=ModelSettings(temperature=0), tools=[ask_human, write_txt], )


async def run_spec_writer() -> str:
    """Interview the user and write the task description; returns the agent's final message."""
    print("📝  Let's create a task description!")
    result = await Runner.run(TASK_AGENT, "Start.", max_turns=30)
    return str(result.final_output)

@function_tool
async def generate_desp() -> str:
    """Interview the user about the task and save the forward task description to 'desp_flex.txt'."""
    return await run_spec_writer()

if __name__ == "__main__":
    config.require_api_key()
    print(asyncio.run(run_spec_writer()))
