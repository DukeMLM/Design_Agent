# interactive_spec_writer.py
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool, ModelSettings
import openai
import os
# Load from env or set default
api_key =""
openai.api_key = api_key
os.environ['OPENAI_API_KEY'] = ""

''' AIDEML Task Description Writer Agent'''



@function_tool
def ask_human(prompt: str) -> str:
    '''Prompt the user for input and return their response.'''
    print("📝  Let's create a task description!")
    """Display *prompt* in the console and return the user’s reply."""
    return input(f"\n👤 {prompt}\n> ")

@function_tool
def write_txt(content: str) -> str:
    """
    Write *content* into 'description.txt' (UTF-8) and return the absolute path.
    """
    with open("./desp_flex.txt", "w", encoding="utf-8") as f:
      f.write(content)
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
   1. Automatically explore and train deep neural networks to predict the 2001D spectrum from 14D geometry inputs.  

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
   write_txt(<the full markdown string you generated>)
""", model_settings=ModelSettings(temperature=0), tools=[ask_human, write_txt], )

from agents.tool import function_tool
@function_tool
async def generate_desp():
    result = await Runner.run(TASK_AGENT, "")

    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(generate_desp())