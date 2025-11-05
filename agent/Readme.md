## Setup Instructions

Before running any scripts, make sure to set your **OpenAI API key** in `agent_system.py`.

You can do this by editing the following lines near the top of the file:

```python
# Load from env or set default
api_key =""
openai.api_key = api_key
os.environ['OPENAI_API_KEY'] = ""
```

You also need to create the following folders:


./dataset — contains the full dataset (please download and place it here).


./input — used for temporary data during forward modeling.

Once your API key is set, you can start the system by running:
python agent_system.py

This will launch the interactive agent framework for forward model training and inverse design.
Please note that the entire run may take more than 20 hours to complete.
