## Setup Instructions

Before running any scripts, make sure to set your **OpenAI API key** in `agent_system.py`.

You can do this by editing the following lines near the top of the file:

```python
# Load from env or set default
api_key =""
openai.api_key = api_key
os.environ['OPENAI_API_KEY'] = ""
```
Once your API key is set, you can start the system by running:
python agent_system.py

## Run on your dataset

Please modify generate_dataset function in forward_loop.py 
