## Setup Instructions

Before running any scripts, make your **OpenAI API key** available. The code reads it from the `OPENAI_API_KEY` environment variable; no file needs to be edited.

Either export it in your shell:

```
export OPENAI_API_KEY=sk-...
```

or put it in a file named `.env` in this folder (ignored by git):

```
OPENAI_API_KEY=sk-...
```

Prepare the dataset once (after extracting ADM.zip into `./dataset`):
python prepare_dataset.py

Once your API key is set, you can start the system by running:
python agent_system.py

The other settings (target MSE, dataset size limit, AIDE steps and models, time limits) are in `config.py` and can be changed with environment variables of the same name.

## Run on your dataset

Currently, to save the time, we are simulating the data generation by downsampling from the full dataset. If you want to use your own dataset or change the way to generate the data, please modify generate_dataset function in forward_loop.py 

If you have multiple targets, modify the inverse designer's tool in subagent.py (`train_inverse_model` in inverse_NA.py runs a whole file of targets).
