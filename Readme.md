# Metamaterials Agent: Forward Modeling & Inverse Design

LLM-driven agents that **train a forward surrogate model** (geometry → spectrum) and perform **Neural-Adjoint inverse design** (target spectrum → geometry). 

- 🔁 End-to-end workflow: data prep → forward training → inverse design
- 🧠 iterative code generation & evaluation
- 🧪 Neural-Adjoint backpropagation for inverse design

---

## Quick Start

### 1) Requirements
- simple requirement in requirements.txt
- full requirement in environment_full.yml

To dive into the details of forward modeling, please check the used AIDE (https://github.com/WecoAI/aideml).

### 2) Datasets
- Download the data in: https://research.repository.duke.edu/concern/datasets/z316q2403?locale=en and place them in ./agent/dataset
- Inverse dataset spilit in: ./agent/dataset

### 3) Run
- python agent_system.py
- It will take more then 20 hours

---
## Typical Workflow
👤 Hi there! I'm here to assist you with metamaterials deep learning tasks. Briefly: do you want to run Forward training, Inverse design, or Both? (reply: forward / inverse / both)

User: both

👤 Please describe your task and the dataset you have, in one or two paragraphs.

User: The task involves designing and optimizing a deep learning regression model to predict the electromagnetic spectrum from geometry parameters. The goal is to achieve a mean squared error (MSE) of 2e-3.

👤 Could you clarify the model’s input/output dimensions?

User: input 14; output: 2001

Then, agent will perform the forward modeling.

---

We will continue to improve it!
