# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
# ---

# %% [markdown]
# # Models (PyTorch)

# %% [markdown]
# Install the Transformers, Datasets, and Evaluate libraries to run this notebook.

# %%
# !pip install datasets evaluate transformers[sentencepiece]

# %%
from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)

# %%
print(config)

# %%
from transformers import BertConfig, BertModel

config = BertConfig()
model = BertModel(config)

# Model is randomly initialized!

# %%
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")

# %%
model.save_pretrained("directory_on_my_computer")

# %%
sequences = ["Hello!", "Cool.", "Nice!"]

# %%
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

# %%
import torch

model_inputs = torch.tensor(encoded_sequences)

# %%
output = model(model_inputs)
