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
# # Tokenizers (PyTorch)

# %% [markdown]
# Install the Transformers, Datasets, and Evaluate libraries to run this notebook.

# %%
# !pip install datasets evaluate transformers[sentencepiece]

# %%
tokenized_text = "Jim Henson was a puppeteer".split()
print(tokenized_text)

# %%
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# %%
tokenizer("Using a Transformer network is simple")

# %%
tokenizer.save_pretrained("directory_on_my_computer")

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)

# %%
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)

# %%
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
