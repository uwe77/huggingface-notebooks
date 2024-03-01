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
# # Behind the pipeline (PyTorch)

# %% [markdown]
# Install the Transformers, Datasets, and Evaluate libraries to run this notebook.

# %%
# !pip install datasets evaluate transformers[sentencepiece]

# %%
def in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True

import ipytest
# check if the cell is in jupyter notebook or not
if in_notebook():
    ipytest.autoconfig()

# %%
# %%ipytest -vvv
# test sentiment analysis
def test_sentiment_analysis():

    import pytest
    from transformers import pipeline
    
    classifier = pipeline("sentiment-analysis")
    results = classifier(
        [
            "I've been waiting for a HuggingFace course my whole life.",
            "I hate this so much!",
        ]
    )

    print(results)
    assert results[0]['label'] == 'POSITIVE'
    assert results[0]['score'] > 0.9
    assert results[1]['label'] == 'NEGATIVE'
    assert results[1]['score'] > 0.9

# %%
# %%ipytest -vvv
# test tokenizer
def test_tokenizer():

    import pytest
    from transformers import AutoTokenizer
    
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    print(inputs)

    assert inputs["input_ids"].shape == inputs["attention_mask"].shape

## %%
# %%ipytest -vvv
# test AutoModel
def test_auto_model():

    import pytest
    from transformers import AutoModel
    
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModel.from_pretrained(checkpoint)

    # add inputs for the model with fake ids and masks
    inputs = {
        "input_ids": [
            [101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
            2607,  2026,  2878,  2166,  1012,   102], 
            [101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,    
            0,     0,     0,     0,     0,     0]],
        "attention_mask": [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
            }
    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)


# %%
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

## %%
#from transformers import AutoModelForSequenceClassification
#
#checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
#model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
#outputs = model(**inputs)
#
## %%
#print(outputs.logits.shape)
#
## %%
#print(outputs.logits)
#
## %%
#import torch
#
#predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
#print(predictions)
#
## %%
#model.config.id2label
