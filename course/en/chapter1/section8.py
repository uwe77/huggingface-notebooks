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
# # Bias and limitations

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
def test_fill_mask():

    from transformers import pipeline
    import pytest

    unmasker = pipeline("fill-mask", model="bert-base-uncased")
    result = unmasker("This man works as a [MASK].")
    print([r["token_str"] for r in result])
    list_a = ([r["token_str"] for r in result])
    # check if list_a includes 'doctor'
    assert 'doctor' in list_a

    result = unmasker("This woman works as a [MASK].")
    print([r["token_str"] for r in result])
    list_b = ([r["token_str"] for r in result])
    # check if list_b includes 'nurse'
    assert 'nurse' in list_b


