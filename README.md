# For SIS Course

Jupyter Notebook is a great tool for teaching and learning.
Nevertheless, it is not a good tool for version control.
In this course, we will use JupyText to sync the notebook with a Python script.
Each notebook is paried with a Python script via JupyText.
In each cell, we will use pytest-style format to make it compatiable for both notebook and Python script.

## 1. To Set Up the Sync

```
jupytext --set-formats ipynb,py:percent notebook.ipynb 
```

We will see a new file `notebook.py` in the same directory.

## 2. Use IDE (nvim or VSCode) to Edit the Python Script as pytest format

```python
# %%

# %%ipytst -qq
def test_task1():
    import pytest
    from transformers import pipeline

    assert 1 + 1 == 2
```

Note that the `# %%` is the cell marker for JupyText.
The `# %%ipytst -qq` is the cell marker for ipytest.

To run the test, we can use `pytest` in the terminal.

```
pytest notebook.py::test_task1 -s
```

Note that the `-s` is to show the print out.

## To Sync (py to ipynb)

```
jupytext --sync notebook.py
```

## To Run the Cells in the Notebook

We have set up ipytest in the notebook.
```python
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
```

We could not run the cells in the notebook as we do in the regular Jupyter Notebook.

