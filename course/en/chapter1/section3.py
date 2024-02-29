# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Transformers, what can they do?

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
# %%ipytest -qq
def test_sentiment_analysis():

    import pytest
    from transformers import pipeline
    
    classifier = pipeline("sentiment-analysis")
    results = classifier("I've been waiting for a HuggingFace course my whole life.")

    print(results)
    assert results[0]['label'] == 'POSITIVE'
    assert results[0]['score'] > 0.9

# %%
def test_sentiment_analysis2():

    import pytest
    from transformers import pipeline
    
    classifier = pipeline("sentiment-analysis")
    results = classifier(
        ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
    )

    print(results)
    assert results[0]['label'] == 'POSITIVE'
    assert results[0]['score'] > 0.9

    assert results[1]['label'] == 'NEGATIVE'
    assert results[1]['score'] > 0.9

# %%
def test_zero_shot_classification():

    from transformers import pipeline
    import pytest

    classifier = pipeline("zero-shot-classification")
    results = classifier(
        "This is a course about the Transformers library",
        candidate_labels=["education", "politics", "business"],
    )
    print(results)
    scores_list = results['scores']
    max_index = scores_list.index(max(scores_list))
    assert max_index == 0

# %%
# write a test for text generation pipeline
def test_text_generation():
    from transformers import pipeline
    import pytest

    generator = pipeline("text-generation")
    results = generator("In this course, we will teach you how to")
    print(results)
    assert len(results) == 1

# %%
# write a test for text generation pipeline with distilgpt2
def test_text_generation2():
    from transformers import pipeline
    import pytest

    generator = pipeline("text-generation", model="distilgpt2")
    results = generator(
        "In this course, we will teach you how to",
        max_length=30,
        num_return_sequences=2,
    )
    print(results)
    assert len(results) == 2

# %%
# write a test for fill mask task
def test_fill_mask():
    from transformers import pipeline
    import pytest

    unmasker = pipeline("fill-mask")
    results = unmasker("This course will teach you all about <mask> models.", top_k=2)
    print(results)
    assert results[0]['token_str'] == 'language'
    assert results[0]['score'] > 0.9

# %%
# write a test for NER task
def test_ner():
    from transformers import pipeline
    import pytest

    ner = pipeline("ner", grouped_entities=True)
    results = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
    print(results)
    assert results[0]['word'] == 'Sylvain'
    assert results[0]['score'] > 0.9

# %%
# write a test for question answering task
def test_question_answering():
    from transformers import pipeline
    import pytest

    question_answerer = pipeline("question-answering")
    results = question_answerer(
        question="Where do I work?",
        context="My name is Sylvain and I work at Hugging Face in Brooklyn",
    )
    print(results)
    assert results['answer'] == 'Hugging Face'
    assert results['score'] > 0.9

# %%
# write a test for summarization task
def test_summarization():
    from transformers import pipeline
    import pytest

    summarizer = pipeline("summarization")
    results = summarizer(
        """
        America has changed dramatically during recent years. Not only has the number of 
        graduates in traditional engineering disciplines such as mechanical, civil, 
        electrical, chemical, and aeronautical engineering declined, but in most of 
        the premier American universities engineering curricula now concentrate on 
        and encourage largely the study of engineering science. As a result, there 
        are declining offerings in engineering subjects dealing with infrastructure, 
        the environment, and related issues, and greater concentration on high 
        technology subjects, largely supporting increasingly complex scientific 
        developments. While the latter is important, it should not be at the expense 
        of more traditional engineering.

        Rapidly developing economies such as China and India, as well as other 
        industrial countries in Europe and Asia, continue to encourage and advance 
        the teaching of engineering. Both China and India, respectively, graduate 
        six and eight times as many traditional engineers as does the United States. 
        Other industrial countries at minimum maintain their output, while America 
        suffers an increasingly serious decline in the number of engineering graduates 
        and a lack of well-educated engineers.
        """
    )
    print(results)
    assert len(results) == 1

# %%
# write a test for translation task
def test_translation():
    from transformers import pipeline
    import pytest

    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
    results = translator("Ce cours est produit par Hugging Face.")
    print(results)
    assert results[0]['translation_text'] == 'This course is produced by Hugging Face.'

