## Homework: Evaluation and Monitoring

In this homework, we'll evaluate the quality of our RAG system.

> It's possible that your answers won't match exactly. If it's the case, select the closest one.

Solution:

* Video: TBA
* Notebook: TBA

## Getting the data

Let's start by getting the dataset. We will use the data we generated in the module.

In particular, we'll evaluate the quality of our RAG system
with [gpt-4o-mini](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-monitoring/data/results-gpt4o-mini.csv)


Read it:

```python
import pandas as pd

github_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-monitoring/data/results-gpt4o-mini.csv'

url = f'{github_url}?raw=1'
df = pd.read_csv(url)
```

We will use only the first 300 documents:

```python
df = df.iloc[:300]
```

## Q1. Getting the embeddings model

Now, get the embeddings model `multi-qa-mpnet-base-dot-v1` from
[the Sentence Transformer library](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#model-overview)

> Note: this is not the same model as in HW3

```python
from sentence_transformers import SentenceTransformer
model_name = 'multi-qa-mpnet-base-dot-v1'
model = SentenceTransformer(model_name)
```

Create the embeddings for the first LLM answer:

```python
answer_llm = df.iloc[0].answer_llm
```

What's the first value of the resulting vector?

```python
answer_llm_emb = model.encode(answer_llm)
answer_llm_emb[0]
```

```text
-0.4224466
```

- `-0.42`
- -0.22
- -0.02
- 0.21


## Q2. Computing the dot product

Now for each answer pair, let's create embeddings and compute dot product between them

We will put the results (scores) into the `evaluations` list

```python
import numpy as np
from tqdm import tqdm

answer_embeddings_list = []
evaluations = []

for index, row in tqdm(df.iterrows()):
    answer_llm = row.answer_llm
    answer_gt = row.answer_orig

    answer_llm_embeddings = model.encode(answer_llm)
    answer_gt_embeddings = model.encode(answer_gt)

    answer_embeddings_list.append((answer_llm_embeddings, answer_gt_embeddings))
    score = np.dot(answer_llm_embeddings, answer_gt_embeddings)
    evaluations.append(score)
np.percentile(evaluations, 75)
```

```text
31.674306869506836
```


What's the 75% percentile of the score?

- 21.67
- `31.67`
- 41.67
- 51.67

## Q3. Computing the cosine

From Q2, we can see that the results are not within the [0, 1] range. It's because the vectors coming from this model are not normalized.

So we need to normalize them.

To do it, we 

* Compute the norm of a vector
* Divide each element by this norm

So, for vector `v`, it'll be `v / ||v||`

In numpy, this is how you do it:

```python
norm = np.sqrt((v * v).sum())
v_norm = v / norm
```

For normalizing
```python
normalized_embeddings = []
for x in answer_embeddings_list:
    normalized_sublist = []
    for y in x:
        norm = np.linalg.norm(y)
        if norm == 0:
            normalized_sublist.append(y)
        else:
            normalized_sublist.append(y / norm)
    normalized_embeddings.append(normalized_sublist)
```

For calculating the dot product
```python
dot_products = []
for pair in normalized_embeddings:
    dot_product = np.dot(pair[0], pair[1])
    dot_products.append(dot_product)
```

What's the 75% cosine in the scores?

```python
np.percentile(dot_products, 75)
```

```text
0.8362347036600113
```

- 0.63
- 0.73
- `0.83`
- 0.93

## Q4. Rouge

Now we will explore an alternative metric - the ROUGE score.  

This is a set of metrics that compares two answers based on the overlap of n-grams, word sequences, and word pairs.

It can give a more nuanced view of text similarity than just cosine similarity alone.

We don't need to implement it ourselves, there's a python package for it:

```bash
pip install rouge
```

(The latest version at the moment of writing is `1.0.1`)

Let's compute the ROUGE score between the answers at the index 10 of our dataframe (`doc_id=5170565b`)

```python
from rouge import Rouge
rouge_scorer = Rouge()

scores = rouge_scorer.get_scores(r['answer_llm'], r['answer_orig'])[0]

```

There are three scores: `rouge-1`, `rouge-2` and `rouge-l`, and precision, recall and F1 score for each.

* `rouge-1` - the overlap of unigrams,
* `rouge-2` - bigrams,
* `rouge-l` - the longest common subsequence

What's the F score for `rouge-1`?

```python
scores['rouge-1']['f']
```

```text
0.45454544954545456
```

- 0.35
- `0.45`
- 0.55
- 0.65

## Q5. Average rouge score

Let's compute the average F-score between `rouge-1`, `rouge-2` and `rouge-l` for the same record from Q4

```python
np.mean([scores['rouge-1']['f'],scores['rouge-2']['f'],scores['rouge-l']['f']])
```

```text
0.35490034990035496
```

- `0.35`
- 0.45
- 0.55
- 0.65

## Q6. Average rouge score for all the data points

Now let's compute the score for all the records and create a dataframe from them.

```python
json_rouge = {
    'rouge_1': list(),
    'rouge_2': list(),
    'rouge_l': list()
}
for item in tqdm(df.itertuples()):
    scores = rouge_scorer.get_scores(item.answer_llm, item.answer_orig)[0]
    json_rouge['rouge_1'].append(scores['rouge-1']['f'])
    json_rouge['rouge_2'].append(scores['rouge-2']['f'])
    json_rouge['rouge_l'].append(scores['rouge-l']['f'])
df_rouge = pd.DataFrame(json_rouge)
```

What's the average `rouge_2` across all the records?

```python
df_rouge.describe()['rouge_2']['mean']
```

```python
0.20696501983423318
```

- 0.10
- `0.20`
- 0.30
- 0.40


## Submit the results

* Submit your results here: https://courses.datatalks.club/llm-zoomcamp-2024/homework/hw4
* It's possible that your answers won't match exactly. If it's the case, select the closest one.
