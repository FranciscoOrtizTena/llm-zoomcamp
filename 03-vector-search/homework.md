## Homework: Vector Search

In this homework, we'll experiemnt with vector with and without Elasticsearch

> It's possible that your answers won't match exactly. If it's the case, select the closest one.


## Q1. Getting the embeddings model

First, we will get the embeddings model `multi-qa-distilbert-cos-v1` from
[the Sentence Transformer library](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html#model-overview)

```python
from sentence_transformers import SentenceTransformer
model_name = 'multi-qa-distilbert-cos-v1'
model = SentenceTransformer(model_name)
```

Create the embedding for this user question:

```python
user_question = "I just discovered the course. Can I still join it?"
q1= model.encode(user_question)
print(q1[0])
```

```text
0.07822259
```

What's the first value of the resulting vector?

- -0.24
- -0.04
- `0.07`
- 0.27


## Prepare the documents

Now we will create the embeddings for the documents.

Load the documents with ids that we prepared in the module:

```python
import requests 

base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/documents-with-ids.json'
docs_url = f'{base_url}/{relative_url}?raw=1'
docs_response = requests.get(docs_url)
documents = docs_response.json()
```

We will use only a subset of the questions - the questions
for `"machine-learning-zoomcamp"`. After filtering, you should
have only 375 documents. To filter use the following code

```python
documents = [item for item in documents if item['course'] == 'machine-learning-zoomcamp']
len(documents)
```

```text
375
```

## Q2. Creating the embeddings

Now for each document, we will create an embedding for both question and answer fields.

We want to put all of them into a single matrix `X`:

- Create a list `embeddings` 
- Iterate over each document 
- `qa_text = f'{question} {text}'`
- compute the embedding for `qa_text`, append to `embeddings`
- At the end, let `X = np.array(embeddings)` (`import numpy as np`) 

```python
from tqdm import tqdm
import numpy as np
embeddings = []
for doc in tqdm(documents):
    qa_text = f"{doc['question']} {doc['text']}"
    embedding = model.encode(qa_text)
    embeddings.append(embedding)

X = np.array(embeddings)
```

What's the shape of X? (`X.shape`). Include the parantheses. 

```python
print(X.shape)
```

```text
(375, 768)
```

## Q3. Search

We have the embeddings and the query vector. Now let's compute the 
cosine similarity between the vector from Q1 (let's call it `v`) and the matrix from Q2. 

The vectors returned from the embedding model are already
normalized (you can check it by computing a dot product of a vector
with itself - it should return something very close to 1.0). This means that in order
to compute the coside similarity, it's sufficient to 
multiply the matrix `X` by the vector `v`:


```python
scores = X.dot(q1)
print(np.max(scores))
```

```text
0.6506574
```

What's the highest score in the results?

- 65.0 
- 6.5
- `0.65`
- 0.065


## Vector search

We can now compute the similarity between a query vector and all the embeddings.

Let's use this to implement our own vector search

```python
class VectorSearchEngine():
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def search(self, v_query, num_results=10):
        scores = self.embeddings.dot(v_query)
        idx = np.argsort(-scores)[:num_results]
        return [self.documents[i] for i in idx]

search_engine = VectorSearchEngine(documents=documents, embeddings=X)
search_engine.search(v, num_results=5)
```

If you don't understand how the `search` function work:

* Ask ChatGTP or any other LLM of your choice to explain the code
* Check our pre-course workshop about implementing a search engine [here](https://github.com/alexeygrigorev/build-your-own-search-engine)

(Note: you can replace `argsort` with `argpartition` to make it a lot faster)


## Q4. Hit-rate for our search engine

Let's evaluate the performance of our own search engine. We will
use the hitrate metric for evaluation.

First, load the ground truth dataset:

```python
import pandas as pd

base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/ground-truth-data.csv'
ground_truth_url = f'{base_url}/{relative_url}?raw=1'

df_ground_truth = pd.read_csv(ground_truth_url)
df_ground_truth = df_ground_truth[df_ground_truth.course == 'machine-learning-zoomcamp']
ground_truth = df_ground_truth.to_dict(orient='records')
```

Now use the code from the module to calculate the hitrate of
`VectorSearchEngine` with `num_results=5`.

First, we need to determine if our search engine retrieves the correct answer with the following code.

```python
relevance_total = []

for q in tqdm(ground_truth):
    doc_id = q['document']
    v_doc_question = model.encode(q['question'])
    results = search_engine.search(v_doc_question, num_results=5)
    relevance = [d['id'] == doc_id for d in results]
    relevance_total.append(relevance)
```

Now, we need to create a hit rate function and evaluate it.

```python
def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)

hit_rate(relevance_total)
```

```text
0.9398907103825137
```

What did you get?

- `0.93`
- 0.73
- 0.53
- 0.33

## Q5. Indexing with Elasticsearch

Now let's index these documents with elasticsearch

* Create the index with the same settings as in the module (but change the dimensions)

```python
from elasticsearch import Elasticsearch

es_client = Elasticsearch('http://localhost:9200') 

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
            "question_text_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}

index_name = "course-questions"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)
```

* Index the embeddings (note: you've already computed them)

```python
for doc in tqdm(documents):
    question = doc['question']
    text = doc['text']
    qt = question + ' ' + text

    doc['question_text_vector'] = model.encode(qt).tolist()

    es_client.index(index=index_name, id=doc['id'], body=doc)
```


After indexing, let's perform the search of the same query from Q1.

```python
def elastic_search_knn(field, vector, course):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "filter": {
            "term": {
                "course": course
            }
        }
    }

    search_query = {
        "knn": knn,
        "_source": ["text", "section", "question", "course", "id"]
    }

    result_docs = es_client.search(
        index=index_name,
        body=search_query
    )

    return result_docs['hits']['hits']
```

What's the ID of the document with the highest score?

```python
user_question = "I just discovered the course. Can I still join it?"
q5 = model.encode(user_question).tolist()
response = elastic_search_knn("question_text_vector", q5, "machine-learning-zoomcamp")

for metric in response:
    print(f"Score: {metric['_score']}, ID: {metric['_id']}, Question: {metric['_source']['question']}")
```

```text
Score: 0.8253288, ID: ee58a693, Question: The course has already started. Can I still join it?
Score: 0.7358538, ID: 0a278fb2, Question: I just joined. What should I do next? How can I access course materials?
Score: 0.7295, ID: 6ba259b1, Question: I filled the form, but haven't received a confirmation email. Is it normal?
Score: 0.72849524, ID: 9f261648, Question: Can I do the course in other languages, like R or Scala?
Score: 0.7252793, ID: e7ba6b8a, Question: The course videos are from the previous iteration. Will you release new ones or we’ll use the videos from 2021?
```

`ee58a693`

## Q6. Hit-rate for Elasticsearch

The search engine we used in Q4 computed the similarity between
the query and ALL the vectors in our database. Usually this is 
not practical, as we may have a lot of data.

Elasticsearch uses approximate techniques to make it faster. 

Let's evaluate how worse the results are when we switch from
exact search (as in Q4) to approximate search with Elastic.

What's hitrate for our dataset for Elastic?

Let's redifine the kkn elastic search function:

```python
def elastic_search_knn(field, vector, course):
    knn = {
        "field": field,
        "query_vector": vector,
        "k": 5,
        "num_candidates": 10000,
        "filter": {
            "term": {
                "course": course
            }
        }
    }

    search_query = {
        "knn": knn,
        "_source": ["text", "section", "question", "course", "id"]
    }

    es_results = es_client.search(
        index=index_name,
        body=search_query
    )
    
    result_docs = []
    
    for hit in es_results['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs
```

Now, a function to enconde each function

```python
def vector_question_text_elastic(q):
    question = q['question']
    
    v_q = model.encode(question)

    return elastic_search_knn("question_text_vector", v_q, "machine-learning-zoomcamp")
```

Finally, the function to evaluate the hit rate

```python
def evaluate(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        doc_id = q['document']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)

    return hit_rate(relevance_total)
```

```python
evaluate(ground_truth, vector_question_text_elastic)
```

```text
0.9398907103825137
```

- `0.93`
- 0.73
- 0.53
- 0.33


## Submit the results

* Submit your results here: https://courses.datatalks.club/llm-zoomcamp-2024/homework/hw3
* It's possible that your answers won't match exactly. If it's the case, select the closest one.
