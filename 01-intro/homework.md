## Homework: Introduction

In this homework, we'll learn more about search and use Elastic Search for practice. 

> It's possible that your answers won't match exactly. If it's the case, select the closest one.

## Q1. Running Elastic 

Run Elastic Search 8.4.3, and get the cluster information. If you run it on localhost, this is how you do it:

```bash
curl localhost:9200
```

What's the `version.build_hash` value?

`42f05b9372a9a4a470db3b52817899b99a76ee73`

## Getting the data

Now let's get the FAQ data. You can run this snippet:

```python
import requests 

docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
```

Note that you need to have the `requests` library:

```bash
pip install requests
```

## Q2. Indexing the data

Index the data in the same way as was shown in the course videos. Make the `course` field a keyword and the rest should be text. 

Don't forget to install the ElasticSearch client for Python:

```bash
pip install elasticsearch
```

Which function do you use for adding your data to elastic?

- insert
- `index`
- put
- add

```python
es_client_homework = Elasticsearch('http://localhost:9200')

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
            "course": {"type": "keyword"} 
        }
    }
}

index_name = "course-questions-homework"

es_client_homework.indices.create(index=index_name, body=index_settings)

for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)
```

## Q3. Searching

Now let's search in our index. 

We will execute a query "How do I execute a command in a running docker container?". 

Use only `question` and `text` fields and give `question` a boost of 4, and use `"type": "best_fields"`.

What's the score for the top ranking result?

* 94.05
* `84.05`
* 74.05
* 64.05

Look at the `_score` field.

```python
def elastic_search_homework(query):

    search_query = {
        "size": 10,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^4", "text"],
                        "type": "best_fields"
                    }
                },
#                "filter": {
#                    "term": {
#                        "course": "data-engineering-zoomcamp"
#                    }
#                }
            }
        }
    }
    
    response = es_client.search(index=index_name, body=search_query)
    
    result_docs = []
    for hit in response['hits']['hits']:
        result_docs.append(hit['_score'])

    return result_docs

elastic_search_homework('"How do I execute a command in a running docker container?"')[0]
```

## Q4. Filtering

Now let's only limit the questions to `machine-learning-zoomcamp`.

Return 3 results. What's the 3rd question returned by the search engine?

- How do I debug a docker container?
- `How do I copy files from a different folder into docker container’s working directory?`
- How do Lambda container images work?
- How can I annotate a graph?

```python
def elastic_search_homework(query):

    search_query = {
        "size": 3,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^4", "text"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "machine-learning-zoomcamp"
                    }
                }
            }
        }
    }
    
    response = es_client.search(index=index_name, body=search_query)
    
    result_docs = []
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs
elastic_search_homework('"How do I execute a command in a running docker container?"')[2]['question']
```

## Q5. Building a prompt

Now we're ready to build a prompt to send to an LLM. 

Take the records returned from Elasticsearch in Q4 and use this template to build the context. Separate context entries by two linebreaks (`\n\n`)
```python
context_template = """
Q: {question}
A: {text}
""".strip()
```

Now use the context you just created along with the "How do I execute a command in a running docker container?" question 
to construct a prompt using the template below:

```
prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()
```

What's the length of the resulting prompt? (use the `len` function)

- 962
- `1462`
- 1962
- 2462

```python
query = 'How do I execute a command in a running docker container?'

search_query = {
    "size": 3,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^4", "text"],
                    "type": "best_fields"
                }
            },
            "filter": {
                "term": {
                    "course": "machine-learning-zoomcamp"
                }
            }
        }
    }
}

response = es_client.search(index=index_name, body=search_query)

result_docs = []
for hit in response['hits']['hits']:
    result_docs.append(hit['_source'])
```

```python
def build_prompt(query, search_results):

    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
    """.strip()
    
    context = ""
    
    for doc in search_results:
        context = context + f"Q: {doc['question']}\nA: {doc['text']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()

    return prompt
len(build_prompt(query, result_docs))
```

## Q6. Tokens

When we use the OpenAI Platform, we're charged by the number of 
tokens we send in our prompt and receive in the response.

The OpenAI python package uses `tiktoken` for tokenization:

```bash
pip install tiktoken
```

Let's calculate the number of tokens in our query: 

```python
encoding = tiktoken.encoding_for_model("gpt-4o")
```

Use the `encode` function. How many tokens does our prompt have?

- 122
- 222
- `322`
- 422

```python
encoding = tiktoken.encoding_for_model("gpt-4o")
promtp_homework = build_prompt(query, result_docs)
tokens = encoding.encode(promtp_homework)
len(tokens)
```

Note: to decode back a token into a word, you can use the `decode_single_token_bytes` function:

```python
encoding.decode_single_token_bytes(63842)
```

## Bonus: generating the answer (ungraded)

Let's send the prompt to OpenAI. What's the response?  

Note: you can replace OpenAI with Ollama. See module 2.

## Bonus: calculating the costs (ungraded)

Suppose that on average per request we send 150 tokens and receive back 250 tokens.

How much will it cost to run 1000 requests?

You can see the prices [here](https://openai.com/api/pricing/)

On June 17, the prices for gpt4o are:

* Input: $0.005 / 1K tokens
* Output: $0.015 / 1K tokens

You can redo the calculations with the values you got in Q6 and Q7.


## Submit the results

* Submit your results here: https://courses.datatalks.club/llm-zoomcamp-2024/homework/hw1
* It's possible that your answers won't match exactly. If it's the case, select the closest one.
