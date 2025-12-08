# Semantic Search vs Keyword Search

Retrieval performance has a direct impact on the accuracy of any RAG system. Traditional search engines rely on keyword matching, while modern systems use embeddings to understand meaning. This module compares both methods to illustrate how they differ and when each approach makes sense.

## What This Module Covers

- Implementing a simple keyword-based search using TF-IDF
- Implementing a semantic search pipeline using embeddings
- Comparing results for different query types
- Understanding strengths and weaknesses of each approach

The goal is to develop intuition about how retrieval methods behave before integrating them into a RAG pipeline.

## When Keyword Search Works Well

- Documents use consistent vocabulary  
- Queries match the same wording as source text  
- You need exact phrase matching  
- Lightweight systems with minimal dependencies

## When Semantic Search Wins

- Documents use varied or ambiguous language  
- Users phrase queries differently from the document  
- You need meaning-based retrieval rather than exact words  
- The system must generalize beyond exact keywords

## Running the Demo

```bash
python search.py
```