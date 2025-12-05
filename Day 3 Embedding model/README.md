# Embedding Model Comparison

Retrieval quality depends heavily on the embedding model used. Different models produce different vector spaces, cluster text differently, and vary widely in how well they capture semantic meaning. This module compares several embedding models side-by-side to understand how they behave and how model choice impacts retrieval.

## What This Module Covers

- Encoding the same text using multiple embedding models  
- Measuring cosine similarity between chunks  
- Comparing the semantic tightness of embeddings  
- Observing how different models influence retrieval relevance  
- Building intuition around model selection for different tasks  

The goal is to develop a practical understanding of how embeddings work rather than choosing models blindly.

## Models Included

This module evaluates three commonly used embedding models:

- `all-MiniLM-L6-v2` – lightweight baseline  
- `all-mpnet-base-v2` – stronger semantic representation  
- `intfloat/e5-small-v2` – instruction-tuned encoder designed for retrieval  

These cover the typical spectrum from fast to high-quality.

## How to Run

```bash
python compare_models.py
```