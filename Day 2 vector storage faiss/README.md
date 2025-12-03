# Vector Storage with FAISS

Efficient retrieval depends on how embeddings are stored and searched. This module introduces FAISS, a fast similarity search library that supports millions of vectors with low latency. The goal here is not just to store vectors, but to understand how indexing choices affect retrieval quality and performance.

## Why Vector Storage Matters

A retrieval system must:
- Embed text into vectors
- Store those vectors efficiently
- Find the nearest neighbors quickly
- Return stable and reproducible search results

Storing vectors in Python lists or NumPy arrays might work for tiny datasets, but anything beyond a few thousand vectors becomes slow. FAISS solves this by giving you GPU/CPU-optimized search structures.

## What This Module Covers

- Creating embeddings for chunks
- Building a basic FAISS index (L2)
- Adding and querying vectors
- Comparing search results
- Saving and loading the index for reuse

This is the foundation for all retrieval tasks that follow.

## When to Use This Approach

- Small to medium datasets (under a few million vectors)
- CPU-only environments
- Prototyping or educational projects
- Testing retrieval techniques without a full vector database

For larger or distributed use cases, a specialized vector DB (Qdrant, Milvus, Pinecone) is more appropriate.

## Running the Demo

```
pip install -r requirements.txt
```
```
python demo_faiss.py
```

You should see a list of retrieved chunks along with similarity scores.

## Notes

- This module uses a lightweight embedding model to keep the demo simple.
- The index is built using `IndexFlatL2`, which performs exact nearest-neighbor search.