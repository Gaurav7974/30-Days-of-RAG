# Fused Retrieval with Reciprocal Rank Fusion (RRF)

## Overview

This approach combines multiple search methods using Reciprocal Rank Fusion (RRF) to improve retrieval quality in RAG systems. By merging results from vector search (semantic) and hybrid search (semantic + keyword), we get better coverage and more robust results than any single method alone.

## Why RRF?

Single retrieval methods have limitations:
- Vector search misses exact keyword matches
- Keyword search struggles with paraphrases
- Hybrid search uses only one ranking signal

RRF fuses multiple ranked lists without needing score normalization:

```python
score(doc) = Σ (1 / (k + rank_i))
```
Where
- `rank_i` is the position of the document in retrieval method i
- `k` is a constant (default: 60) that controls how rapidly scores decrease

## Architecture

1. **Parallel Retrieval** – Run vector search and hybrid search simultaneously
2. **RRF Fusion** – Merge ranked lists using reciprocal rank scoring
3. **Final Ranking** – Return top-k documents based on fused scores

## Installation

```bash
pip install chromadb
```

## Quick Start

```bash
python rrf_fusion.py
```

## Configuration

- `top_k_each`: Results per method (default: 20, range: 10-50)
- `final_k`: Final results after fusion (default: 10, range: 5-20)
- `k`: RRF constant controlling score decay (default: 60, range: 30-100)

## When to Use

✅ **Good for:**
- Varied query types (semantic + keyword-heavy)
- Inconsistent single-method performance
- Combining multiple data sources

❌ **Not ideal for:**
- Only one retrieval method available
- Extreme low-latency requirements

## Performance

- **Accuracy**: Typically 5-15% improvement over single methods
- **Latency**: Parallel searches are faster than sequential but slower than single-method
- **Scalability**: Fusion overhead is minimal (O(n log n))

# Learn More

## Core Papers
* [Original RRF Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) - Reciprocal Rank Fusion
* [RAG-Fusion](https://arxiv.org/html/2402.03367v2) - RRF in RAG systems
* [Rank Fusion Survey](https://arxiv.org/abs/2402.03367) - Modern fusion methods

## Vector Databases
* [ChromaDB Docs](https://docs.trychroma.com/) - Official documentation
* [Pinecone Hybrid Search](https://www.pinecone.io/learn/hybrid-search-intro/) - Hybrid search guide
* [Milvus Filtering](https://milvus.io/docs/filtering.md) - Advanced filtering
* [Qdrant Hybrid Search](https://qdrant.tech/articles/hybrid-search/) - Implementation guide

## RAG & Retrieval
* [Building Production RAG](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1) - End-to-end guide
* [Advanced RAG Techniques](https://arxiv.org/abs/2312.10997) - Survey paper
* [LlamaIndex Guide](https://docs.llamaindex.ai/en/stable/use_cases/q_and_a/) - RAG patterns

## Benchmarks
* [BEIR Benchmark](https://github.com/beir-cellar/beir) - Retrieval evaluation
* [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Embedding rankings
* [RAGAs](https://github.com/explodinggradients/ragas) - RAG evaluation