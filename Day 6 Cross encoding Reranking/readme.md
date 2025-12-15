# Cross-Encoder Reranking with Milvus

## Overview

A **Cross-Encoder** is a neural network model that takes a query and a document as input and directly predicts a relevance score for that specific query-document pair. Unlike bi-encoders (which compute embeddings separately for queries and documents), cross-encoders jointly process both inputs, allowing them to capture fine-grained interactions between the query and document.

This makes cross-encoders significantly more accurate for ranking and relevance assessment, though at the cost of higher computational expense.

## Cross-Encoder vs Bi-Encoder

| Aspect | Bi-Encoder | Cross-Encoder |
|--------|-----------|---------------|
| **Computation** | Encodes query and documents separately | Jointly encodes query-document pairs |
| **Speed** | Fast - O(D + Q) | Slow - O(N × Q) |
| **Accuracy** | Good for initial filtering | Excellent for precise ranking |
| **Use Case** | Large-scale retrieval | Reranking top-K results |
| **Example** | all-MiniLM-L6-v2 | cross-encoder/ms-marco-MiniLM-L-6-v2 |

## Why Use Cross-Encoders for Reranking?

1. **Higher Accuracy** - Captures complex query-document interactions
2. **Better Relevance Scores** - Provides calibrated relevance predictions
3. **Efficient Pipeline** - Use bi-encoder for fast retrieval, cross-encoder for precise reranking
4. **Reduced False Positives** - Eliminates irrelevant results ranked high by semantic similarity

## Installation

Install the required dependencies:

```bash
pip install --upgrade pymilvus
pip install "pymilvus[model]"
```

## Quick Start
```
python reranking.py
```

## API Reference

### CrossEncoderRerankFunction

The main class for cross-encoder reranking in Milvus.

#### Initialization Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `model_name` | string | Name of the cross-encoder model to use | `"cross-encoder/ms-marco-MiniLM-L-6-v2"` |
| `device` | string | Device to run the model on: `"cpu"` or `"cuda:n"` | `"cpu"` or `"cuda:0"` |

#### Call Parameters

```python
results = ce_rf(
    query=query,           # str: The search query
    documents=documents,   # List[str]: List of documents to rerank
    top_k=3,              # int: Number of top results to return
)
```


## Supported Models

Milvus supports various pre-trained cross-encoder models. Popular lightweight options include:

| Model Name | Size | Speed | Accuracy | Use Case |
|------------|------|-------|----------|----------|
| `cross-encoder/ms-marco-TinyBERT-L-2-v2` | 2L | Very Fast | Good | Edge devices, low latency |
| `cross-encoder/ms-marco-MiniLM-L-2-v2` | 2L | Very Fast | Good | Mobile, resource-constrained |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 6L | Fast | Very Good | **Recommended for most use cases** |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | 12L | Medium | Excellent | Production systems |
| `cross-encoder/ms-marco-MultiBERT-L-12` | 12L | Slow | Best | Highest accuracy required |

For the complete list of available models, refer to the [Pretrained Cross-Encoders documentation](https://www.sbert.net/docs/pretrained_cross-encoders.html).


### Cost Analysis

- **Bi-encoder cost:** O(D + Q) - scales linearly
- **Cross-encoder cost:** O(N × Q) - only scales with reranked set
- **Hybrid approach:** Use bi-encoder for fast filtering, cross-encoder for precise reranking

**Example:** 1M documents, 100 queries, top-5 reranking
- Bi-encoder: ~1M + 100 operations
- Cross-encoder: ~5 × 100 = 500 operations only

## References

- **Milvus Cross-Encoder Documentation:** https://milvus.io/docs/rerankers-cross-encoder.md
- **Sentence Transformers:** https://www.sbert.net/
- **Pretrained Cross-Encoders:** https://www.sbert.net/docs/pretrained_cross-encoders.html
- **Cross-Encoders Paper:** https://arxiv.org/abs/1910.14424


