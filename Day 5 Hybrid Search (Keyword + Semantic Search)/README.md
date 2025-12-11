# Hybrid Search (Keyword + Semantic Retrieval)

Hybrid retrieval combines keyword-based scoring with semantic similarity to produce more reliable search results. Keyword methods handle exact terminology well, while semantic models capture meaning even when the wording differs. Blending both approaches helps avoid the weaknesses of using either method alone.

## Overview

This module demonstrates how to:

- Compute keyword relevance using BM25
- Compute semantic similarity using embeddings
- Normalize and combine both scoring signals
- Test multiple hybrid scoring formulas
- Perform a two-stage retrieval process commonly used in production systems

The goal is to show how hybrid methods create more stable and accurate rankings, especially when queries and documents use different vocabulary or when precise terminology matters.

## Components

### BM25 Keyword Scoring
A strong baseline for keyword retrieval. It emphasizes rare but important words and reduces the influence of very common terms.

### Semantic Embedding Scoring
Uses a SentenceTransformer model to evaluate similarity in meaning rather than relying on exact wording.

### Hybrid Scoring
Several approaches are implemented:

- **Weighted linear combination**  
- **Harmonic mean scoring**  
- **Maximum blend scoring**

Each scoring method highlights different aspects of relevance and allows tuning based on the use case.

### Multi-Query Expansion
Expands the original query into several related variants. This increases recall and helps semantic search capture a broader interpretation of the question.

### Two-Stage Retrieval
A practical pattern used in many real systems:

1. **Initial retrieval** using hybrid scoring  
2. **Re-ranking** of top candidates using only semantic similarity  

This approach balances speed and accuracy.

## Running the Script

```bash
python hybrid_search.py
