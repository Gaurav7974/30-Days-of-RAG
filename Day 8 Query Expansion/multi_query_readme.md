# Multi-Query RAG System

## Overview

Multi-Query RAG generates multiple variations of a user's question before retrieval. Instead of searching once with the original query, the system creates 3-5 alternative phrasings and retrieves documents for each. Results are merged and deduplicated to maximize coverage.

This solves a core RAG problem: relevant documents often use different terminology or phrasing than the user's query, causing semantic search to miss them.

## Why Multi-Query Works

A single query assumes users phrase questions optimally. In reality:
- Technical concepts have multiple names
- Questions can be asked from different angles  
- Domain-specific language varies across documents

By generating variations, the system captures documents that match any reasonable phrasing of the same question.

**Example**

Query: "How does Tesla make money?"

Generated variations:
1. "What are Tesla's revenue sources?"
2. "How does Tesla generate income?"  
3. "What is Tesla's business model for profitability?"

Each variation retrieves different but related documents. Combined, they provide comprehensive coverage that a single query would miss.

## Architecture

```
User Query
    ↓
Generate 3-5 Query Variations (LLM)
    ↓
Retrieve Documents for Each Variation (Vector DB)
    ↓
Merge Results + Remove Duplicates
    ↓
Final Document Set → LLM for Answer Generation
```

The LLM acts as a query rewriter, not a retriever. Actual retrieval uses standard vector similarity on ChromaDB.

## Key Benefits

- **Better Recall** – Finds relevant docs missed by single queries
- **Handles Ambiguity** – Multiple perspectives catch different phrasings
- **Robust to User Phrasing** – Less sensitive to how questions are asked

## Trade-offs

| Aspect | Benefit | Cost |
|--------|---------|------|
| Coverage | +40-60% more relevant docs | +1-2s latency for generation |
| Robustness | Works across phrasings | 3-5x retrieval calls |
| Quality | Higher recall | May include more noise |

Multi-query prioritizes recall over speed. For latency-critical applications, consider query rewriting or hybrid approaches instead.

## When to Use

**Good fit:**
- Open-ended questions with multiple valid phrasings
- Domain-specific terminology with synonyms
- User queries are short or ambiguous
- Recall matters more than precision

**Poor fit:**
- Well-formed queries with clear intent
- Time-sensitive applications (<500ms required)
- Queries already comprehensive and specific
- Document corpus uses consistent terminology

## Installation

```bash
pip install langchain-openai langchain-chroma langchain-core chromadb python-dotenv pydantic
```

Setup `.env`:
```
OPENROUTER_API_KEY=your-key-here
```

Run:
```bash
python Multi_query.py
```

## Technical Stack

- **LLM**: GPT-4o-mini via OpenRouter
- **Embeddings**: text-embedding-3-small  
- **Vector DB**: ChromaDB (persistent local storage)
- **Framework**: LangChain with structured output parsing

## Comparison with Other Approaches

**vs HyDE:** Multi-query generates question variations; HyDE generates hypothetical answers. Multi-query is simpler but HyDE often has better precision.

**vs Query Rewriting:** Query rewriting improves a single query; multi-query runs multiple. Multi-query has better recall; rewriting is faster.

**vs RAG Fusion:** Similar concept. RAG Fusion adds reciprocal rank fusion for scoring. This implementation uses simple deduplication.

## Production Considerations

**Caching:** Cache generated variations for common queries to reduce latency.

**Async Retrieval:** Run retrievals in parallel instead of sequentially to cut latency by 60-70%.

**Variation Count:** Start with 3 variations. More variations = better recall but higher cost. Tune based on your latency budget.

**Deduplication:** Current implementation uses content hash. Consider semantic deduplication for better quality.

# References

## LangChain Documentation
- [Multi-Query Retriever](https://python.langchain.com/docs/how_to/multi_query/)
- [Query Transformations](https://python.langchain.com/docs/how_to/query_transformations/)

## Research & Concepts
- [RAG Fusion Paper](https://github.com/Raudaschl/RAG-Fusion) - Advanced multi-query with reciprocal rank fusion
- [Query Expansion Techniques](https://arxiv.org/abs/2305.03653) - Academic overview of query expansion methods

## Related Techniques
- [HyDE (Hypothetical Document Embeddings)](https://www.chitika.com/hyde-query-expansion-rag/)
- [Query Rewriting for RAG](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83)

## Production RAG Systems
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Pinecone Query Expansion Guide](https://www.pinecone.io/learn/query-expansion/)
- [Weaviate Hybrid Search](https://weaviate.io/developers/weaviate/search/hybrid)