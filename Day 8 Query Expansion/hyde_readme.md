# HyDE RAG System

## Overview

HyDE (Hypothetical Document Embeddings) improves RAG retrieval by generating a hypothetical answer first, then using it to find similar documents. Instead of searching with the user's question directly, the system creates what an ideal answer would look like and searches for documents that match that answer.

This solves the semantic gap problem: questions and answers often use different vocabulary and structure, making direct query-to-document matching less effective.

## Why HyDE Works

Traditional RAG searches by matching the query to documents. This fails because:
- Questions are short and lack context
- Documents contain answers, not questions
- Query vocabulary differs from document vocabulary

HyDE bridges this gap by matching answer-to-answer instead of query-to-answer, significantly improving semantic alignment.

**Example**

Query: "How do neural networks learn?"

Traditional approach:
- Searches directly with the question
- Matches on keywords: "neural", "networks", "learn"
- May miss documents using terms like "training", "backpropagation", "optimization"

HyDE approach:
- Generates hypothetical answer: "Neural networks learn through backpropagation, adjusting weights based on error gradients..."
- Searches using this hypothetical answer
- Finds documents with similar content structure and vocabulary
- Better matches even if original query phrasing differs

## Architecture

```
User Query
    ↓
Generate Hypothetical Answer (LLM)
    ↓
Embed Hypothetical Answer
    ↓
Retrieve Similar Documents (Cosine Similarity)
    ↓
Generate Final Answer with Retrieved Context (LLM)
    ↓
Final Answer
```

The key innovation: use the hypothetical answer's embedding for retrieval, not the query's embedding.

## Key Benefits

- **Better Semantic Matching** – Answer-to-answer similarity works better than query-to-answer
- **Zero-Shot Capability** – No training data or fine-tuning needed
- **Handles Vocabulary Mismatch** – Bridges gap between how users ask and how documents explain

## Trade-offs

| Aspect | Benefit | Cost |
|--------|---------|------|
| Precision | Higher quality matches | Extra LLM call adds latency |
| Semantic alignment | Better than direct queries | Hypothetical doc may be inaccurate |
| Flexibility | Works across domains | Quality depends on LLM capability |

HyDE adds one LLM generation step before retrieval. This typically adds 0.5-1s latency but significantly improves result quality.

## When to Use

**Good fit:**
- Queries require domain-specific or technical knowledge
- Vocabulary mismatch between users and documents
- Short queries that lack context
- Precision is more important than speed

**Poor fit:**
- Queries are already detailed and well-formed
- Documents match user vocabulary closely
- Ultra-low latency requirements (<500ms)
- Simple keyword matching works well

## Installation

```bash
pip install openai numpy python-dotenv
```

Setup `.env`:
```
OPENROUTER_API_KEY=your-key-here
```

Run:
```bash
python HyDE.py
```

## Technical Stack

- **LLM**: GPT-4o-mini via OpenRouter (for both hypothetical doc and final answer)
- **Embeddings**: Bag-of-words with TF normalization (simple but effective)
- **Similarity**: Cosine similarity on normalized vectors
- **Storage**: In-memory (suitable for small-to-medium document sets)

## How It Differs from Other Approaches

**vs Standard RAG:** Standard RAG embeds the query directly. HyDE embeds a generated answer. HyDE typically has 15-30% better precision.

**vs Multi-Query:** Multi-Query generates multiple question variations. HyDE generates one answer. HyDE is faster but less comprehensive.

**vs Query Rewriting:** Query rewriting improves the question. HyDE replaces it with an answer. HyDE handles semantic gaps better.

**vs Dense Retrieval:** Dense retrieval uses learned embeddings. HyDE uses simple embeddings but with hypothetical answers. HyDE works without training.

## Production Considerations

**Embedding Quality:** Current implementation uses bag-of-words. For production, consider:
- OpenAI embeddings (text-embedding-3-small)
- Sentence transformers (all-MiniLM-L6-v2)
- Domain-specific embedding models

**Caching:** Cache hypothetical documents for common queries to reduce latency by 50%.

**Error Handling:** If hypothetical doc generation fails, fall back to standard query-based retrieval.

**Token Usage:** Hypothetical docs use 200-400 tokens. Monitor costs if handling high query volume.

**Vector Database:** For larger datasets, replace in-memory storage with:
- Pinecone, Weaviate, or Qdrant for production scale
- ChromaDB or FAISS for local/medium scale

## Limitations

- Adds latency due to hypothetical doc generation
- Quality depends on LLM's ability to generate good hypothetical answers
- May generate incorrect hypothetical answers for niche topics
- Simple bag-of-words embeddings limit semantic understanding

## Improvements for Production

1. **Better Embeddings**: Use OpenAI or Sentence-BERT embeddings
2. **Hybrid Search**: Combine HyDE with keyword search (BM25)
3. **Reranking**: Add cross-encoder reranking after retrieval
4. **Fallback**: If HyDE returns no results, retry with original query
5. **Caching**: Cache hypothetical docs and embeddings

# References

## Original Research
- [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) - Original HyDE paper from CMU
- [HyDE Technical Report](https://boston.lti.cs.cmu.edu/luyug/HyDE/HyDE.pdf) - Detailed methodology

## Implementation Guides
- [Chitika - HyDE Query Expansion](https://www.chitika.com/hyde-query-expansion-rag/) - Comprehensive overview
- [AutoRAG HyDE Documentation](https://docs.auto-rag.com/nodes/query_expansion/hyde.html) - Production implementation patterns

## Related Techniques
- [Multi-Query RAG](https://python.langchain.com/docs/how_to/multi_query/) - Alternative query expansion approach
- [Query2Doc](https://arxiv.org/abs/2303.07678) - Similar concept using document generation

## Production RAG
- [LlamaIndex Query Transformations](https://docs.llamaindex.ai/en/stable/examples/query_transformations/HyDEQueryTransformDemo/) - HyDE in LlamaIndex
- [Haystack Query Expansion](https://haystack.deepset.ai/blog/extracting-metadata-filter) - Query enhancement patterns
- [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/) - RAG best practices