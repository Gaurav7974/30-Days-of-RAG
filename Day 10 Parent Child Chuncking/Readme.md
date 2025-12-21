# Parent-Child Chunking for RAG

## What This Does

When you search documents, you want to find specific information quickly. But when you generate answers, you need enough context to make sense. Parent-child chunking solves this problem by **searching small pieces** but **returning larger sections**.

Think of it like this:
- You search through chapter summaries (children) to find what you need
- But you read the full chapter (parent) to understand it properly

## The Problem It Solves

Traditional RAG systems face a dilemma:

**Small chunks (200-500 tokens):**
-  Find specific information easily
-  Missing surrounding context
-  Answers feel incomplete or "choppy"

**Large chunks (1000+ tokens):**
-  Plenty of context
-  Hard to find exact information
-  Slow and noisy retrieval

**Parent-child chunking gives you both:**
- Search precision from small chunks
- Answer quality from large chunks

## How It Works

```
Document → Split into parents (large sections)
         → Each parent → Split into children (small pieces)
         
When you search:
1. Find relevant children (precise matching)
2. Look up their parent (rich context)
3. Return the parent to the language model
```

## Quick Start

### Installation

```bash
pip install chromadb sentence-transformers
```

### Run It

```bash
python main.py
```

### What You'll See

```
> Why do RAG systems fail?

[1] Score: 0.8367 | Matched: "RAG systems often fail because small chunks lack context."
[Full parent document with complete context...]
```

## Architecture

```
documents.py  → Your text documents and their child chunks
main.py       → Retrieval system that searches children, returns parents
```

**Key components:**
- `index_documents()` - Creates child-parent mappings
- `retrieve()` - Searches children, expands to parents
- `search()` - User-facing query interface

## Configuration

You can adjust these parameters in `retrieve()`:

```python
k=5          # Number of child chunks to search (default: 5)
threshold=1.0  # Relevance cutoff - lower is stricter (default: 1.0)
```

In `search()`:

```python
top_k=3  # Final number of parent chunks returned (default: 3)
```

## When to Use Parent-Child Chunking

 **Best for:**
- Technical documentation with clear sections
- Long-form content where context matters
- Questions that need complete explanations
- Reducing "choppy" or fragmented answers

 **Not ideal for:**
- Very short documents (no benefit from hierarchy)
- Simple fact lookup (single sentences are enough)
- Extremely large parents (>3000 tokens causes token waste)

## How Results Are Scored

The system converts vector distances to similarity scores:

```python
score = 1 - distance
```

- **1.0** = Perfect match
- **0.8+** = Highly relevant
- **0.5-0.8** = Moderately relevant
- **<0.5** = Weakly relevant

If no results score above the threshold, you'll get an error suggesting a different query.

## Performance Characteristics

**Pros:**
- Better answer coherence without losing retrieval precision
- Automatic deduplication (multiple children from same parent)
- Memory efficient (only children are embedded)

**Cons:**
- Slightly more storage (need both parent and child chunks)
- More complex indexing logic
- Returns larger context (higher token costs for LLMs)

**Speed:**
- Indexing: ~Same as regular chunking
- Retrieval: ~Same speed (still searching children)
- Generation: Slower (more tokens to process)

## Customizing Your Documents

Edit `documents.py` to add your own content:

```python
DOCUMENTS = {
    "parent_1": "Your large section of text here...",
    "parent_2": "Another section...",
}

PARENT_CHILDREN = {
    "parent_1": [
        "First small chunk",
        "Second small chunk",
    ],
}
```

Or load from files, PDFs, databases - whatever works for your use case.

## Learn More

- [LangChain Parent Document Retriever](https://python.langchain.com/docs/how_to/parent_document_retriever/) - Official implementation
- [Advanced RAG Techniques](https://arxiv.org/abs/2312.10997) - Comprehensive survey paper
- [ChromaDB Documentation](https://docs.trychroma.com/) - Vector database guide
- [Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/) - Different approaches explained