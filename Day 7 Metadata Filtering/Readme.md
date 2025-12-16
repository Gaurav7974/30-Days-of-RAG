# Query-Time Metadata Filtering for RAG 

## Overview

This project implements query-time metadata extraction to improve retrieval quality in Retrieval-Augmented Generation (RAG) systems.  

Instead of relying only on embeddings or keyword similarity, the system extracts structured metadata (for example, company, year, disease) from the user query and converts it into hard retrieval filters applied before ranking.

This design mirrors how production RAG stacks such as Elastic, Azure AI Search, Pinecone, and Qdrant enforce structured constraints on top of semantic search.

## Why Metadata Filtering Matters

Pure semantic similarity often returns documents that are conceptually related but factually misaligned with the query (for example, wrong year or wrong entity).  Query-time metadata filtering restricts retrieval to documents whose metadata exactly matches the constraints expressed in the query, improving factual accuracy and stability.

**Example**

- Query: “What was Nvidia’s revenue in 2022?”
- Without metadata filters:
    - Documents from other years (for example, 2020) may rank highly.
    - Results depend heavily on embedding behavior and training data.
- With metadata filters:
    - Only documents where `company = "nvidia"` and `year = 2022` are retrieved.
    - Ranking operates on a small, accurate candidate set.

This leads to higher factual accuracy, reduced retrieval cost, and more predictable generation behavior.

## High‑Level Architecture

The retrieval flow is:

1. **User Query** – Natural language question.
2. **Metadata Extraction (LLM)** – Extracts only allowed metadata fields from the query.
3. **Structured Filters** – JSON filter object describing constraints.
4. **Filter‑First Retrieval** – Retriever searches only documents satisfying the filters.
5. **Ranking (BM25 / Vector)** – Scores documents within the filtered set.
6. **Context for Generation** – Top documents are passed to the generator (outside this project’s scope).

The LLM is used strictly as a structured parser; retrieval logic remains deterministic and filter‑driven.


## Metadata Filter Format

Extracted metadata is converted into a standard logical filter:

```bash
{
  "operator": "AND",
  "conditions": [
    { "field": "meta.company", "operator": "==", "value": "nvidia" },
    { "field": "meta.year", "operator": "==", "value": 2022 }
  ]
}
```

This structure is compatible with Haystack retrievers and maps naturally onto Qdrant, Pinecone, and Elasticsearch metadata filters.


## Metadata Filtering vs Embeddings

In practice, metadata filters narrow the candidate set, and BM25 or embedding-based retrievers rank what remains.


| Aspect | Embeddings Only | Metadata Filtering |
| :-- | :-- | :-- |
| Constraint type | Soft (similarity-based) | Hard (exact match) |
| Accuracy | Approximate | Precise on constrained axes |
| Search space | Large | Strongly reduced |
| Failure mode | Semantically related, wrong facts | Empty result (safe) |
| Debuggability | Hard to inspect scores | Easy to inspect filters |

## Installation

Create a virtual environment and install the minimal dependencies:
```bash
pip install haystack-ai
pip install transformers torch
```
- `haystack-ai` for the pipeline and retrievers.
- `transformers` and `torch` for the local LLM backend.

## Quick Start 
Run the test pipeline:
```bash
python test.py
```
## Design Decisions and RAG Alignment

Design choices emphasize correctness, robustness, and production alignment:

- **Small LLM as parser**: Treat the model as a deterministic schema-bound parser, not as a retrieval or reasoning engine.
- **Optional filters**: If no metadata is found, the system falls back to standard retrieval.

## When to Use This Pattern

Query-time metadata filtering is especially useful when:

- Queries mention explicit attributes such as time ranges, entities, or categories.
- Factual correctness and consistency are critical.
- Predictable, debuggable retrieval is required (for example, analytics, compliance, financial QA).

It is less suitable when:

- Queries are open-ended or exploratory.
- There is no meaningful structured metadata attached to documents.

In such cases, pure semantic or hybrid retrieval may be sufficient without additional metadata constraints.

# Read more...
## Haystack – Metadata Filtering & Query-Time Extraction
* https://docs.haystack.deepset.ai/docs/metadata-filtering
* https://haystack.deepset.ai/cookbook/extracting_metadata_filters_from_a_user_query
* https://haystack.deepset.ai/blog/extracting-metadata-filter
* https://docs.haystacksearch.org/en/v2.0.0/search.html

## Practical Guides & Blogs
* https://davidsbatista.net/blog/2024/05/13/Extracting-Metadata-Haystack/

## Milvus – Advanced Filtering Concepts
* https://milvus.io/ai-quick-reference/how-do-i-implement-advanced-filtering-in-haystack-queries
* https://milvus.io/ai-quick-reference/how-do-i-add-additional-filters-or-constraints-to-search-queries-in-haystack

## Prompt Engineering & Research
* https://knowledge.dataiku.com/latest/gen-ai/text-processing/tutorial-prompt-engineering.html
* https://arxiv.org/html/2409.00847v3