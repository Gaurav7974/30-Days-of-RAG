
# Sentence Window Retrieval for RAG

## What This Does

When answering questions from long documents, you need highly relevant text but also enough surrounding context so the model understands what that sentence actually means [web:1][web:2].

Sentence Window Retrieval solves this by:
- Searching at sentence level for maximum precision
- Expanding the matched sentence with neighboring sentences (a "window") to provide context

Think of it like this:
- You search for a single sentence that best matches your question
- But you also read the few sentences around it so nothing is taken out of context

## The Problem It Solves

Traditional RAG chunking fails in two ways [web:2][web:8]:

**Small chunks (single sentence / very small text)**
- Very precise
- But missing surrounding meaning
- Leads to wrong or hallucinated answers

**Large chunks (500–1500 tokens)**
- Lots of context
- But low retrieval accuracy
- Slow and often returns irrelevant noise

Sentence Window Retrieval gives you both [web:2]:
- Sentence-level accuracy
- Local contextual understanding

## Sentence Window Retrieval Process

The process of **Sentence Window Retrieval** progresses through the following stages:

## 1. Tokenization
The document or corpus is tokenized into individual sentences or segments to prepare it for retrieval and generation.

## 2. Query Formation
A query or keyword is used to search for relevant information within the document. This query guides the retrieval and generation process.

## 3. Window Selection
A window of sentences is selected around the query to capture the relevant context. The size of the window is determined based on the specific requirements and can be adjusted to include more or fewer sentences.

## 4. Scoring and Ranking
The selected sentences within the window are scored based on their relevance to the query using **RAG’s retrieval and ranking mechanisms**. This may involve leveraging pre-trained language models and fine-tuning them for retrieval tasks.

## 5. Retrieval and Generation
The top-ranked sentences are retrieved and used as context for generation. **RAG** then generates responses or summaries based on the retrieved context, providing relevant and coherent outputs.

## Quick Start

### 1. Install Dependencies

Activate your venv and install requirements:
``` bash 
pip install llama-index sentence-transformers
```


If using OpenRouter / OpenAI models, ensure your key is set:
``` bash 
setx OPENROUTER_API_KEY "your_key_here" # Windows
```

### 2. Run It

From inside the folder:
``` bash 
python main.py
```
### 3. What You Will See

Interactive query mode starts:
- Query: What is sentence window retrieval?

- Response: ...

It will:
- Build an index (or load an existing one)
- Retrieve best matching sentences
- Expand context
- Generate grounded response

## Architecture

documents.py → Your source content
main.py → Index creation + query engine
sentence_index_* → Persisted sentence-window indexes
.cache/ → HuggingFace cache

### Key Components Inside Code

- `SentenceWindowNodeParser`
- `HuggingFaceEmbedding`
- `MetadataReplacementPostProcessor`
- `SentenceTransformerRerank`
- `VectorStoreIndex`

## Configuration

### Window Size

Controls how many sentences to include around the matched one [web:5].

In `build_sentence_window_index()`:
sentence_window_size=3


Meaning:
- 1 sentence before
- matched sentence
- 1 sentence after

Increase if your document writing style is long-winded.

### Retrieval Settings

In query engine:
similarity_top_k=6
rerank_top_n=2



- `similarity_top_k`: how many candidates to retrieve
- `rerank_top_n`: how many best ones to keep after reranking

## When to Use Sentence Window Retrieval

### Best For
- Technical documentation
- Research papers
- Legal text
- Manuals
- Long structured documents
- When precision matters but context still matters

### Not Ideal For
- Very short documents
- Pure fact lookup
- Cases requiring full-document reasoning
- Tasks needing extremely long-range context

## Performance Characteristics

### Pros
- Very accurate retrieval [web:8]
- Better grounding (38.2% improvement) [web:8]
- Lower hallucinations
- Efficient token usage compared to giant chunks

### Cons
- Slightly more compute than naive chunking
- Requires preprocessing into sentence units
- Still limited to "local" context windows [web:7]

## Example Output Behavior

Query: Why can small chunks fail in RAG?

- System retrieves:
best relevant sentence
sentences before and after

- Result:
More coherent, less fragmented responses.



Sentence window retrieval improves Answer Relevance by 22.7% and Groundedness by 38.2% compared to basic RAG [web:8].

## Learn More

**Concept and explanations:**
- [Building and Evaluating Advanced RAG](https://velog.io/@jjlee6496/Building-and-Evaluating-Advanced-RAG-1)
- [Haystack Tutorial - Sentence Window Retriever](https://haystack.deepset.ai/tutorials/42_sentence_window_retriever)
- [LlamaIndex Implementation Guide](https://medium.com/@govindarajpriyanthan/advanced-rag-building-and-evaluating-a-sentence-window-retriever-setup-using-llamaindex-and-67bcab2d241e)

**Research:**
- Sentence Windowing in ARAGOG (Advanced RAG)
 [arXiv:2404.01037](https://arxiv.org/pdf/2404.01037.pdf)

**General RAG:**
- [Retrieval-Augmented Generation - Wikipedia](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)
