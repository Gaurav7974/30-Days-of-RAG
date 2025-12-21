# PARENT DOCUMENTS
DOCUMENTS = {
    "parent_1": """
Retrieval Augmented Generation systems often fail because small chunks lack context.
Parent child chunking solves this by retrieving small chunks but returning larger sections.
This improves answer coherence while keeping retrieval precise.
The key insight is separating what you search from what you show the LLM.
Small chunks are good for matching queries but bad for generating complete answers.
By retrieving precise children and returning contextual parents, you get the best of both worlds.
""",
    
    "parent_2": """
Vector databases enable semantic search over text embeddings.
Traditional keyword search misses semantic similarity between concepts.
Hybrid search combines both approaches for better recall.
Most production RAG systems use vector databases like Pinecone or Weaviate.
Embeddings capture meaning in high-dimensional space, allowing similarity matching.
This semantic understanding is crucial for modern information retrieval systems.
""",
    
    "parent_3": """
Reranking improves retrieval by scoring candidates with cross-encoders.
Initial retrieval uses fast bi-encoders for broad recall.
Reranking then applies slower but more accurate models to top results.
This two-stage approach balances speed and quality.
Cross-encoders can see both query and document together, enabling better relevance scoring.
The computational cost is justified by the dramatic improvement in precision.
""",
    
    "parent_4": """
Transformer architectures revolutionized natural language processing.
Self-attention mechanisms allow models to weigh token importance dynamically.
Multi-head attention enables parallel processing of different representation subspaces.
The encoder-decoder structure supports sequence-to-sequence tasks effectively.
Positional encodings provide sequential information without recurrence.
These innovations led to breakthrough models like BERT, GPT, and T5.
""",
    
    "parent_5": """
Prompt engineering is critical for extracting good outputs from language models.
Clear instructions with examples dramatically improve response quality.
Few-shot learning allows models to adapt to new tasks without fine-tuning.
Chain-of-thought prompting encourages step-by-step reasoning.
The format and structure of prompts significantly impact model behavior.
Iterative refinement of prompts is often necessary for production systems.
"""
}

# CHILD CHUNKS (Manually defined for demo clarity)
PARENT_CHILDREN = {
    "parent_1": [
        "RAG systems often fail because small chunks lack context.",
        "Parent child chunking retrieves small chunks but returns larger sections.",
        "This improves answer coherence without hurting retrieval.",
        "The key is separating what you search from what you show the LLM.",
        "Small chunks are good for matching but bad for generating answers.",
        "Retrieving children and returning parents gives the best of both worlds."
    ],
    
    "parent_2": [
        "Vector databases enable semantic search over text embeddings.",
        "Traditional keyword search misses semantic similarity.",
        "Hybrid search combines both approaches for better recall.",
        "Production RAG systems use vector databases like Pinecone.",
        "Embeddings capture meaning in high-dimensional space.",
        "Semantic understanding is crucial for information retrieval."
    ],
    
    "parent_3": [
        "Reranking improves retrieval with cross-encoders.",
        "Initial retrieval uses fast bi-encoders for broad recall.",
        "Reranking applies slower accurate models to top results.",
        "Two-stage approach balances speed and quality.",
        "Cross-encoders see query and document together for better scoring.",
        "Computational cost is justified by precision improvement."
    ],
    
    "parent_4": [
        "Transformers revolutionized natural language processing.",
        "Self-attention mechanisms weigh token importance dynamically.",
        "Multi-head attention enables parallel processing of subspaces.",
        "Encoder-decoder structure supports sequence-to-sequence tasks.",
        "Positional encodings provide sequential information without recurrence.",
        "These innovations led to BERT, GPT, and T5."
    ],
    
    "parent_5": [
        "Prompt engineering is critical for good LLM outputs.",
        "Clear instructions with examples improve response quality.",
        "Few-shot learning adapts to new tasks without fine-tuning.",
        "Chain-of-thought prompting encourages step-by-step reasoning.",
        "Prompt format significantly impacts model behavior.",
        "Iterative refinement is necessary for production systems."
    ]
}

#Sample queries
SAMPLE_QUERIES = [
    "Why do RAG systems fail?",
    "How does vector search work?",
    "What is reranking in RAG?",
    "Explain transformer architecture",
    "How to write better prompts?",
    "What is semantic search?",
    "How does parent-child chunking help?",
    "What are cross-encoders used for?"
]