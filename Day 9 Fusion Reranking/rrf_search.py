from collections import defaultdict
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# Use PersistentClient to load data from disk
client = chromadb.PersistentClient(path="./chroma_db")
model = SentenceTransformer("all-MiniLM-L6-v2")

class SentenceTransformerEmbedding(embedding_functions.EmbeddingFunction):
    def __init__(self):
        pass
    
    def __call__(self, input):
        return model.encode(input).tolist()

COLLECTION_NAME = "fusion_demo"

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=SentenceTransformerEmbedding()
)

# Check if collection has documents
print(f"Collection has {collection.count()} documents")


def vector_search(query, k):
    # Semantic search using embeddings
    res = collection.query(query_texts=[query], n_results=k)
    return res["ids"][0]


def keyword_search(query, k):
    # Simple keyword matching
    all_docs = collection.get()
    query_terms = query.lower().split()
    
    scores = []
    for i, doc in enumerate(all_docs["documents"]):
        # Count matching terms
        score = sum(1 for term in query_terms if term in doc.lower())
        if score > 0:
            scores.append((all_docs["ids"][i], score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scores[:k]]


def rrf_fusion(rank_lists, k=60):
    # Reciprocal Rank Fusion: score = sum(1 / (k + rank))
    scores = defaultdict(float)
    
    for ranked_list in rank_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    
    # Sort by score descending
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in fused]


def fused_retrieval(query, top_k_each=5, final_k=5):
    # Get results from both methods
    vec_ids = vector_search(query, top_k_each)
    kw_ids = keyword_search(query, top_k_each)
    
    # Fuse with RRF
    fused_ids = rrf_fusion([vec_ids, kw_ids])
    
    return fused_ids[:final_k]


if __name__ == "__main__":
    result = fused_retrieval(
        query="what is fusion in rag pipelines",
        top_k_each=5,
        final_k=5
    )
    
    print(f"Top {len(result)} Results: {result}")