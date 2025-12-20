import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from docs import documents

# Use PersistentClient to save data to disk
client = chromadb.PersistentClient(path="./chroma_db")

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

class SentenceTransformerEmbedding(embedding_functions.EmbeddingFunction):
    def __init__(self):
        pass
    
    def __call__(self, input):
        return model.encode(input).tolist()

COLLECTION_NAME = "fusion_demo"

# Delete existing collection to start fresh
try:
    client.delete_collection(COLLECTION_NAME)
    print("Deleted old collection")
except Exception:
    pass

# Create collection with embedding function
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=SentenceTransformerEmbedding()
)

# Index all documents
collection.add(
    ids=[d["id"] for d in documents],
    documents=[d["text"] for d in documents]
)

print(f"Indexed {len(documents)} documents")

# Verify indexing
count = collection.count()
print(f"Collection now has {count} documents")