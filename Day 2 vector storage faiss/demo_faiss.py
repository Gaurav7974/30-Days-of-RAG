import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from chunkers import sentence_chunk
from embedder import Embedder
from faiss_store import VectorStore
import numpy as np


def load_sample():
    return """
Vector search plays a critical role in modern retrieval systems.
Embeddings allow us to map text into a numerical space where similar ideas
are close together. Storing these embeddings efficiently ensures fast
and accurate search for relevant information.

FAISS is one of the most widely used libraries for similarity search.
It supports both exact and approximate indices and is optimized for large datasets.
"""


def main():
    text = load_sample()

    # Step 1: Chunking
    chunks = sentence_chunk(text, max_chars=200)

    # Step 2: Embeddings
    embedder = Embedder()
    vectors = embedder.encode(chunks)

    # Step 3: Vector store
    store = VectorStore(dim=vectors.shape[1])
    store.add(vectors, chunks)

    # Step 4: Query
    query = "How is fast similarity search done?"
    query_vec = embedder.encode([query])

    results = store.search(query_vec, k=3)

    print("\nQuery:", query)
    print("\nTop Matches:\n")

    for chunk, dist in results:
        print(f"Distance: {dist:.4f}")
        print(chunk)
        print("---")


if __name__ == "__main__":
    main()
