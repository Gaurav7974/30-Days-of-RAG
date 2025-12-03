import faiss
import numpy as np


class VectorStore:
    """
    Minimal wrapper around a FAISS IndexFlatL2 index.
    Encapsulates adding vectors, searching, and serialization.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.data = []  # stores the original text chunks

    def add(self, vectors: np.ndarray, chunks: list):
        """
        Adds vectors and stores their associated text.
        The number of vectors must match the number of chunks.
        """
        if vectors.shape[0] != len(chunks):
            raise ValueError("Vector count must match chunk count")

        self.index.add(vectors.astype(np.float32))
        self.data.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int = 3):
        """
        Searches for the k nearest neighbors of the query vector.
        Returns a list of (chunk_text, distance).
        """
        distances, indices = self.index.search(query_vec.astype(np.float32), k)

        results = []
        for i, d in zip(indices[0], distances[0]):
            if i < len(self.data):
                results.append((self.data[i], float(d)))

        return results

    def save(self, path: str):
        faiss.write_index(self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)
