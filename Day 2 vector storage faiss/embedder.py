from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    """
    Wraps a SentenceTransformer model and exposes a simple encode method.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        Encodes a list of strings into vectors.
        Returns a numpy array of shape (n, dim).
        """
        vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        return vectors.astype(np.float32)
