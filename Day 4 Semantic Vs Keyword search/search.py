import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_documents():
    return [
        "Deep learning models can understand patterns in data.",
        "Neural networks are powerful tools for machine learning.",
        "Football is a popular sport played worldwide.",
        "The game of soccer requires stamina and teamwork.",
        "Artificial intelligence enables semantic search capabilities."
    ]


def keyword_search(query, documents):
    """
    Implements a simple TF-IDF keyword search.
    """
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)
    query_vector = vectorizer.transform([query])

    scores = (doc_vectors @ query_vector.T).toarray().flatten()
    ranked_indices = np.argsort(scores)[::-1]

    return [(documents[i], scores[i]) for i in ranked_indices]


def semantic_search(query, documents):
    """
    Performs semantic search using SentenceTransformer embeddings.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    doc_vecs = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    scores = cosine_similarity(query_vec, doc_vecs)[0]
    ranked_indices = np.argsort(scores)[::-1]

    return [(documents[i], scores[i]) for i in ranked_indices]


def display_results(title, results, top_k=3):
    print(f"\n{title}")
    print("-" * len(title))

    for doc, score in results[:top_k]:
        print(f"Score: {score:.4f}  |  {doc}")


def main():
    documents = load_documents()

    query = "How do machines learn patterns?"

    keyword_results = keyword_search(query, documents)
    semantic_results = semantic_search(query, documents)

    print(f"\nQuery: {query}")

    display_results("Keyword Search Results", keyword_results)
    display_results("Semantic Search Results", semantic_results)


if __name__ == "__main__":
    main()