import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bm25_utils import bm25_scores


def load_documents():
    return [
        "Transformers allow deep models to learn contextual representations.",
        "Neural networks can detect patterns in complex datasets.",
        "Python is widely used for machine learning, scripting, and data analysis.",
        "Football involves coordinated teamwork and physical endurance.",
        "Soccer players must maintain agility, stamina, and ball control.",
        "Large language models enable semantic search over large corpora.",
        "Machine learning engineers often use PyTorch or TensorFlow.",
        "The sport of football has different rules depending on the region."
    ]


def semantic_scores(query, documents, model):
    doc_vecs = model.encode(documents, normalize_embeddings=True)
    q_vec = model.encode([query], normalize_embeddings=True)
    return cosine_similarity(q_vec, doc_vecs)[0]


def normalize(x):
    if np.max(x) == 0:
        return x
    return x / np.max(x)


def combine_linear(bm25, sem, w1=0.4, w2=0.6):
    return normalize(bm25) * w1 + normalize(sem) * w2


def combine_harmonic(bm25, sem):
    bm25_n = normalize(bm25)
    sem_n = normalize(sem)
    return 2 * (bm25_n * sem_n) / (bm25_n + sem_n + 1e-9)


def combine_max(bm25, sem):
    return np.maximum(normalize(bm25), normalize(sem))


def multi_query_expansion(query, model):
    """
    Expands the query into multiple reformulations.
    This is a lightweight version of real multi-query retrieval.
    """
    expansions = [
        query,
        f"{query} meaning",
        f"{query} explanation",
        f"{query} related concepts",
    ]
    return expansions


def hybrid_two_pass(query, documents, model):
    # Stage 1: hybrid scoring
    bm25 = bm25_scores(query, documents)
    sem = semantic_scores(query, documents, model)
    hybrid = combine_linear(bm25, sem)

    # Select top candidates
    top_idx = np.argsort(hybrid)[::-1][:5]
    candidates = [documents[i] for i in top_idx]

    # Stage 2: re-rank using semantic-only similarity
    re_scores = semantic_scores(query, candidates, model)
    final_idx = np.argsort(re_scores)[::-1]

    return [(candidates[i], re_scores[i]) for i in final_idx]


def show(title, docs, scores):
    print(f"\n{title}")
    print("-" * len(title))
    idx = np.argsort(scores)[::-1]
    for i in idx[:3]:
        print(f"{scores[i]:.4f} | {docs[i]}")


def main():
    documents = load_documents()
    query = "machine learning with python"

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # BM25
    bm25 = bm25_scores(query, documents)
    show("BM25 Results", documents, bm25)

    # Semantic search
    sem = semantic_scores(query, documents, model)
    show("Semantic Results", documents, sem)

    # Hybrid scoring: linear combination
    hybrid_lin = combine_linear(bm25, sem)
    show("Hybrid Results (Linear)", documents, hybrid_lin)

    # Hybrid scoring: harmonic mean
    hybrid_hm = combine_harmonic(bm25, sem)
    show("Hybrid Results (Harmonic Mean)", documents, hybrid_hm)

    # Hybrid scoring: max-blend
    hybrid_mx = combine_max(bm25, sem)
    show("Hybrid Results (Max Blend)", documents, hybrid_mx)

    # Two-pass retrieval
    final = hybrid_two_pass(query, documents, model)

    print("\nTwo-Pass Reranked Results")
    for text, score in final:
        print(f"{score:.4f} | {text}")


if __name__ == "__main__":
    main()
