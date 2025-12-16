from rank_bm25 import BM25Okapi

def bm25_scores(query: str, documents: list[str]) -> list[float]:
    """
    Computes BM25 lexical similarity scores.
    Intended for hybrid retrieval demonstrations.
    """
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25.get_scores(query.lower().split())
