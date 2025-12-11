from rank_bm25 import BM25Okapi


def bm25_scores(query, documents):
    """
    Computes BM25 scores for a list of documents.
    """
    tokenized = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized)
    return bm25.get_scores(query.lower().split())