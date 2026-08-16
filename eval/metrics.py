

import math


def precision_at_k(retrieved, relevant, k):
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for s in top if s in relevant)
    return hits / len(top)


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for s in top if s in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved, relevant, k):
    for rank, s in enumerate(retrieved[:k], start=1):
        if s in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved, relevant, k):
    top = retrieved[:k]
    dcg = sum(
        (1.0 if s in relevant else 0.0) / math.log2(rank + 1)
        for rank, s in enumerate(top, start=1)
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def score_query(retrieved, relevant, k):
    relevant = set(relevant)
    return {
        "precision": precision_at_k(retrieved, relevant, k),
        "recall": recall_at_k(retrieved, relevant, k),
        "mrr": reciprocal_rank(retrieved, relevant, k),
        "ndcg": ndcg_at_k(retrieved, relevant, k),
    }


def average_scores(per_query_scores):
    if not per_query_scores:
        return {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    keys = per_query_scores[0].keys()
    return {k: sum(s[k] for s in per_query_scores) / len(per_query_scores) for k in keys}
