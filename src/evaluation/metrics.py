from typing import List, Set
import numpy as np


def calculate_recall(ranked_docs: List[int], relevant_docs: Set[int], k: int = 10) -> float:
    if not relevant_docs:
        return 0.0
    
    top_k_docs = set(ranked_docs[:k])
    relevant_found = len(top_k_docs & relevant_docs)
    return relevant_found / len(relevant_docs)


def calculate_ndcg(ranked_docs: List[int], relevant_docs: Set[int], k: int = 10) -> float:
    if not relevant_docs:
        return 0.0
    
    dcg = 0.0
    for i, doc_idx in enumerate(ranked_docs[:k], start=1):
        if doc_idx in relevant_docs:
            dcg += 1.0 / np.log2(i + 1)
    
    idcg = 0.0
    num_relevant = min(len(relevant_docs), k)
    for i in range(1, num_relevant + 1):
        idcg += 1.0 / np.log2(i + 1)
    
    return dcg / idcg if idcg > 0 else 0.0


def calculate_precision(ranked_docs: List[int], relevant_docs: Set[int], k: int = 10) -> float:
    top_k_docs = set(ranked_docs[:k])
    relevant_found = len(top_k_docs & relevant_docs)
    return relevant_found / k

