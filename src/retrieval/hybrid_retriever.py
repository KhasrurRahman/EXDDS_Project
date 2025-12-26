from typing import List, Tuple
from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever


class HybridRetriever:
    def __init__(self, documents: List[str], alpha: float = 0.5):
        self.alpha = alpha
        self.bm25_retriever = BM25Retriever(documents)
        self.dense_retriever = DenseRetriever(documents)
        self.documents = documents
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k * 2)
        dense_results = self.dense_retriever.retrieve(query, top_k=top_k * 2)
        
        bm25_scores = {idx: score for idx, score in bm25_results}
        dense_scores = {idx: score for idx, score in dense_results}
        
        if bm25_scores:
            max_bm25 = max(bm25_scores.values())
            min_bm25 = min(bm25_scores.values())
            if max_bm25 > min_bm25:
                bm25_scores = {
                    idx: (score - min_bm25) / (max_bm25 - min_bm25)
                    for idx, score in bm25_scores.items()
                }
        
        if dense_scores:
            max_dense = max(dense_scores.values())
            min_dense = min(dense_scores.values())
            if max_dense > min_dense:
                dense_scores = {
                    idx: (score - min_dense) / (max_dense - min_dense)
                    for idx, score in dense_scores.items()
                }
        
        combined_scores = {}
        all_indices = set(bm25_scores.keys()) | set(dense_scores.keys())
        
        for idx in all_indices:
            bm25_score = bm25_scores.get(idx, 0.0)
            dense_score = dense_scores.get(idx, 0.0)
            combined_scores[idx] = self.alpha * bm25_score + (1 - self.alpha) * dense_score
        
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return sorted_results
    
    def get_document(self, index: int) -> str:
        return self.documents[index]

