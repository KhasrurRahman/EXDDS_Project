from rank_bm25 import BM25Okapi
from typing import List, Tuple


class BM25Retriever:
    def __init__(self, documents: List[str]):
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents
        
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        top_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:top_k]
        
        return [(idx, scores[idx]) for idx in top_indices]
    
    def get_document(self, index: int) -> str:
        return self.documents[index]

