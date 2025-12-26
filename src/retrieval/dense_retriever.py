from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple


class DenseRetriever:
    def __init__(self, documents: List[str], model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        print(f"Encoding {len(documents):,} documents...")
        import sys
        sys.stdout.flush()
        self.doc_embeddings = self.model.encode(
            documents, 
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )
        print("Encoding complete!")
        sys.stdout.flush()
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        similarities = np.dot(self.doc_embeddings, query_embedding) / (
            np.linalg.norm(self.doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def get_document(self, index: int) -> str:
        return self.documents[index]

