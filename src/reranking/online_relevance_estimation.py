from typing import List, Dict, Tuple, Callable, Optional
import numpy as np


class OnlineRelevanceEstimation:
    def __init__(
        self,
        documents: List[str],
        initial_scores: Dict[int, float],
        rerank_model: Optional[Callable] = None,
        batch_size: int = 10,
        exploration_factor: float = 0.1
    ):
        self.documents = documents
        self.current_scores = initial_scores.copy()
        self.rerank_model = rerank_model
        self.batch_size = batch_size
        self.exploration_factor = exploration_factor
        self.reranked_docs = set()
        self.rerank_history = []
    
    def select_batch(self, query: str) -> List[int]:
        ucb_scores = {}
        
        max_score = max(self.current_scores.values()) if self.current_scores.values() else 1.0
        min_score = min(self.current_scores.values()) if self.current_scores.values() else 0.0
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        for doc_idx in self.current_scores.keys():
            score = self.current_scores[doc_idx]
            normalized_score = (score - min_score) / score_range if score_range > 0 else score
            
            if doc_idx in self.reranked_docs:
                uncertainty = 0.0
            else:
                uncertainty = self.exploration_factor * (1.0 - normalized_score)
            
            ucb_scores[doc_idx] = normalized_score + uncertainty
        
        sorted_docs = sorted(
            ucb_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [doc_idx for doc_idx, _ in sorted_docs[:self.batch_size]]
    
    def update_scores(self, doc_indices: List[int], new_scores: Dict[int, float]):
        old_scores_before_update = {}
        for doc_idx in doc_indices:
            if doc_idx in new_scores:
                old_scores_before_update[doc_idx] = self.current_scores.get(doc_idx, 0.0)
                self.current_scores[doc_idx] = new_scores[doc_idx]
                self.reranked_docs.add(doc_idx)
        
        self._propagate_scores(doc_indices, new_scores, old_scores_before_update)
    
    def _propagate_scores(self, reranked_indices: List[int], new_scores: Dict[int, float], old_scores: Dict[int, float]):
        positive_improvements = []
        for doc_idx in reranked_indices:
            if doc_idx in new_scores and doc_idx in old_scores:
                improvement = new_scores[doc_idx] - old_scores[doc_idx]
                if improvement > 0:
                    positive_improvements.append(improvement)
        
        if positive_improvements:
            avg_improvement = np.mean(positive_improvements)
            for doc_idx in self.current_scores.keys():
                if doc_idx not in self.reranked_docs:
                    self.current_scores[doc_idx] += avg_improvement * 0.03
    
    def rerank(self, query: str, budget: int = 100, verbose: bool = False) -> List[Tuple[int, float]]:
        iterations = 0
        total_reranked = 0
        
        if verbose:
            print(f"Starting ORE with budget: {budget}")
        
        while total_reranked < budget:
            batch = self.select_batch(query)
            remaining_budget = budget - total_reranked
            
            if remaining_budget <= 0 or not batch:
                break
            
            actual_batch_size = min(len(batch), remaining_budget)
            batch = batch[:actual_batch_size]
            
            if self.rerank_model:
                new_scores = self.rerank_model(query, batch)
            else:
                new_scores = {idx: self.current_scores[idx] for idx in batch}
            
            self.update_scores(batch, new_scores)
            
            total_reranked += len(batch)
            iterations += 1
            
            if verbose or iterations % 5 == 0:
                import sys
                print(f"    Iteration {iterations}: Re-ranked {len(batch)} docs "
                      f"(Total: {total_reranked}/{budget})", end='\r')
                sys.stdout.flush()
        
        final_ranking = sorted(
            self.current_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        if verbose or iterations > 0:
            import sys
            print(f"\n    ORE completed: {total_reranked} documents re-ranked in {iterations} iterations")
            sys.stdout.flush()
        
        return final_ranking
    
    def get_document(self, index: int) -> str:
        return self.documents[index]

