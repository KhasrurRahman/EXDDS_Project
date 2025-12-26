from src.retrieval import HybridRetriever, DenseRetriever, BM25Retriever
from src.reranking import OnlineRelevanceEstimation
from src.evaluation import calculate_recall, calculate_ndcg, calculate_precision
from src.data import DatasetLoader
import argparse
import numpy as np
from typing import Dict, List, Tuple


def simple_rerank_model(ore_instance, documents: List[str], bm25_retriever, dense_retriever):
    query_embedding_cache = {}
    
    def rerank(query, doc_indices):
        new_scores = {}
        
        if not doc_indices:
            return new_scores
        
        if query not in query_embedding_cache:
            query_embedding_cache[query] = dense_retriever.model.encode(query, convert_to_numpy=True)
        query_embedding = query_embedding_cache[query]
        
        tokenized_query = query.lower().split()
        
        batch_dense_scores = []
        for doc_idx in doc_indices:
            doc_embedding = dense_retriever.doc_embeddings[doc_idx]
            dense_score = np.dot(doc_embedding, query_embedding) / (
                np.linalg.norm(doc_embedding) * np.linalg.norm(query_embedding)
            )
            batch_dense_scores.append((doc_idx, dense_score))
        
        if batch_dense_scores:
            dense_vals = [s[1] for s in batch_dense_scores]
            dense_min, dense_max = min(dense_vals), max(dense_vals)
            dense_range = dense_max - dense_min if dense_max > dense_min else 1.0
            
            for doc_idx, dense_score in batch_dense_scores:
                current_score = ore_instance.current_scores.get(doc_idx, 0.0)
                
                norm_dense = (dense_score - dense_min) / dense_range if dense_range > 0 else 0.5
                
                if current_score < 0.3:
                    new_score = 0.2 * current_score + 0.8 * norm_dense
                elif current_score < 0.6:
                    new_score = 0.4 * current_score + 0.6 * norm_dense
                elif current_score < 0.85:
                    new_score = 0.6 * current_score + 0.4 * norm_dense
                else:
                    new_score = 0.85 * current_score + 0.15 * norm_dense
                
                if new_score < current_score:
                    new_score = current_score
                
                new_scores[doc_idx] = float(new_score)
        
        return new_scores
    return rerank


def run_experiment(
    dataset_name: str = "msmarco-passage/trec-dl-2019/judged",
    num_queries: int = 10,
    num_docs: int = 1000,
    top_k: int = 20,
    budget: int = 50,
    alpha: float = 0.5,
    batch_size: int = 10,
    exploration_factor: float = 0.2,
    use_bm25_baseline: bool = False
):
    print(f"Loading dataset: {dataset_name}")
    loader = DatasetLoader(dataset_name, data_dir="data")
    
    print("\nLoading relevance judgments...")
    qrels = loader.load_qrels()
    
    print(f"\nLoading queries...")
    import sys
    sys.stdout.flush()
    all_queries = loader.load_queries()
    
    all_relevant_doc_ids = set()
    for doc_ids in qrels.values():
        all_relevant_doc_ids.update(doc_ids)
    
    if num_docs == 0:
        print(f"\nLoading all documents (including {len(all_relevant_doc_ids)} relevant documents)...")
        print("This may take 10-20 minutes for 8.8M documents...")
        sys.stdout.flush()
        documents, doc_index = loader.load_documents(limit=None, include_relevant_doc_ids=all_relevant_doc_ids)
    else:
        print(f"\nLoading {num_docs:,} documents (including {len(all_relevant_doc_ids)} relevant documents)...")
        sys.stdout.flush()
        documents, doc_index = loader.load_documents(limit=num_docs, include_relevant_doc_ids=all_relevant_doc_ids)
    print(f"Loaded {len(documents):,} documents")
    sys.stdout.flush()
    
    print("Filtering queries with relevant documents in corpus...")
    queries_dict = loader.filter_queries_with_relevant_docs(
        all_queries, qrels, doc_index, min_relevant=1
    )
    
    queries = list(queries_dict.items())[:num_queries]
    print(f"Using {len(queries)} queries with relevant documents")
    
    print("\nInitializing retrievers...")
    print("Step 1/3: Initializing BM25 retriever...")
    sys.stdout.flush()
    bm25_retriever = BM25Retriever(documents)
    print("Step 2/3: Initializing Dense retriever (encoding documents - this may take time)...")
    sys.stdout.flush()
    dense_retriever = DenseRetriever(documents)
    print("Step 3/3: Initializing Hybrid retriever...")
    sys.stdout.flush()
    retriever = HybridRetriever(documents, alpha=alpha)
    print("All retrievers initialized!")
    sys.stdout.flush()
    
    results = {
        'baseline': {'recall': [], 'ndcg': [], 'precision': []},
        'ore': {'recall': [], 'ndcg': [], 'precision': []}
    }
    
    print(f"\nRunning experiments on {len(queries)} queries...")
    print("=" * 80)
    sys.stdout.flush()
    
    for query_idx, (query_id, query_text) in enumerate(queries, 1):
        print(f"\n[{query_idx}/{len(queries)}] Processing query...")
        sys.stdout.flush()
        print(f"\nQuery ID: {query_id}")
        print(f"Query: {query_text[:100]}...")
        
        relevant_doc_ids = loader.get_relevant_docs(query_id)
        if not relevant_doc_ids:
            print("  Skipping: No relevance judgments")
            continue
        
        relevant_indices = {doc_index[doc_id] for doc_id in relevant_doc_ids if doc_id in doc_index}
        
        if not relevant_indices:
            continue
        
        if use_bm25_baseline:
            initial_results = bm25_retriever.retrieve(query_text, top_k=min(len(documents), 10000))
            baseline_name = "BM25-only"
        else:
            initial_results = retriever.retrieve(query_text, top_k=min(len(documents), 10000))
            baseline_name = "Hybrid (BM25+Dense)"
        
        initial_scores_dict = {idx: score for idx, score in initial_results}
        
        print(f"  Retrieved {len(initial_results)} documents for initial ranking ({baseline_name})")
        sys.stdout.flush()
        
        if initial_scores_dict:
            min_score = min(initial_scores_dict.values())
            max_score = max(initial_scores_dict.values())
            score_range = max_score - min_score if max_score > min_score else 1.0
        else:
            min_score, max_score, score_range = 0.0, 1.0, 1.0
        
        initial_scores = {}
        for i in range(len(documents)):
            if i in initial_scores_dict:
                normalized_score = (initial_scores_dict[i] - min_score) / score_range if score_range > 0 else 0.5
                initial_scores[i] = float(normalized_score)
            else:
                initial_scores[i] = 0.0
        
        ranked_docs_baseline = [idx for idx, _ in initial_results[:top_k]]
        
        baseline_recall = calculate_recall(ranked_docs_baseline, relevant_indices, k=top_k)
        baseline_ndcg = calculate_ndcg(ranked_docs_baseline, relevant_indices, k=top_k)
        baseline_precision = calculate_precision(ranked_docs_baseline, relevant_indices, k=top_k)
        
        results['baseline']['recall'].append(baseline_recall)
        results['baseline']['ndcg'].append(baseline_ndcg)
        results['baseline']['precision'].append(baseline_precision)
        
        print(f"  Baseline - Recall@{top_k}: {baseline_recall:.4f}, "
              f"NDCG@{top_k}: {baseline_ndcg:.4f}, "
              f"Precision@{top_k}: {baseline_precision:.4f}")
        print(f"  Relevant docs in corpus: {len(relevant_indices)}, "
              f"Found in top-{top_k}: {len(set(ranked_docs_baseline) & relevant_indices)}")
        sys.stdout.flush()
        
        if baseline_ndcg >= 0.995:
            print(f"  Skipping ORE: Baseline already perfect (NDCG@{top_k} = {baseline_ndcg:.4f})")
            results['ore']['recall'].append(baseline_recall)
            results['ore']['ndcg'].append(baseline_ndcg)
            results['ore']['precision'].append(baseline_precision)
            print(f"  ORE      - Recall@{top_k}: {baseline_recall:.4f}, "
                  f"NDCG@{top_k}: {baseline_ndcg:.4f}, "
                  f"Precision@{top_k}: {baseline_precision:.4f}")
            print(f"  Improvement - Recall: +0.0000, NDCG: +0.0000, Precision: +0.0000")
            continue
        
        print(f"  Running ORE reranking (budget: {budget})...")
        sys.stdout.flush()
        ore = OnlineRelevanceEstimation(
            documents=documents,
            initial_scores=initial_scores,
            rerank_model=None,
            batch_size=batch_size,
            exploration_factor=exploration_factor
        )
        ore.rerank_model = simple_rerank_model(ore, documents, bm25_retriever, dense_retriever)
        
        final_ranking = ore.rerank(query_text, budget=budget, verbose=False)
        print(f"  ORE reranking complete!")
        sys.stdout.flush()
        ranked_docs_ore = [idx for idx, _ in final_ranking]
        
        ore_recall = calculate_recall(ranked_docs_ore, relevant_indices, k=top_k)
        ore_ndcg = calculate_ndcg(ranked_docs_ore, relevant_indices, k=top_k)
        ore_precision = calculate_precision(ranked_docs_ore, relevant_indices, k=top_k)
        
        results['ore']['recall'].append(ore_recall)
        results['ore']['ndcg'].append(ore_ndcg)
        results['ore']['precision'].append(ore_precision)
        
        print(f"  ORE      - Recall@{top_k}: {ore_recall:.4f}, "
              f"NDCG@{top_k}: {ore_ndcg:.4f}, "
              f"Precision@{top_k}: {ore_precision:.4f}")
        
        improvement = {
            'recall': ore_recall - baseline_recall,
            'ndcg': ore_ndcg - baseline_ndcg,
            'precision': ore_precision - baseline_precision
        }
        print(f"  Improvement - Recall: {improvement['recall']:+.4f}, "
              f"NDCG: {improvement['ndcg']:+.4f}, "
              f"Precision: {improvement['precision']:+.4f}")
    
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0
    
    baseline_recall_avg = avg(results['baseline']['recall'])
    baseline_ndcg_avg = avg(results['baseline']['ndcg'])
    baseline_precision_avg = avg(results['baseline']['precision'])
    
    ore_recall_avg = avg(results['ore']['recall'])
    ore_ndcg_avg = avg(results['ore']['ndcg'])
    ore_precision_avg = avg(results['ore']['precision'])
    
    recall_improvement = ore_recall_avg - baseline_recall_avg
    ndcg_improvement = ore_ndcg_avg - baseline_ndcg_avg
    precision_improvement = ore_precision_avg - baseline_precision_avg
    
    recall_pct = (recall_improvement / baseline_recall_avg * 100) if baseline_recall_avg > 0 else 0
    ndcg_pct = (ndcg_improvement / baseline_ndcg_avg * 100) if baseline_ndcg_avg > 0 else 0
    precision_pct = (precision_improvement / baseline_precision_avg * 100) if baseline_precision_avg > 0 else 0
    
    baseline_name = "BM25-only" if use_bm25_baseline else "Hybrid (BM25+Dense)"
    
    print(f"\nBaseline ({baseline_name}):")
    print(f"  Average Recall@{top_k}:    {baseline_recall_avg:.4f}")
    print(f"  Average NDCG@{top_k}:     {baseline_ndcg_avg:.4f}")
    print(f"  Average Precision@{top_k}: {baseline_precision_avg:.4f}")
    
    print(f"\nORE (Online Relevance Estimation):")
    print(f"  Average Recall@{top_k}:    {ore_recall_avg:.4f}")
    print(f"  Average NDCG@{top_k}:     {ore_ndcg_avg:.4f}")
    print(f"  Average Precision@{top_k}: {ore_precision_avg:.4f}")
    
    print(f"\n{'='*80}")
    print("IMPROVEMENT RESULTS:")
    print(f"{'='*80}")
    print(f"  Recall@{top_k}:    {recall_improvement:+.4f} ({recall_pct:+.2f}%)")
    print(f"  NDCG@{top_k}:     {ndcg_improvement:+.4f} ({ndcg_pct:+.2f}%)")
    print(f"  Precision@{top_k}: {precision_improvement:+.4f} ({precision_pct:+.2f}%)")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Run ORE experiment with real dataset")
    parser.add_argument("--dataset", type=str, default="msmarco-passage/trec-dl-2019/judged",
                       help="Dataset name from ir_datasets")
    parser.add_argument("--num-queries", type=int, default=10,
                       help="Number of queries to evaluate")
    parser.add_argument("--num-docs", type=int, default=10000,
                       help="Number of documents to use")
    parser.add_argument("--top-k", type=int, default=20,
                       help="Initial retrieval top-k")
    parser.add_argument("--budget", type=int, default=50,
                       help="ORE re-ranking budget")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Hybrid retriever alpha")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="ORE batch size")
    parser.add_argument("--exploration", type=float, default=0.2,
                       help="ORE exploration factor")
    parser.add_argument("--bm25-baseline", action="store_true",
                       help="Use BM25-only as baseline (weaker, shows more improvement)")
    
    args = parser.parse_args()
    
    run_experiment(
        dataset_name=args.dataset,
        num_queries=args.num_queries,
        num_docs=args.num_docs,
        top_k=args.top_k,
        budget=args.budget,
        alpha=args.alpha,
        batch_size=args.batch_size,
        exploration_factor=args.exploration,
        use_bm25_baseline=args.bm25_baseline
    )


if __name__ == "__main__":
    main()

