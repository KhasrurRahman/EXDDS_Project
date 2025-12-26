from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from src.reranking import OnlineRelevanceEstimation


def create_mock_data():
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Python is a popular programming language for data science",
        "Deep learning uses neural networks with multiple layers",
        "Information retrieval finds relevant documents for queries",
        "Search engines use ranking algorithms to order results",
        "Natural language processing helps computers understand text",
        "Computer vision enables machines to interpret images",
        "Reinforcement learning trains agents through trial and error",
        "Supervised learning uses labeled training data",
        "Unsupervised learning finds patterns without labels",
        "Neural networks are inspired by biological neurons",
        "Gradient descent optimizes model parameters",
        "Backpropagation calculates gradients in neural networks",
        "Convolutional networks excel at image recognition",
        "Recurrent networks process sequential data",
    ]
    
    queries = [
        "What is machine learning?",
        "How do neural networks work?",
        "Python programming for data science",
    ]
    
    return documents, queries


def test_bm25():
    documents, queries = create_mock_data()
    retriever = BM25Retriever(documents)
    results = retriever.retrieve(queries[0], top_k=5)
    
    print("BM25 Results:")
    for i, (doc_idx, score) in enumerate(results, 1):
        print(f"  {i}. [Score: {score:.3f}] {documents[doc_idx]}")


def test_dense():
    documents, queries = create_mock_data()
    retriever = DenseRetriever(documents)
    results = retriever.retrieve(queries[0], top_k=5)
    
    print("\nDense Results:")
    for i, (doc_idx, score) in enumerate(results, 1):
        print(f"  {i}. [Score: {score:.3f}] {documents[doc_idx]}")


def test_hybrid():
    documents, queries = create_mock_data()
    retriever = HybridRetriever(documents, alpha=0.5)
    results = retriever.retrieve(queries[0], top_k=5)
    
    print("\nHybrid Results:")
    for i, (doc_idx, score) in enumerate(results, 1):
        print(f"  {i}. [Score: {score:.3f}] {documents[doc_idx]}")


def test_ore():
    documents, queries = create_mock_data()
    hybrid_retriever = HybridRetriever(documents)
    query = queries[0]
    initial_results = hybrid_retriever.retrieve(query, top_k=20)
    initial_scores = {idx: score for idx, score in initial_results}
    
    def simple_rerank_model(query, doc_indices):
        new_scores = {}
        for doc_idx in doc_indices:
            old_score = initial_scores.get(doc_idx, 0.0)
            new_scores[doc_idx] = old_score * 1.2
        return new_scores
    
    ore = OnlineRelevanceEstimation(
        documents=documents,
        initial_scores=initial_scores,
        rerank_model=simple_rerank_model,
        batch_size=5,
        exploration_factor=0.2
    )
    
    final_ranking = ore.rerank(query, budget=15, verbose=True)
    
    print("\nORE Final Results:")
    for i, (doc_idx, score) in enumerate(final_ranking[:10], 1):
        print(f"  {i}. [Score: {score:.3f}] {documents[doc_idx]}")


if __name__ == "__main__":
    try:
        test_bm25()
        test_dense()
        test_hybrid()
        test_ore()
        print("\nAll tests completed!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

