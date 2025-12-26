from src.retrieval import HybridRetriever
from src.reranking import OnlineRelevanceEstimation


def main():
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn",
        "Python is a popular programming language used for data science and machine learning",
        "Deep learning uses neural networks with multiple layers to process complex data",
        "Information retrieval systems find relevant documents for user queries",
        "Search engines use ranking algorithms to order search results by relevance",
        "Natural language processing helps computers understand and process human language",
        "Computer vision enables machines to interpret and understand visual information",
        "Reinforcement learning trains agents through trial and error interactions",
        "Supervised learning uses labeled training data to teach models",
        "Unsupervised learning finds hidden patterns in data without labels",
        "Neural networks are computational models inspired by biological neurons",
        "Gradient descent is an optimization algorithm used to train machine learning models",
        "Backpropagation calculates gradients in neural networks for training",
        "Convolutional neural networks excel at image recognition tasks",
        "Recurrent neural networks process sequential data like text and time series",
    ]
    
    query = "What is machine learning and how does it work?"
    
    retriever = HybridRetriever(documents, alpha=0.5)
    initial_results = retriever.retrieve(query, top_k=10)
    initial_scores = {idx: score for idx, score in initial_results}
    
    def rerank_model(query, doc_indices):
        new_scores = {}
        for doc_idx in doc_indices:
            old_score = initial_scores.get(doc_idx, 0.0)
            new_scores[doc_idx] = old_score * 1.3
        return new_scores
    
    ore = OnlineRelevanceEstimation(
        documents=documents,
        initial_scores=initial_scores,
        rerank_model=rerank_model,
        batch_size=3,
        exploration_factor=0.2
    )
    
    final_ranking = ore.rerank(query, budget=9, verbose=True)
    
    print("\nFinal Top 10 Ranking:")
    for i, (doc_idx, score) in enumerate(final_ranking[:10], 1):
        print(f"{i:2d}. [Score: {score:.3f}] {documents[doc_idx][:70]}...")


if __name__ == "__main__":
    main()

