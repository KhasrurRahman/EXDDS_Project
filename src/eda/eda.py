import ir_datasets
import numpy as np
from collections import Counter


def analyze_dataset(dataset_name: str = "msmarco-passage/trec-dl-2020/judged"):
    print(f"\n{'='*50}")
    print(f"  EDA: {dataset_name}")
    print(f"{'='*50}\n")
    
    dataset = ir_datasets.load(dataset_name)
    
    queries = {q.query_id: q.text for q in dataset.queries_iter()}
    
    qrels = {}
    for qrel in dataset.qrels_iter():
        if qrel.query_id not in qrels:
            qrels[qrel.query_id] = []
        qrels[qrel.query_id].append((qrel.doc_id, qrel.relevance))
    
    docs = []
    for i, doc in enumerate(dataset.docs_iter()):
        if i >= 10000:
            break
        docs.append(doc)
    
    print("1. STRUCTURE")
    print(f"   Queries: {len(queries)}")
    print(f"   Judgments: {sum(len(v) for v in qrels.values())}")
    print(f"   Docs sampled: {len(docs)}")
    
    sample_qid = list(queries.keys())[0]
    print(f"\n   Sample query: [{sample_qid}] {queries[sample_qid]}")
    print(f"   Sample doc: {docs[0].text[:200]}...")
    
    sample_qrel = qrels[sample_qid][0]
    print(f"   Sample judgment: Query {sample_qid} -> Doc {sample_qrel[0]} -> Score {sample_qrel[1]}")
    
    print("\n2. QUERY STATS")
    query_lengths = [len(q.split()) for q in queries.values()]
    print(f"   Total: {len(queries)}")
    print(f"   Avg length: {np.mean(query_lengths):.1f} words")
    print(f"   Min: {min(query_lengths)} words")
    print(f"   Max: {max(query_lengths)} words")
    print(f"   Median: {np.median(query_lengths):.1f} words")
    
    print("\n3. DOCUMENT STATS")
    doc_lengths = [len(d.text.split()) for d in docs]
    print(f"   Analyzed: {len(docs)}")
    print(f"   Avg length: {np.mean(doc_lengths):.1f} words")
    print(f"   Min: {min(doc_lengths)} words")
    print(f"   Max: {max(doc_lengths)} words")
    print(f"   Median: {np.median(doc_lengths):.1f} words")
    
    print("\n4. RELEVANCE DISTRIBUTION")
    all_relevance = [rel for rels in qrels.values() for _, rel in rels]
    rel_counts = Counter(all_relevance)
    labels = {0: "Not relevant", 1: "Related", 2: "Highly rel", 3: "Perfect"}
    print(f"   Total judgments: {len(all_relevance)}")
    for score in sorted(rel_counts.keys()):
        pct = rel_counts[score] / len(all_relevance) * 100
        bar = "#" * int(pct / 3)
        print(f"   {score} ({labels[score]}): {rel_counts[score]} ({pct:.1f}%) {bar}")
    
    relevant_per_query = [len([r for _, r in rels if r >= 1]) for rels in qrels.values()]
    print(f"\n   Relevant docs per query:")
    print(f"      Avg: {np.mean(relevant_per_query):.1f}")
    print(f"      Min: {min(relevant_per_query)}")
    print(f"      Max: {max(relevant_per_query)}")
    
    print("\n5. PASSAGE LENGTH DISTRIBUTION")
    bins = [(0, 20), (21, 40), (41, 60), (61, 80), (81, 100), (101, 150), (151, 300)]
    for low, high in bins:
        count = sum(1 for l in doc_lengths if low <= l <= high)
        pct = count / len(doc_lengths) * 100
        bar = "#" * int(pct / 2)
        print(f"   {low:3d}-{high:3d} words: {count:4d} ({pct:5.1f}%) {bar}")
    
    print("\n6. QUERY LENGTH DISTRIBUTION")
    q_bins = [(1, 3), (4, 5), (6, 7), (8, 10), (11, 20)]
    for low, high in q_bins:
        count = sum(1 for l in query_lengths if low <= l <= high)
        pct = count / len(query_lengths) * 100
        bar = "#" * int(pct / 2)
        print(f"   {low:2d}-{high:2d} words: {count:3d} ({pct:5.1f}%) {bar}")
    
    print("\n7. QUERY TOPICS")
    categories = {"what": 0, "who": 0, "how": 0, "why": 0, "when": 0, "where": 0, "define": 0, "other": 0}
    for q in queries.values():
        q_lower = q.lower()
        first = q_lower.split()[0]
        if first in ["what", "who", "how", "why", "when", "where"]:
            categories[first] += 1
        elif "define" in q_lower or "definition" in q_lower or "meaning" in q_lower:
            categories["define"] += 1
        else:
            categories["other"] += 1
    
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        if count > 0:
            pct = count / len(queries) * 100
            bar = "#" * int(pct / 2)
            print(f"   {cat:8s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    print("\n8. SAMPLE QUERIES BY TYPE")
    for qtype in ["what", "who", "why", "how"]:
        samples = [q for q in queries.values() if q.lower().startswith(qtype)][:3]
        if samples:
            print(f"   {qtype.upper()}:")
            for s in samples:
                print(f"      - {s}")
    
    define_samples = [q for q in queries.values() if "define" in q.lower() or "definition" in q.lower()][:3]
    if define_samples:
        print(f"   DEFINITION:")
        for s in define_samples:
            print(f"      - {s}")
    
    print("\n9. SAMPLE QUERIES WITH RELEVANCE COUNT")
    for i, (qid, text) in enumerate(list(queries.items())[:5]):
        rel_count = len([r for _, r in qrels.get(qid, []) if r >= 1])
        print(f"   {i+1}. [{qid}] \"{text}\"")
        print(f"      Relevant docs: {rel_count}")
    
    print(f"\n{'='*50}\n")


if __name__ == "__main__":
    analyze_dataset("msmarco-passage/trec-dl-2020/judged")
    analyze_dataset("msmarco-passage/trec-dl-2019/judged")