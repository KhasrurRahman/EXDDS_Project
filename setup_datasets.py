import os
from pathlib import Path
from tqdm import tqdm

def set_data_directory(data_dir: str = "data"):
    data_path = Path(data_dir).resolve()
    data_path.mkdir(parents=True, exist_ok=True)
    os.environ['IR_DATASETS_HOME'] = str(data_path)
    return data_path

set_data_directory("data")

import ir_datasets


def setup_datasets():
    data_path = set_data_directory("data")
    
    datasets = [
        "msmarco-passage/trec-dl-2019/judged",
        "msmarco-passage/trec-dl-2020/judged",
    ]
    
    print("Setting up datasets...")
    print(f"Download location: {data_path}")
    print("=" * 80)
    
    for dataset_name in datasets:
        print(f"\n[{dataset_name}]")
        try:
            dataset = ir_datasets.load(dataset_name)
            
            # Need to iterate to trigger download (lazy loading)
            print("Downloading documents...")
            doc_count = 0
            for doc in tqdm(dataset.docs_iter(), desc="  Docs", total=100):
                doc_count += 1
                if doc_count >= 100:
                    break
            
            print("Downloading queries...")
            query_count = 0
            for query in tqdm(dataset.queries_iter(), desc="  Queries"):
                query_count += 1
            
            print("Downloading qrels...")
            qrel_count = 0
            for qrel in tqdm(dataset.qrels_iter(), desc="  Qrels"):
                qrel_count += 1
            
            print(f"✓ Ready: {query_count} queries, {qrel_count} judgments")
            
            # Check if collection file exists
            collection_path = data_path / ".ir_datasets" / "msmarco-passage" / "collection.tsv"
            if collection_path.exists():
                size_gb = collection_path.stat().st_size / (1024**3)
                print(f"Collection size: {size_gb:.2f} GB")
                
        except Exception as e:
            print(f"Error: {e}")
            print("You may need to accept the data usage agreement.")
    
    print("\nSetup complete!")


if __name__ == "__main__":
    setup_datasets()