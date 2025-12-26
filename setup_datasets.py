import os
from pathlib import Path

def set_data_directory(data_dir: str = "data"):
    data_path = Path(data_dir).resolve()
    data_path.mkdir(parents=True, exist_ok=True)
    os.environ['IR_DATASETS_HOME'] = str(data_path)
    return data_path

set_data_directory("data")

import ir_datasets


def setup_datasets():
    data_dir = Path("data")
    data_path = set_data_directory("data")
    
    datasets = [
        "msmarco-passage/trec-dl-2019/judged",
        "msmarco-passage/trec-dl-2020/judged",
    ]
    
    print("Setting up datasets...")
    print("=" * 80)
    print(f"Datasets will be downloaded to: {data_path}")
    print("This is a one-time download. Subsequent runs will use cached data.")
    print("=" * 80)
    
    for dataset_name in datasets:
        print(f"\n[{dataset_name}]")
        print("Loading dataset (this will download if not already cached)...")
        try:
            dataset = ir_datasets.load(dataset_name)
            print(f"  Dataset loaded successfully!")
            print(f"  Documents: {dataset.docs_count():,}" if hasattr(dataset, 'docs_count') else "  Documents: Available")
            print(f"  Queries: {dataset.queries_count():,}" if hasattr(dataset, 'queries_count') else "  Queries: Available")
        except Exception as e:
            print(f"  Error loading dataset: {e}")
            print(f"  You may need to accept the data usage agreement.")
            print(f"  Run the experiment script and follow the prompts.")
    
    print("\n" + "=" * 80)
    print("Setup complete!")
    print("=" * 80)
    print("\nYou can now run experiments with:")
    print("  python experiment.py --num-queries 10 --num-docs 1000")


if __name__ == "__main__":
    setup_datasets()

