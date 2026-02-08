# Online Relevance Estimation (ORE)

Implementation of "Breaking the Lens of the Telescope: Online Relevance Estimation over Large Retrieval Sets"

**Paper:** [ACM Digital Library](https://doi.org/10.1145/3726302.3729910)  
**Original GitHub:** [https://github.com/elixir-research-group/ORE](https://github.com/elixir-research-group/ORE)

## Setup

### 1. Environment Setup

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Dataset Download

Download datasets to project `data/` folder:

```bash
python setup_datasets.py
```

This downloads TREC DL 2019/2020 datasets (~3.8GB) to `data/.ir_datasets/`.

**Note:** First run requires accepting MS MARCO data usage agreement.

If datasets already exist in `~/.ir_datasets/`, move them:
```bash
python move_datasets.py
```

## Testing

Run tests with mock data:
```bash
python -m tests.test_basic
```

## Running Experiments

**Quick test (small dataset):**
```bash
python experiment.py --num-queries 10 --num-docs 5000
```

**Recommended (full dataset, matches paper):**
```bash
python experiment.py --num-docs 0 --num-queries 43 --budget 200
```

**Test on both TREC DL datasets:**
```bash
# TREC DL 2019 (43 queries)
python experiment.py --dataset msmarco-passage/trec-dl-2019/judged --num-docs 0 --num-queries 43 --budget 200

# TREC DL 2020 (54 queries)
python experiment.py --dataset msmarco-passage/trec-dl-2020/judged --num-docs 0 --num-queries 54 --budget 200
```

**Custom parameters:**
```bash
python experiment.py \
    --dataset msmarco-passage/trec-dl-2019/judged \
    --num-queries 20 \
    --num-docs 100000 \
    --top-k 20 \
    --budget 200 \
    --alpha 0.5 \
    --batch-size 10 \
    --exploration 0.2
```

### Parameters

- `--dataset`: Dataset name (default: `msmarco-passage/trec-dl-2019/judged`)
- `--num-queries`: Number of queries to evaluate (default: 10)
- `--num-docs`: Number of documents to use. Use `0` to load all 8.8M documents (default: 1000)
- `--top-k`: Initial retrieval top-k (default: 20)
- `--budget`: ORE re-ranking budget (default: 50, recommended: 200+)
- `--alpha`: Hybrid retriever alpha (default: 0.5)
- `--batch-size`: ORE batch size (default: 10)
- `--exploration`: ORE exploration factor (default: 0.2)

## Project Structure

```
├── src/
│   ├── retrieval/          # BM25, Dense, Hybrid retrievers
│   ├── reranking/          # ORE algorithm
│   ├── evaluation/         # Metrics (Recall, NDCG, Precision)
│   └── data/               # Dataset loader
├── tests/                  # Test scripts
├── data/                   # Dataset storage
├── experiment.py           # Main experiment script
├── setup_datasets.py       # Dataset download script
└── requirements.txt        # Dependencies
```

## Quick Start (All-in-One)

For first-time setup:
```bash
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && python setup_datasets.py && python experiment.py --num-queries 5 --num-docs 500
```



Task	Priority
1. Paper Analysis Report (Doc/paper_analysis.md)	🔴 HIGH
2. Results Documentation (Doc/results.md)	🔴 HIGH
3. Statistical Significance Tests	🟡 MEDIUM
4. Presentation Slides	🟡 MEDIUM
5. Comprehensive Tests	🟡 MEDIUM
6. Code Documentation
