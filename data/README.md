# Data Folder

This folder contains downloaded datasets for experiments.

## Structure

```
data/
├── .ir_datasets/          # Dataset cache (hidden folder, ~3.8GB)
│   └── msmarco-passage/   # MS MARCO passage datasets (ONLY dataset needed)
│       ├── collection.tsv # Document collection (8.8M documents)
│       ├── trec-dl-2019/  # TREC DL 2019 queries and qrels
│       └── trec-dl-2020/  # TREC DL 2020 queries and qrels
└── README.md              # This file
```

**Note:** Only `msmarco-passage` is needed for this project. Other datasets have been removed.

## Viewing Datasets

The `.ir_datasets` folder is hidden (starts with a dot). To view it:

**In Terminal:**
```bash
ls -la data/
```

**In Finder (Mac):**
- Press `Cmd + Shift + .` to show hidden files
- Or use: `open data/.ir_datasets`

**In VS Code:**
- Hidden folders are shown by default
- Look for `.ir_datasets` folder

## Dataset Size

- Total size: ~3.8 GB
- Includes: 
  - MS MARCO passage collection (8.8M documents)
  - TREC DL 2019 queries and relevance judgments (43 queries)
  - TREC DL 2020 queries and relevance judgments (54 queries)

## What's Included

- **collection.tsv**: 8.8 million document passages
- **trec-dl-2019/**: 43 queries with relevance judgments
- **trec-dl-2020/**: 54 queries with relevance judgments

This is the only dataset needed for reproducing the ORE paper experiments.

## Usage

Datasets are automatically used by:
- `experiment.py` - Main experiment script
- `setup_datasets.py` - Dataset setup script

No manual configuration needed!

