# PROMISE+ Dataset Exploration

Exploratory data analysis for the PROMISE+ requirements dataset. The original `Promise+.arff` is **not modified**.

## Directory placement

Copy this `promise_exploration/` folder into `~/Unofficial Thesis/`, alongside `dataset/`.

Expected source dataset:

```
Unofficial Thesis/dataset/Promise+.arff
```

## Ubuntu setup

```bash
cd ~/Unofficial\ Thesis/promise_exploration
sudo apt install -y python3-venv  # only if not already installed
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run sequentially

```bash
python scripts/01_load_dataset.py
python scripts/02_dataset_overview.py
python scripts/03_data_quality.py
python scripts/04_class_distribution.py
python scripts/05_text_analysis.py
```

**Important:** Scripts 02–05 use heuristics to guess the text and class columns. First inspect script 01's printed columns and verify the guesses. When needed, override them:

```bash
python scripts/04_class_distribution.py --label-column 'ACTUAL_COLUMN_NAME'
python scripts/05_text_analysis.py --text-column 'ACTUAL_COLUMN_NAME'
```

Results are written into `outputs/tables/`, `outputs/figures/` and `outputs/reports/`. The Excel file is retained in `dataset/` but not analyzed in these initial scripts.
