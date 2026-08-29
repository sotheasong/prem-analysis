# Premier League Playstyle Analysis

Analyzing how football playing styles differ between teams, using StatsBomb open
event data. Current scope: data extraction and per-match feature engineering
(possession, passing, transition).

## Layout

```
data/                  # gitignored; subdirs auto-created by config on import
  raw/                 # untouched StatsBomb pulls (events_parts/, events_json/, csv)
  processed/           # cleaned canonical stores: events_<season>.parquet, matches_<season>.csv
  possessions/         # possession-chain & rhythm outputs
  passing/             # passing-intent & passing-structure outputs
  transitions/         # transition outputs

notebooks/
  extraction/          # pull & build the raw -> processed event stores
  feature_engineering/ # possession.ipynb, passing.ipynb, transition.ipynb (per-match features)

src/
  extraction/          # StatsBomb download + event normalization
  features/            # possession.py, passing.py, transition.py, batch.py
  diagnostics/         # style.py — season style diagnostic (feature registry)
  visualization/       # possession.py, passing.py, transition.py
  utils/               # constants.py
  config.py            # all data paths; auto-creates the data/ subdirs on import
```

## Reproducibility

`src/config.py` defines every data path and creates the `data/` subdirectories on
import. Notebooks only ever read from / write to those constants (never hard-coded
paths), so re-running them rebuilds the organized `data/` tree from scratch:

1. `notebooks/extraction/` → builds `data/raw/` and `data/processed/`.
2. `notebooks/feature_engineering/` → writes feature tables to
   `data/possessions/`, `data/passing/`, `data/transitions/`.
