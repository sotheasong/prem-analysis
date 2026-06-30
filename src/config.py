"""Project paths — derived from this file's location, not the process cwd.

The ``data/`` tree is the single source of truth for all inputs and outputs.
Every notebook reads from / writes to these constants (never a hard-coded path)
so that re-running the notebooks reproduces the same organized layout:

    data/
    ├── raw/          # untouched StatsBomb pulls (events_parts, events_json, csv)
    ├── processed/    # cleaned, canonical event/match stores (events_*.parquet, matches_*.csv)
    ├── possessions/  # possession-chain & rhythm outputs
    ├── passing/      # passing-intent & passing-structure outputs
    └── transitions/  # transition outputs
"""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
POSSESSIONS_DIR = DATA_DIR / "possessions"
PASSING_DIR = DATA_DIR / "passing"
TRANSITIONS_DIR = DATA_DIR / "transitions"

# Output subdirectories — created on import so notebooks can write freely.
DATA_SUBDIRS = (
    RAW_DIR,
    PROCESSED_DIR,
    POSSESSIONS_DIR,
    PASSING_DIR,
    TRANSITIONS_DIR,
)

for _d in DATA_SUBDIRS:
    _d.mkdir(parents=True, exist_ok=True)
