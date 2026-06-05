"""Project paths — derived from this file's location, not the process cwd."""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
DATA_DIR = NOTEBOOK_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
