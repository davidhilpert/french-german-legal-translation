"""Data acquisition and preprocessing modules."""

from pathlib import Path

# Data directories
DATA_ROOT = Path(__file__).parent.parent.parent / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"

__all__ = ["DATA_ROOT", "RAW_DATA_DIR", "PROCESSED_DATA_DIR"]
