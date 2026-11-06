"""CLI bootstrap for the frozen final inference, independent of archive/."""

import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from sleep_edf_final.runner import main

if __name__ == "__main__":
    main()
