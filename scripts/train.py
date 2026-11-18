#!/usr/bin/env python
"""I-JEPA (+SUSReg/SIGReg) pretraining + evaluation. Run from the repo root:

    .venv/bin/python scripts/train.py --config configs/c1.yaml [--seed 1] [--smoke]
    .venv/bin/python scripts/train.py --config configs/c6.yaml --max-steps 200   # throughput bench

See src/ijepa_sphere/trainer.py for the options, outputs and split semantics.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf"))
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from ijepa_sphere.trainer import main  # noqa: E402

if __name__ == "__main__":
    main()
