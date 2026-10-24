"""Validate both fixed mission-phase candidates across the entire design set."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
from hyra_exp3.phase_matrix import run_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    print(run_matrix(ROOT, workers=args.workers, resume=args.resume))
