"""Render completed phase-candidate validation and its failure diagnosis."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
from hyra_exp3.phase_matrix_reporting import plot, write_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    args = parser.parse_args()
    print(plot(ROOT, args.source_run))
    print(write_report(ROOT, args.source_run))
