"""Run the fixed seven-stage development probe, leaving the original protocol intact."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
from hyra_exp3.phase_probe import run_probe

if __name__ == "__main__":
    print(run_probe(ROOT))
