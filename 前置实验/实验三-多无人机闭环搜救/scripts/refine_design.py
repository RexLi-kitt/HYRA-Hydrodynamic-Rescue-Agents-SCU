from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
from hyra_exp3.refinement import run_refinement
from hyra_exp3.protocol import load_protocol
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    source = load_protocol(ROOT / "results/latest_phase_matrix.json")["run"]
    print(run_refinement(ROOT, source, workers=args.workers, resume=args.resume))
