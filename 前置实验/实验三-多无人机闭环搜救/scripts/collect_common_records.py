from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
from hyra_exp3.common_data import run_collection
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 7))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    print(run_collection(ROOT, workers=args.workers, resume=args.resume))
