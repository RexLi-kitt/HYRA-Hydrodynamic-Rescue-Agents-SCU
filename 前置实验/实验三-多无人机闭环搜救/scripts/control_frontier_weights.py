from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
from hyra_exp3.frontier_control import run_control
from hyra_exp3.protocol import load_protocol

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=6)
    args = parser.parse_args()
    source = args.source or Path(load_protocol(ROOT / "results/latest_planner_replay.json")["run"])
    print(run_control(ROOT, source, workers=args.workers))
