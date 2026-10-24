from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run
from hyra_exp3.planner_replay_audit import audit_replay, audit_control
from hyra_exp3.protocol import load_protocol

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=4)
    args = parser.parse_args()
    source = args.source or Path(load_protocol(ROOT / "results/latest_planner_replay.json")["run"])
    audit = audit_control if (source / "matched_control_summary.json").exists() else audit_replay
    print(json.dumps(audit(source, workers=args.workers), ensure_ascii=False, indent=2))
