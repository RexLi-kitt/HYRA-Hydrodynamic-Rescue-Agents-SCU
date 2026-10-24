"""Write the design opportunity report from an explicit completed/partial run."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run  # process-local runtime setup, with no command execution
from hyra_exp3.design_reporting import write_design_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    args = parser.parse_args()
    print(write_design_report(ROOT, args.source_run))
