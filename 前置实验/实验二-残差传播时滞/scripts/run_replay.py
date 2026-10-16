"""Fixed-observation replay (information diagnosis, not rescue yield)."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from hyra_validation.pipeline import run_replay
from hyra_validation.artifacts import ROOT

if __name__ == "__main__":
    run_replay(sys.argv[1] if len(sys.argv)>1 else ROOT/"configs/pilot.v2.json")
