"""Experiment 3 development entry point. Every run writes a new evidence folder."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import site
import sys

ROOT = Path(__file__).resolve().parent
if (ROOT / 'BUNDLE_ORIGIN.json').exists():
    bundle_origin=json.loads((ROOT / 'BUNDLE_ORIGIN.json').read_text(encoding='utf-8'))
    os.environ.setdefault('HYRA3_RELOCATE_FROM',bundle_origin['original_root'])
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Some relocated Windows distributions expose prefix='D:' and omit their own
# site-packages. Repair this process only, and record the resulting sys.path.
if importlib.util.find_spec("numpy") is None and os.name == "nt":
    candidate = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages"
    if candidate.is_dir():
        site.addsitedir(str(candidate))
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit", help="Audit draft protocol and planned episode counts, without simulation")
    commands.add_parser("test", help="Run physical/causal/clock regression tests")
    commands.add_parser("report", help="Summarize completed development runs and remaining prerequisites")
    design = commands.add_parser("design-check", help="Run all six design families, twelve hydrologies and three scales")
    design.add_argument("--workers", type=int, choices=range(1, 7), default=4)
    design.add_argument("--resume", type=Path, default=None, help="Resume a compatible design-check run without overwriting raw outputs")
    preflight = commands.add_parser("preflight", help="Generate independent ANUGA truth/nominal fields and run nominal-only closed-loop benchmark")
    preflight.add_argument("--side", type=int, choices=(32, 50, 70), default=70)
    preflight.add_argument("--duration", type=float, default=1200.)
    preflight.add_argument("--mesh-spacing", type=float, default=10.)
    preflight.add_argument("--output-interval", type=float, choices=(1., 2.), default=2.)
    preflight.add_argument("--family-seed", type=int, default=None)
    preflight.add_argument("--rise", type=float, choices=(.6, .9), default=.9)
    preflight.add_argument("--rise-duration", type=float, choices=(600., 900.), default=600.)
    preflight.add_argument("--perturbation", type=float, choices=(0., .1, .2), default=.1)
    preflight.add_argument("--delay", type=float, choices=(0., 10.), default=0.)
    preflight.add_argument("--layout-seed", type=int, default=31001)
    numerical = commands.add_parser("numerical", help="Design-only 10m/5m mesh and 1s/2s output comparison")
    numerical.add_argument("--side", type=int, choices=(32, 50, 70), default=50)
    numerical.add_argument("--duration", type=float, default=1200.)
    numerical.add_argument("--family-seed", type=int, default=None)
    numerical.add_argument("--rise", type=float, choices=(.6, .9), default=.9)
    numerical.add_argument("--rise-duration", type=float, choices=(600., 900.), default=600.)
    numerical.add_argument("--perturbation", type=float, choices=(0., .1, .2), default=.1)
    episode = commands.add_parser("episode", help="Run a nominal-only preflight episode on saved ANUGA fields")
    episode.add_argument("--truth", type=Path, required=True)
    episode.add_argument("--nominal", type=Path, required=True)
    episode.add_argument("--delay", type=float, choices=(0., 10.), default=0.)
    episode.add_argument("--layout-seed", type=int, default=31001)
    args = parser.parse_args()
    os.chdir(ROOT)
    from hyra_exp3.pipeline import execute
    execute(ROOT, args)


if __name__ == "__main__":
    main()
