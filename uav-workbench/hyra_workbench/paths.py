"""Workspace paths resolved independently of the current working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'console-static'
