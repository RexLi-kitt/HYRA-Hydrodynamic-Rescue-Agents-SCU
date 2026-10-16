"""Immutable run directories and source fingerprints."""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, obj):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=convert,allow_nan=False),encoding="utf-8")


def new_run(stage, config):
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    path = ROOT/"results"/"runs"/f"{stamp}_{stage}"
    path.mkdir(parents=True,exist_ok=False)
    write_json(path/"config.json",config)
    files = list((ROOT/"src").rglob("*.py"))+list((ROOT/"scripts").glob("*.py"))+list((ROOT/"configs").glob("*.json"))
    write_json(path/"provenance.json",dict(command=sys.argv,executable=sys.executable,
        python=sys.version,platform=platform.platform(),time_zone="Asia/Shanghai",
        files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}))
    lock = subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True)
    (path/"dependencies.txt").write_text(lock.stdout+lock.stderr,encoding="utf-8")
    return path
