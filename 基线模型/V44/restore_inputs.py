"""Verify the Git LFS archives and restore only the registered V44 input files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

BASE = Path(__file__).resolve().parent
DATA = BASE / "原始数据"
ROOT = BASE / "runtime"
GROUPS = {"training": {"fit", "select"}, "evaluation": {"repeat", "fresh"}, "mixed": {"mixed"}}


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def restore(group, entries, hashes):
    path = DATA / f"V44-{group}.zip"
    if not path.exists() or digest(path) != hashes[path.name]:
        raise ValueError(f"Missing or incomplete archive. Run git lfs pull and retry: {path}")
    expected = {row[key] for row in entries.values() if row["phase"] in GROUPS[group]
                for key in ("water", "history")}
    with zipfile.ZipFile(path) as archive:
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected:
            raise ValueError(f"Unexpected file list in {path.name}")
        for name in sorted(expected):
            dest = (ROOT / name).resolve()
            dest.relative_to(ROOT.resolve())
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
                temp = Path(tmp.name)
            try:
                with archive.open(name) as src, temp.open("wb") as out:
                    shutil.copyfileobj(src, out)
                if dest.exists():
                    if digest(dest) != digest(temp):
                        raise ValueError(f"Existing file differs from archive: {dest}")
                else:
                    os.replace(temp, dest)
            finally:
                temp.unlink(missing_ok=True)
    print(f"{group}: {len(expected)} registered inputs restored", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=(*GROUPS, "all"), default="all")
    args = parser.parse_args()
    rows = json.loads((ROOT / "inputs.json").read_text(encoding="utf8"))
    hashes = json.loads((DATA / "SHA256.json").read_text(encoding="utf8"))
    for group in GROUPS if args.group == "all" else (args.group,):
        restore(group, rows, hashes)
