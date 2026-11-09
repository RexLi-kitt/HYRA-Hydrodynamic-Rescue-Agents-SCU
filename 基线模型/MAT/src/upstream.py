"""Fetch and load the exact upstream files used by this checkpoint."""

import argparse
import builtins
import hashlib
import json
from pathlib import Path
from types import ModuleType
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "_upstream"
MANIFEST = json.loads((ROOT / "upstream.json").read_text(encoding="utf-8"))


def checked_bytes(name):
    path = CACHE / name
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        raise RuntimeError("Run python src/upstream.py in the MAT directory first") from None
    if hashlib.sha256(content).hexdigest() != MANIFEST["files"][name]["sha256"]:
        raise ValueError(f"Upstream checksum mismatch for {path}")
    return content


def load_transformer():
    modules = {}

    def import_upstream(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name in modules:
            return modules[name]
        return builtins.__import__(name, globals, locals, fromlist, level)

    # Resolve only the pinned imports for these modules, without changing
    # sys.path or any other installed copy of the upstream mat package.
    def load(name, filename):
        module = ModuleType(name)
        module.__file__ = str(CACHE / filename)
        module.__dict__["__builtins__"] = dict(vars(builtins), __import__=import_upstream)
        exec(compile(checked_bytes(filename), module.__file__, "exec"), vars(module))
        return module

    modules["mat.algorithms.utils.util"] = load("mat.algorithms.utils.util", "algorithm_util.py")
    modules["mat.algorithms.utils.transformer_act"] = load(
        "mat.algorithms.utils.transformer_act", "transformer_act.py"
    )
    return load("mat.algorithms.mat.algorithm.ma_transformer", "ma_transformer.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify cached files without downloading")
    args = parser.parse_args()
    for name, info in MANIFEST["files"].items():
        path = CACHE / name
        if not args.check and not path.exists():
            url = (
                "https://raw.githubusercontent.com/PKU-MARL/Multi-Agent-Transformer/"
                f"{MANIFEST['commit']}/{info['source']}"
            )
            with urlopen(url, timeout=30) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != info["sha256"]:
                raise ValueError(f"Downloaded checksum mismatch for {name}")
            CACHE.mkdir(exist_ok=True)
            path.write_bytes(content)
        checked_bytes(name)
        print(f"Verified {name}")


if __name__ == "__main__":
    main()
