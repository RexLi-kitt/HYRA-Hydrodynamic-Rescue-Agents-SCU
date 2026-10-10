"""Load unchanged FreeTerraForged Cell fields into NumPy arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def load_grid(directory: str | Path):
    import numpy as np

    directory = Path(directory)
    metadata = json.loads((directory / "grid.json").read_text())
    shape = tuple(metadata["shape"])
    result = {}
    for name, field in metadata["fields"].items():
        values = np.fromfile(directory / field["file"], dtype=field["dtype"])
        if values.size != shape[0] * shape[1]:
            raise ValueError(f"Invalid data length: {name}")
        result[name] = values.reshape(shape)
    # Keep the original height; expose its continuous conversion separately.
    result["height_blocks"] = result["height"] * metadata["terrain_scale_factor"]
    result["river_water_level_blocks"] = result["riverWaterLevel"] * metadata["terrain_scale_factor"]
    return result, metadata


def verify_upstream(root: str | Path):
    root = Path(root)
    manifest = json.loads((root / "upstream-manifest.json").read_text())
    mismatches = []
    for name, expected in manifest["sha256"].items():
        file = root / "upstream" / name
        actual = hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None
        if actual != expected:
            mismatches.append(name)
    return {"commit": manifest["commit"], "checked_files": len(manifest["sha256"]), "mismatches": mismatches}


def load_volume(directory: str | Path):
    import numpy as np

    directory = Path(directory)
    metadata = json.loads((directory / "volume.json").read_text())
    nz, ny, nx = metadata["shape"]
    result = {"blocks": np.fromfile(directory / "blocks.i32", dtype="<i4").reshape(nz, ny, nx)}
    result["surface_biomes"] = np.fromfile(directory / "surface_biomes.i32", dtype="<i4").reshape(nz, nx)
    for file in metadata["heightmaps"]:
        result[Path(file).stem] = np.fromfile(directory / file, dtype="<i4").reshape(nz, nx)
    return result, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path)
    parser.add_argument("--npz", type=Path)
    parser.add_argument("--verify-source", action="store_true")
    parser.add_argument("--volume", action="store_true")
    args = parser.parse_args()
    if args.verify_source:
        report = verify_upstream(Path(__file__).resolve().parent)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["mismatches"]:
            raise SystemExit(1)
    if args.directory:
        arrays, metadata = (load_volume if args.volume else load_grid)(args.directory)
        if args.npz:
            import numpy as np

            np.savez_compressed(args.npz, **arrays)
        print(json.dumps({"shape": metadata["shape"], "fields": list(arrays), "revision": metadata["upstream_revision"]}, indent=2))
    elif not args.verify_source:
        parser.error("Supply a grid directory or --verify-source")


if __name__ == "__main__":
    main()
