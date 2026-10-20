"""Lossless columnar observation logs for replay without large JSON dictionaries."""
from pathlib import Path
import numpy as np

from .artifacts import sha256


def save_observations(path, records):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    count = len(records)
    fields = dict(
        sample_time_s=np.fromiter((r["sample_time"] for r in records), float, count),
        arrival_time_s=np.fromiter((r["arrival_time"] for r in records), float, count),
        depth_m=np.fromiter((r["depth_m"] for r in records), float, count),
        row=np.fromiter((r["cell"][0] for r in records), np.int32, count),
        col=np.fromiter((r["cell"][1] for r in records), np.int32, count),
        drone=np.fromiter((r["drone"] for r in records), np.int32, count),
    )
    if np.any(fields["arrival_time_s"] < fields["sample_time_s"]):
        raise ValueError("Observation arrives before it is sampled")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **fields)
    temporary.replace(path)
    return dict(path=str(path.resolve()), sha256=sha256(path), records=count, bytes=path.stat().st_size,
                schema="hyra3-observation-columns-v1", columns=list(fields),
                precision="float64_times_and_observed_depths_int32_cells_and_drone", arrived_records_only=True)
