"""External replay-JSON compatibility; frozen models and inference stay intact.

The registered evaluator can produce NumPy scalar coordinates in real rescue
events. Install this adapter in the controller process before confirm/report.
Only ``signal_routes.evaluate.write`` is replaced; source/model seals are not
rewritten, and nonfinite values remain forbidden.
"""
from pathlib import Path
import hashlib
import json
import os
import uuid

import numpy as np


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_result_json(path, value):
    """Preserve JSON values, reject NaN/Infinity, then atomically replace a file."""
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                         default=_json_default)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as output:
            output.write(payload)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def adapter_receipt():
    path = Path(__file__).resolve()
    return dict(schema="hyra-signal-routes-result-io-v1", adapter_path=str(path),
                adapter_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                patched_binding="signal_routes.evaluate.write",
                numpy_policy="numeric scalar .item(); array .tolist(); strict finite JSON",
                atomic_replace=True, inference_changed=False,
                training_sources_changed=False, model_metadata_changed=False)


def install_result_io_adapter(study=None, expected_sha256=None):
    """Install idempotently, refusing an unknown writer or an unexpected adapter."""
    from signal_routes import common, evaluate

    receipt = adapter_receipt()
    if expected_sha256 is not None and receipt["adapter_sha256"] != expected_sha256:
        raise ValueError("Result I/O adapter SHA-256 does not match the expected version")
    current = evaluate.write
    if current is not common.write and current is not write_result_json:
        raise RuntimeError("Refuse to replace an unrecognized evaluator JSON writer")
    if study is not None:
        # This separate receipt never changes the registered source/model metadata.
        common.write(Path(study) / "result-io-adapter.json", receipt)
    evaluate.write = write_result_json
    return receipt
