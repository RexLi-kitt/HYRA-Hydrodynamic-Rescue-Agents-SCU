"""Append-only run records and process-local runtime diagnostics."""
from __future__ import annotations

import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import uuid

TZ = timezone(timedelta(hours=8))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_memory_bytes():
    if os.name != "nt":
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in (
                "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                "PagefileUsage", "PeakPagefileUsage")]

    info = Counters()
    info.cb = ctypes.sizeof(info)
    handle = ctypes.windll.kernel32.GetCurrentProcess
    handle.restype = ctypes.c_void_p
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not query(handle(), ctypes.byref(info), info.cb):
        return None
    return int(info.PeakWorkingSetSize)


def memory_capacity():
    if os.name != "nt":
        if hasattr(os, "sysconf"):
            return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        return None

    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total_phys", "avail_phys", "total_page", "avail_page",
                "total_virtual", "avail_virtual", "avail_extended")]

    status = Status()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.total_phys)


def environment(root):
    packages = {}
    for name in ("numpy", "scipy", "anuga", "pytest", "torch", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return dict(
        recorded_at=datetime.now(TZ).isoformat(), executable=sys.executable,
        python=sys.version, platform=platform.platform(), cpu_count=os.cpu_count(),
        ram_bytes=memory_capacity(), disk_free_bytes=shutil.disk_usage(root).free,
        packages=packages, sys_path=sys.path, threads={k: os.environ.get(k) for k in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
    )


def new_run(root, kind, arguments, *, formal_test=False):
    root = Path(root)
    run = root / "results" / "runs" / (datetime.now(TZ).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8] + "_" + kind)
    run.mkdir(parents=True, exist_ok=False)
    sources = [root / "run.py", root / "pyproject.toml", root / "requirements.txt"]
    for folder, pattern in (("src", "*.py"), ("scripts", "*.py"), ("tests", "*.py"), ("configs", "*.json"), ("manifests", "*.json"),
                            ("docs/protocols", "*.md")):
        sources.extend(sorted((root / folder).rglob(pattern)))
    hashes = {}
    for source in sources:
        if source.is_file():
            relative = source.relative_to(root)
            destination = run / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            hashes[relative.as_posix()] = sha256(source)
    write_json(run / "source_sha256.json", hashes)
    write_json(run / "environment.json", environment(root))
    write_json(run / "arguments.json", arguments)
    write_json(run / "state.json", dict(status="running", phase=kind, formal_test=bool(formal_test),
                                       started_at=datetime.now(TZ).isoformat()))
    return run


def finish_run(root, run, status="completed", *, formal_test=False, **details):
    state_path = Path(run) / "state.json"
    previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    state = {**previous, **dict(status=status, formal_test=bool(formal_test), finished_at=datetime.now(TZ).isoformat(),
                 peak_process_memory_bytes=peak_memory_bytes(), **details)}
    write_json(Path(run) / "state.json", state)
    write_json(Path(root) / "results" / "latest.json", dict(run=str(run), **state))
