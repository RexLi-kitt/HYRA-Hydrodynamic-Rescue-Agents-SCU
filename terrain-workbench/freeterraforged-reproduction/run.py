"""Build and run the original FreeTerraForged grid exporter."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "export", "test", "verify-source"])
    parser.add_argument("--java-home", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--gradle-cache", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    if args.action == "verify-source":
        from load_grid import verify_upstream

        report = verify_upstream(root)
        print(json.dumps(report, indent=2))
        raise SystemExit(bool(report["mismatches"]))
    env = dict(os.environ)
    if args.java_home:
        env["JAVA_HOME"] = str(args.java_home.resolve())
    else:
        # Use the isolated runtime when run in the workspace that built this artifact.
        runtimes = list((root.parent.parent / "work" / "runtime").glob("*/Contents/Home"))
        if runtimes:
            env["JAVA_HOME"] = str(runtimes[0].resolve())
    if args.gradle_cache:
        env["GRADLE_USER_HOME"] = str(args.gradle_cache.resolve())
    elif (root.parent.parent / "work" / "gradle-cache").is_dir():
        env["GRADLE_USER_HOME"] = str((root.parent.parent / "work" / "gradle-cache").resolve())
    else:
        env["GRADLE_USER_HOME"] = str(root / ".runtime" / "gradle")
    if args.request:
        request = args.request.resolve()
        # Passing the path as an explicit Gradle property preserves relative output semantics.
        extra = ["-PreproductionRequest=" + str(request)]
    else:
        extra = []
    tasks = {
        "build": [":fabric:remapJar"],
        "export": [":fabric:runExportTerrain"],
        "test": [":common:test", ":fabric:test"],
    }[args.action]
    wrapper = root / ("gradlew.bat" if os.name == "nt" else "gradlew")
    process = subprocess.run([str(wrapper), *tasks, *extra, "--console=plain", "--no-daemon"], cwd=root, env=env)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
