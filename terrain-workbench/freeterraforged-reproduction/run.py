"""Build and run the original FreeTerraForged grid exporter."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from urllib.request import getproxies
from urllib.parse import urlparse
from pathlib import Path
import subprocess


def java_major(executable):
    try:
        result = subprocess.run([str(executable), '-version'], capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=10)
        match = re.search(r'version "(\d+)', result.stdout + result.stderr)
        return int(match[1]) if result.returncode == 0 and match else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def java_home(root, explicit=None, environment=None):
    env = os.environ if environment is None else environment
    executable = 'java.exe' if os.name == 'nt' else 'java'
    if explicit:
        home = explicit.resolve()
        if java_major(home / 'bin' / executable) != 21 or not (home / 'bin' / ('javac.exe' if os.name == 'nt' else 'javac')).is_file():
            raise RuntimeError('--java-home 必须指向 Java 21 JDK')
        return home
    candidates = list((root / '.runtime' / 'java').glob('*/Contents/Home'))
    candidates += list((root / '.runtime' / 'java').glob('*'))
    candidates += list((root.parent.parent / 'work' / 'runtime').glob('*/Contents/Home'))
    if env.get('JAVA_HOME'): candidates.append(Path(env['JAVA_HOME']))
    on_path = shutil.which(executable, path=env.get('PATH'))
    if on_path: candidates.append(Path(on_path).resolve().parent.parent)
    seen = set()
    versions = []
    for home in candidates:
        home = home.resolve()
        if home in seen: continue
        seen.add(home)
        major = java_major(home / 'bin' / executable)
        if major == 21 and (home / 'bin' / ('javac.exe' if os.name == 'nt' else 'javac')).is_file():
            return home
        if major: versions.append(str(major))
    found = '；当前找到 Java ' + '、'.join(sorted(set(versions))) if versions else ''
    raise RuntimeError('生成新地图需要 Java 21 JDK' + found + '。请设置 JAVA_HOME 为 Java 21，或把解压的 JDK 放入生成器 .runtime/java/。示例地图仍可使用。')


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
    try: home = java_home(root, args.java_home, env)
    except RuntimeError as error: raise SystemExit(str(error))
    env['JAVA_HOME'] = str(home)
    env['PATH'] = str(home / 'bin') + os.pathsep + env.get('PATH', '')
    # Java does not automatically use the desktop's HTTP proxy for Maven downloads.
    java_options = env.get('JAVA_TOOL_OPTIONS', '')
    proxy_options = []
    if 'proxyHost' not in java_options:
        proxies = getproxies()
        for scheme in ('http', 'https'):
            address = proxies.get(scheme) or proxies.get('http')
            proxy = urlparse(address if address and '://' in address else 'http://' + (address or ''))
            if proxy.hostname and proxy.scheme in ('http', 'https') and not proxy.username:
                proxy_options += [f'-D{scheme}.proxyHost={proxy.hostname}', f'-D{scheme}.proxyPort={proxy.port or 80}']
        if proxy_options:
            env['JAVA_TOOL_OPTIONS'] = ' '.join([java_options, *proxy_options]).strip()
            print('HYRA_STAGE: 构建下载已使用系统网络代理', flush=True)
    print('HYRA_STAGE: Java 21 已就绪，正在准备构建依赖', flush=True)
    if args.gradle_cache:
        env["GRADLE_USER_HOME"] = str(args.gradle_cache.resolve())
    elif env.get("GRADLE_USER_HOME"):
        # Keep an explicitly selected cache, including one outside the checkout.
        env["GRADLE_USER_HOME"] = str(Path(env["GRADLE_USER_HOME"]).expanduser().resolve())
    elif (root.parent.parent / "work" / "gradle-cache").is_dir():
        env["GRADLE_USER_HOME"] = str((root.parent.parent / "work" / "gradle-cache").resolve())
    else:
        env["GRADLE_USER_HOME"] = str(root / ".runtime" / "gradle")
    print('HYRA_STAGE: Gradle cache: ' + env["GRADLE_USER_HOME"], flush=True)
    if args.request:
        request = args.request.resolve()
        # Passing the path as an explicit Gradle property preserves relative output semantics.
        extra = ["-PreproductionRequest=" + str(request)]
    else:
        extra = []
    if args.action == 'export':
        # Loom's supported CI mode skips IDE source downloads/remapping; runtime
        # dependencies and upstream terrain generation are still built normally.
        extra += ['-Dfabric.loom.ci=true']
    tasks = {
        "build": [":fabric:remapJar"],
        "export": [":fabric:runExportTerrain"],
        "test": [":common:test", ":fabric:test"],
    }[args.action]
    wrapper = root / ("gradlew.bat" if os.name == "nt" else "gradlew")
    portable = root / '.runtime' / 'gradle-install' / 'gradle-8.11' / 'bin' / ('gradle.bat' if os.name == 'nt' else 'gradle')
    if portable.is_file(): wrapper = portable
    process = subprocess.run([str(wrapper), *tasks, *extra, *proxy_options, '--console=plain', '--info', '--no-daemon',
                              '-Dorg.gradle.internal.http.connectionTimeout=15000',
                              '-Dorg.gradle.internal.http.socketTimeout=30000',
                              '-Djava.net.preferIPv4Stack=true'], cwd=root, env=env)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
