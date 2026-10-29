"""Run portable checks, optionally followed by existing local-artifact integration.

No training or model selection is performed. Integration creates new interactive
replays and requires the existing generated study inputs and learning runtime.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='检查无人机工作台代码和兼容入口')
    parser.add_argument('--integration', action='store_true', help='另运行依赖本地研究资料的契约及真实回放检查')
    args = parser.parse_args()
    commands = [[sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v']]
    node = shutil.which('node')
    if node:
        commands += [[node, '--check', name] for name in
                     ('console-static/workbench-scale100.js', 'static/app.js', 'team-viewer/app.js')]
    else:
        print('未找到 Node.js，跳过网页脚本语法检查。', flush=True)
    if args.integration:
        commands.append([sys.executable, '-B', '-m', 'unittest', '-v',
                         'workbench_tests', 'console_scale100_checks', 'console_scale100_http_checks'])
        # Three-path checks load Torch in this process; use the separate runtime.
        sys.path.insert(0, str(ROOT))
        from hyra_workbench.jobs import learning_python
        bootstrap = (
            'from hyra_learning.runtime import enable_existing_packages; '
            'enable_existing_packages(); '
            'import unittest; '
            'unittest.main(module=None, argv=["unittest", "-v", "console_signal_routes_checks"])'
        )
        commands.append([str(learning_python()), '-B', '-c', bootstrap])
    for command in commands:
        print('运行：' + ' '.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
