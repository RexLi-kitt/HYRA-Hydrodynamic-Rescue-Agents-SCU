"""Regression coverage for the stuck first-generation failure."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import generation_process

spec = importlib.util.spec_from_file_location('generator_runner', ROOT.parent / 'freeterraforged-reproduction/run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class GenerationTests(unittest.TestCase):
    def supervised(self, script, **options):
        with tempfile.TemporaryDirectory() as folder:
            chunks = []
            code = generation_process.run([sys.executable, '-u', '-c', script], folder,
                Path(folder) / 'output.log', lambda text, elapsed, idle: chunks.append(text),
                options.pop('cancelled', lambda: False), **options)
            return code, ''.join(chunks)

    def test_unterminated_download_output(self):
        code, text = self.supervised("import sys,time;sys.stdout.write('Downloading ...');sys.stdout.flush();time.sleep(.5);print('done')")
        self.assertEqual(code, 0)
        self.assertIn('Downloading ...done', text)

    def test_quiet_process_times_out(self):
        start = time.monotonic()
        with self.assertRaises(TimeoutError):
            self.supervised('import time;time.sleep(30)', idle_timeout=.3)
        self.assertLess(time.monotonic()-start, 8)

    def test_cancel_terminates_process(self):
        start = time.monotonic()
        with self.assertRaises(InterruptedError):
            self.supervised('import time;time.sleep(30)', cancelled=lambda: time.monotonic()-start > .3)
        self.assertLess(time.monotonic()-start, 8)

    def test_cancel_terminates_child_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            heartbeat = Path(folder) / 'heartbeat'
            child = "import time;from pathlib import Path;p=Path('heartbeat');\nwhile True:p.write_text(str(time.time()));time.sleep(.1)"
            parent = f'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",{child!r}]);time.sleep(30)'
            start = time.monotonic()
            with self.assertRaises(InterruptedError):
                generation_process.run([sys.executable, '-c', parent], folder, Path(folder) / 'output.log',
                    lambda *args: None, lambda: heartbeat.exists() and time.monotonic()-start > .7)
            last = heartbeat.read_text()
            time.sleep(.4)
            self.assertEqual(heartbeat.read_text(), last)

    def test_exit_failure_is_preserved(self):
        code, text = self.supervised("import sys;print('dependency failed');sys.exit(7)")
        self.assertEqual(code, 7)
        self.assertIn('dependency failed', text)

    def test_java24_is_rejected_before_gradle(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(runner, 'java_major', return_value=24):
            with self.assertRaisesRegex(RuntimeError, 'Java 21'):
                runner.java_home(Path(folder), Path(folder))

    def test_portable_java21_preferred_to_global_java24(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            local = root / '.runtime/java/jdk-21'
            (local / 'bin').mkdir(parents=True)
            (local / 'bin' / ('javac.exe' if sys.platform == 'win32' else 'javac')).touch()
            with patch.object(runner, 'java_major', side_effect=lambda p: 21 if local in p.parents else 24):
                self.assertEqual(runner.java_home(root, environment={'JAVA_HOME': str(root / 'global24'), 'PATH': ''}), local.resolve())


if __name__ == '__main__': unittest.main()
