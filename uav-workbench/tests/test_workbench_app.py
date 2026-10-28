"""Current workbench regression: dispatch, process lifecycle and HTTP contract."""
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from hyra_workbench import jobs, profiles, server
from hyra_workbench.paths import ROOT, STATIC


WORKER = '''import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--job', type=Path)
p.add_argument('--run-id')
a = p.parse_args()
job = json.loads(a.job.read_text(encoding='utf-8'))
identity = a.run_id or job['run_id']
folder = Path(__file__).parent / 'generated/web' / identity
folder.mkdir(parents=True)
summary = dict(scene=job['scene'], validated=job.get('validated', False))
(folder / 'replay.json').write_text(json.dumps(dict(summary=summary)), encoding='utf-8')
print('completed')
'''


class JobTests(unittest.TestCase):
    def test_all_profiles_use_their_worker_and_payload_contract(self):
        for profile, scene in zip(profiles.PROFILES, ('scale100:demo', 'signal-routes:demo', 'flood:demo')):
            with self.subTest(profile=profile.name), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / profile.runner).write_text(WORKER, encoding='utf-8')
                contract = SimpleNamespace(validate=lambda request: dict(scene=request['scene'], validated=True))
                with patch.object(profiles, 'import_module', return_value=contract), \
                        patch.object(jobs, 'learning_python', return_value=Path(sys.executable)):
                    runner = jobs.JobRunner(root)
                    result = runner.execute(dict(scene=scene))
                self.assertEqual(result['summary']['scene'], scene)
                self.assertEqual(result['summary']['validated'], profile.name == 'original_grid')
                self.assertEqual(result['replay_url'], '/runs/' + result['run_id'] + '/replay.json')
                self.assertIn('completed', (root / 'generated/workbench-jobs' / (result['run_id'] + '.log')).read_text())
                self.assertFalse(runner.lock.locked())

    def test_invalid_requests_never_create_job_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(ValueError):
                jobs.JobRunner(root).execute([])
            self.assertEqual(list(root.iterdir()), [])

    def test_busy_runner_never_starts_another_worker(self):
        runner = jobs.JobRunner()
        runner.lock.acquire()
        contract = SimpleNamespace(validate=lambda request: {})
        try:
            with patch.object(profiles, 'import_module', return_value=contract), \
                    patch.object(jobs.subprocess, 'run') as run:
                with self.assertRaises(jobs.JobBusy):
                    runner.execute(dict(scene='scale100:test'))
                run.assert_not_called()
        finally:
            runner.lock.release()

    def test_worker_failure_and_timeout_release_lock(self):
        for code, error in (("import sys; print('failed', file=sys.stderr); sys.exit(1)", RuntimeError),
                            ('import time; time.sleep(10)', subprocess.TimeoutExpired)):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / 'console_scale100_runner.py').write_text(code, encoding='utf-8')
                runner = jobs.JobRunner(root, timeout=.2 if error is subprocess.TimeoutExpired else 10)
                contract = SimpleNamespace(validate=lambda request: {})
                with patch.object(profiles, 'import_module', return_value=contract), \
                        patch.object(jobs, 'learning_python', return_value=Path(sys.executable)):
                    with self.assertRaises(error):
                        runner.execute(dict(scene='scale100:test'))
                self.assertFalse(runner.lock.locked())
                if error is RuntimeError:
                    log = next((root / 'generated/workbench-jobs').glob('*.log'))
                    self.assertIn('failed', log.read_text())

    def test_invalid_explicit_runtime_does_not_silently_fall_back(self):
        with patch.dict(jobs.os.environ, {'HYRA_LEARNING_PYTHON': str(ROOT / 'no-such-python')}):
            with self.assertRaisesRegex(ValueError, 'HYRA_LEARNING_PYTHON'):
                jobs.learning_python()


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.http = ThreadingHTTPServer(('127.0.0.1', 0), server.Handler)
        cls.thread = threading.Thread(target=cls.http.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f'http://127.0.0.1:{cls.http.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.http.shutdown()
        cls.http.server_close()
        cls.thread.join()

    def test_static_assets_and_old_page_aliases_are_identical(self):
        for path, (name, mime) in server.STATIC_ROUTES.items():
            with self.subTest(path=path), urlopen(self.base + path, timeout=10) as response:
                self.assertEqual(response.read(), (STATIC / name).read_bytes())
                self.assertEqual(response.headers['Content-Type'], mime)

    def test_version_and_compatibility_imports(self):
        import console_app
        import console_scale100_app
        self.assertIs(console_app.Handler, server.Handler)
        self.assertIs(console_scale100_app.Handler, server.Handler)
        with urlopen(self.base + '/api/console-version', timeout=10) as response:
            value = json.load(response)
        self.assertEqual(value['action_count'], 8)
        self.assertEqual(value['profiles'], ['scale100', 'signal_routes', 'original_grid'])

    def test_retired_routes_and_unregistered_artifacts_stay_unavailable(self):
        for path, status in (('/api/run', 410), ('/api/catalog', 410),
                             ('/runs/../../README.md', 404), ('/missing.html', 404)):
            with self.subTest(path=path), self.assertRaises(HTTPError) as error:
                urlopen(self.base + path, timeout=10)
            self.assertEqual(error.exception.code, status)

    def test_catalog_routes_keep_their_registered_profiles(self):
        for path, profile in server.CATALOG_ROUTES.items():
            with patch.object(profiles, 'catalog', return_value={'profile': profile}) as catalog:
                with urlopen(self.base + path, timeout=10) as response:
                    self.assertEqual(json.load(response)['profile'], profile)
                catalog.assert_called_once_with(profile)

    def test_post_rejects_malformed_body_and_cross_origin_before_execution(self):
        for raw, origin, status in ((b'{', None, 400), (b'', None, 400),
                                    (b'{}', 'http://example.invalid', 403)):
            request = Request(self.base + '/api/workbench/run', data=raw,
                              headers={'Origin': origin} if origin else {})
            with patch.object(server.JOBS, 'execute') as execute:
                with self.assertRaises(HTTPError) as error:
                    urlopen(request, timeout=10)
                self.assertEqual(error.exception.code, status)
                execute.assert_not_called()

    def test_job_errors_keep_http_status_and_success_response(self):
        for failure, status in ((ValueError('invalid'), 400), (jobs.JobBusy('busy'), 409),
                                (subprocess.TimeoutExpired('worker', 180), 504),
                                (RuntimeError('failed'), 500)):
            with self.subTest(status=status), patch.object(server.JOBS, 'execute', side_effect=failure):
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(self.base + '/api/workbench/run', data=b'{}'), timeout=10)
                self.assertEqual(error.exception.code, status)
        with patch.object(server.JOBS, 'execute', return_value={'run_id': 'test'}) as execute:
            with urlopen(Request(self.base + '/api/workbench/run', data=b'{"scene":"demo"}'), timeout=10) as response:
                self.assertEqual(json.load(response), {'run_id': 'test'})
            execute.assert_called_once_with({'scene': 'demo'})
