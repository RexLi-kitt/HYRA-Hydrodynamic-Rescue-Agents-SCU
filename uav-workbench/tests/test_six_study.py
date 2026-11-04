import unittest
import json
import threading
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from unittest.mock import patch
from hyra_workbench import six_study as s
from hyra_workbench import server
from hyra_workbench.jobs import JobRunner, JobBusy


class SixStudyTests(unittest.TestCase):
    candidates = [dict(id='mat', family='single'), dict(id='maac_coma', family='pair')]
    def check(self, request):
        return s.validate(request, [{}], self.candidates, [{}])

    def test_record_index_is_strict_and_no_arbitrary_file(self):
        self.assertEqual(self.check(dict(mode='record', index=0)), dict(mode='record', index=0))
        for index in [-1, 1, True, '0', '../../file']:
            with self.assertRaises(ValueError):
                self.check(dict(mode='record', index=index))

    def test_demo_cannot_use_test_scene_or_unregistered_weights(self):
        for request in [dict(mode='demo', candidate='mat', case=1),
                        dict(mode='demo', candidate='../../model.pt', case=0),
                        dict(mode='demo', candidate='mat', case=0, assignment=1),
                        dict(mode='demo', candidate='mat', case=True),
                        dict(mode='demo', candidate='mat', case=0, assignment=True)]:
            with self.assertRaises(ValueError): self.check(request)
        self.assertEqual(self.check(dict(mode='demo', candidate='maac_coma', case=0, assignment=1))['assignment'], 1)

    def test_replay_path_cannot_escape_generated_folder(self):
        for name in ['../config.json', 'a'*32+'.json/..', 'G'*32+'.json', '/etc/passwd']:
            with self.assertRaises(ValueError): s.replay_file(name)
        self.assertEqual(s.replay_file('a'*32+'.json').name, 'replay.json')

    def test_missing_runtime_never_starts_process(self):
        with patch.object(s, 'local_config', return_value=None), patch.object(s.subprocess, 'run') as run:
            with self.assertRaises(ValueError): s.execute(dict(mode='record', index=0), JobRunner())
            run.assert_not_called()

    def test_shared_job_lock_prevents_overlapping_inference(self):
        jobs=JobRunner();jobs.lock.acquire()
        try:
            with patch.object(s, 'local_config', return_value={'study':'.'}), \
                 patch.object(s, 'records', return_value=[{}]), \
                 patch.object(s, 'read', return_value={'candidates':self.candidates}), \
                 patch.object(s, 'dev_cases', return_value=[{}]), patch.object(s.subprocess, 'run') as run:
                with self.assertRaises(JobBusy): s.execute(dict(mode='record', index=0), jobs)
                run.assert_not_called()
        finally: jobs.lock.release()

class SixHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host=ThreadingHTTPServer(('127.0.0.1',0),server.Handler)
        cls.thread=threading.Thread(target=cls.host.serve_forever,daemon=True);cls.thread.start()
        cls.url='http://127.0.0.1:'+str(cls.host.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.host.shutdown();cls.host.server_close();cls.thread.join()

    def test_catalog_routes_and_origin_rejection(self):
        with patch.object(s,'catalogue',return_value={'records':[{'index':0}]}) as catalog:
            with urlopen(self.url+'/api/six-study/catalog') as r:
                self.assertEqual(json.load(r)['records'],[{'index':0}])
            catalog.assert_called_once()
        with patch.object(s,'execute') as execute:
            req=Request(self.url+'/api/six-study/run',data=b'{}',headers={'Origin':'https://unrelated.example'})
            with self.assertRaises(HTTPError) as cm: urlopen(req)
            self.assertEqual(cm.exception.code,403);execute.assert_not_called()

    def test_busy_and_validation_return_actionable_errors(self):
        for error,status in [(JobBusy('busy'),409),(ValueError('invalid'),400)]:
            with patch.object(s,'execute',side_effect=error):
                with self.assertRaises(HTTPError) as cm:
                    urlopen(Request(self.url+'/api/six-study/run',data=b'{}'))
                self.assertEqual(cm.exception.code,status)

if __name__ == '__main__': unittest.main()
