"""HTTP integration against the real worker; fit/development scenes only."""
import json
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from http.server import ThreadingHTTPServer
from console_scale100_app import Handler
from console_scale100_contract import default_config


class HttpChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = 'http://127.0.0.1:' + str(cls.server.server_port)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def get(self, path):
        with urlopen(self.base + path, timeout=90) as response:
            return json.load(response)

    def post(self, data, origin=None):
        headers = {'Content-Type': 'application/json'}
        if origin:
            headers['Origin'] = origin
        request = Request(self.base + '/api/workbench/run',
                          data=json.dumps(data).encode(), headers=headers)
        with urlopen(request, timeout=90) as response:
            return json.load(response)

    def request_data(self, task):
        scene = next(s for s in self.get('/api/workbench/scenes')
                     if s.get('kind') == 'synthetic_scale100_swe' and s['split'] == 'development')
        return dict(scene=scene['id'], task=task, group='P', mode='rule', training_seed=29,
                    config=default_config(task, 'P', seed=47))

    def test_catalog_has_both_profiles_and_three_active_groups(self):
        self.assertEqual(self.get('/api/console-version')['action_count'], 8)
        primary = self.get('/api/workbench/scale100/catalog')
        self.assertEqual(primary['profile'], 'scale100')
        self.assertEqual([g['id'] for g in primary['groups']], ['B3', 'D', 'P'])
        self.assertEqual(primary['default_training_seed'], 11)
        self.assertEqual(primary['defaults']['exploration']['duration_s'], 900)
        self.assertTrue(all(set(g['tasks']) == {'exploration', 'exploration_rescue'}
                            for g in primary['groups']))
        self.assertEqual(self.get('/api/workbench/catalog')['version'], 'hyra-eight-workbench-v3')

    def test_maps_preserve_native_dimensions_and_hide_new_holdout(self):
        scenes = self.get('/api/workbench/scenes')
        self.assertEqual(sum(s.get('kind') == 'synthetic_scale100_swe' for s in scenes), 4)
        self.assertFalse(any(s['id'] == 'scale100:procedural-parent-503' for s in scenes))
        large = next(s for s in scenes if s['id'] == 'flood:maps-06cd1ab8313d')
        self.assertEqual(large['grid']['shape'], [256, 256])
        self.assertEqual(large['grid']['dx_m'], 4)

    def test_real_two_task_replays_and_downloads(self):
        for task in ('exploration', 'exploration_rescue'):
            data = self.request_data(task)
            output = self.post(data)
            replay = self.get(output['replay_url'])
            self.assertEqual(replay['contract'], 'scale100-900s-v1')
            self.assertEqual(replay['grid']['shape'], [20, 20])
            self.assertEqual(replay['task'], task)
            self.assertIsNone(replay['group'])
            self.assertEqual(replay['requested_group'], 'P')
            self.assertIsNone(replay['execution']['training_seed'])
            self.assertEqual(replay['execution']['scenario_seed'], 47)
            self.assertEqual(replay['rescue_config']['duration_s'], 900)
            self.assertEqual(len(replay['snapshots']), 181)
            self.assertEqual(len(replay['observations']), 362)
            self.assertTrue(replay['audit']['audit_passed'])
            if task == 'exploration':
                self.assertEqual(replay['summary']['total_targets'], 0)
                self.assertIsNone(replay['summary']['rescue_fraction'])
            else:
                self.assertEqual(replay['summary']['total_targets'], 5)
            for name in ('actions', 'observations', 'audit', 'manifest'):
                self.assertIsNotNone(self.get('/runs/' + output['run_id'] + '/' + name + '.json'))

    def test_bad_input_and_cross_origin_are_rejected(self):
        for data, origin, code in (([], None, 400),
                                   (dict(self.request_data('exploration'), config=dict(duration_s=250)), None, 400),
                                   (self.request_data('exploration'), 'http://example.invalid', 403)):
            with self.assertRaises(HTTPError) as error:
                self.post(data, origin)
            self.assertEqual(error.exception.code, code)

    def test_unpublished_model_request_is_rejected(self):
        # This gate is exercised only before publication. Later tests use the
        # separate contract test's explicit unreleased stub.
        if self.get('/api/workbench/scale100/catalog')['released']:
            self.skipTest('Actual formal release already exists')
        data = self.request_data('exploration')
        data['mode'] = 'model'
        with self.assertRaises(HTTPError) as error:
            self.post(data)
        self.assertEqual(error.exception.code, 400)

    def test_four_direction_routes_remain_retired(self):
        for path in ('/api/run', '/api/catalog'):
            with self.assertRaises(HTTPError) as error:
                self.get(path)
            self.assertEqual(error.exception.code, 410)


if __name__ == '__main__':
    unittest.main(verbosity=2)
