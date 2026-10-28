"""Real UI-worker checks against the frozen evaluator, using development only."""
import sys
import json
import subprocess
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
import console_signal_routes_contract as contract


class SignalRoutesChecks(unittest.TestCase):
    def request(self,method='R1',mode='model'):
        scene=next(r for r in contract.scene_records() if r['split']=='development')
        return dict(scene=scene['id'],task='exploration_rescue',group=method,mode=mode,training_seed=11,
            config=contract.default_config(group=method,seed=11))

    def test_contract_keeps_server_free_of_learning_imports(self):
        subprocess.run([sys.executable,'-B','-c',
            'import sys; import console_signal_routes_contract; assert not any(k in sys.modules for k in ("numpy","torch","signal_routes"))'],
            cwd=contract.ROOT,check=True)

    def test_catalog_and_all_selected_model_bindings(self):
        catalog=contract.catalog(); self.assertTrue(catalog['released']); self.assertEqual(len(catalog['groups']),8)
        self.assertEqual(len(contract.scene_records()),18)
        self.assertTrue(all(s['split'] in ('fit','development') for s in contract.scene_records()))
        for method in contract.LABELS:
            for seed in (11,29,47):
                path,meta,selected=contract.model_record(method,seed)
                self.assertEqual(meta['method'],method); self.assertEqual(meta['seed'],seed)
                self.assertEqual(path.name,f"actor-{selected['episode']:04d}.pt")

    def test_invalid_task_holdout_config_and_checkpoint_are_rejected(self):
        for changes in (dict(task='exploration'),dict(scene='signal-routes:signal-parent-7301:slow'),
                dict(training_seed=True),dict(group='B3'),dict(config={'duration_s':250}),dict(checkpoint='other')):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                contract.validate({**self.request(),**changes})
        with patch.object(contract,'release_record',side_effect=ValueError('unreleased')):
            with self.assertRaisesRegex(ValueError,'unreleased'): contract.validate(self.request())

    def test_real_three_path_rollouts_match_frozen_evaluator(self):
        from console_signal_routes_runner import execute_job
        from signal_routes.evaluate import actor_run
        from signal_routes.common import sources,write
        from scale100_train import FloodCase
        from tools.signal_routes_result_io import install_result_io_adapter
        install_result_io_adapter()
        output=contract.ROOT/'generated/signal-routes-console-checks'/uuid.uuid4().hex
        output.mkdir(parents=True); rows=[]; before=sources()
        for method in ('R1','R2','R3'):
            request=self.request(method); replay=execute_job(request); checked=contract.validate(request)
            path,meta,selected=contract.model_record(method,11)
            saved=output/(method+'-frozen.json')
            actor_run(contract.STUDY,path,method,11,FloodCase(checked['scene']['path']),saved)
            original=contract.read(saved)
            self.assertEqual(replay['actions'],original['actions'])
            self.assertEqual(replay['observations'],original['observations'])
            self.assertEqual(replay['events'],original['events'])
            self.assertEqual(replay['execution']['return_'],original['return_'])
            self.assertEqual(replay['execution']['signal_route'],method)
            self.assertEqual(replay['execution']['actor_sha256'],meta['actor_sha256'])
            self.assertEqual(replay['execution']['selected_episode'],selected['episode'])
            self.assertEqual(len(replay['snapshots']),181)
            self.assertTrue(all(len(s['signal_posterior'])==400 for s in replay['snapshots']))
            self.assertTrue(replay['audit']['audit_passed'] and replay['audit']['frozen_route_replay_equal'])
            rows.append(dict(method=method,actor_sha256=meta['actor_sha256'],frozen_actions_records_events_return_equal=True))
        rule=execute_job(self.request(mode='rule'))
        self.assertIsNone(rule['group']); self.assertIsNone(rule['execution']['actor_sha256'])
        self.assertTrue(rule['audit']['audit_passed'])
        self.assertEqual(before,sources())
        write(output/'verification.json',dict(passed=True,development_only=True,rows=rows,sources=before))
        print('UI-worker evidence:',output)



class HttpSignalRoutesChecks(unittest.TestCase):
    def test_real_http_route_dispatch_and_saved_replay(self):
        import threading
        from http.server import ThreadingHTTPServer
        from urllib.request import urlopen,Request
        from console_scale100_app import Handler
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        base='http://127.0.0.1:'+str(server.server_port)
        try:
            with urlopen(base+'/api/workbench/signal-routes/catalog') as response:
                self.assertTrue(json.load(response)['released'])
            with urlopen(base+'/') as response:
                self.assertIn('studyProfile',response.read().decode())
            for method in ('R1','R2','R3'):
                scene=next(s for s in contract.scene_records() if s['split']=='development')
                data=dict(scene=scene['id'],task='exploration_rescue',group=method,mode='model',training_seed=29)
                request=Request(base+'/api/workbench/run',data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
                with urlopen(request,timeout=120) as response: output=json.load(response)
                with urlopen(base+output['replay_url']) as response: replay=json.load(response)
                self.assertEqual(replay['execution']['signal_route'],method)
                self.assertEqual(replay['group'],method)
                self.assertEqual(replay['execution']['training_seed'],29)
                self.assertTrue(replay['audit']['frozen_route_replay_equal'])
                self.assertEqual(len(replay['snapshots']),181)
                self.assertEqual(len(replay['observations']),362)
                with urlopen(base+'/runs/'+output['run_id']+'/audit.json') as response:
                    self.assertTrue(json.load(response)['audit_passed'])
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__=='__main__': unittest.main(verbosity=2)
