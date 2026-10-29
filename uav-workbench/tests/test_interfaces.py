from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from hyra_uav.core import Config, Memory, Pose, Task
from hyra_uav.interfaces import TaskEnv, merge_arrived
from hyra_uav import experiments
from test_core import TEST_TEMP, bundle


class InterfaceTests(unittest.TestCase):
    def test_equal_time_team_records_have_deterministic_ties(self):
        a=dict(agent_id=1,cells=[0],labels=[1],observed_s=1,arrived_s=1)
        b=dict(a,agent_id=0,labels=[0])
        for observations in ([a,b],[b,a]):
            memory=merge_arrived(Memory((2,2)),observations,1)
            self.assertEqual(memory.labels[0,0],0)
            merge_arrived(memory,[dict(a,observed_s=0)],1)
            self.assertEqual(memory.observed_s[0,0],1)

    def test_reset_reward_done_and_full_coverage_then_staleness(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            labels=np.zeros((5,3,3),int);labels[1:,1,0]=1
            process=bundle(Path(temp)/"p.zip",labels)
            cfg=Config(speed_m_s=1,angular_speed_deg_s=90,range_m=5,fov_deg=60,action_s=1,sample_s=1,duration_s=4)
            env=TaskEnv(process,cfg,Pose(2,1,180))
            initial,_=env.reset(seed=7)
            # A separate panoramic case proves metric remains active after 100% first coverage.
            panoramic=Task(process,replace(cfg,range_m=1.5,fov_deg=360,angular_speed_deg_s=180),Pose(1,1))
            self.assertEqual(panoramic.snapshots[0]["metrics"]["first_coverage"],1)
            panoramic.step("forward")
            self.assertEqual(panoramic.snapshots[-1]["metrics"]["first_coverage"],1)
            self.assertLess(panoramic.snapshots[-1]["metrics"]["correct_coverage"],1)
            panoramic.step("right")
            panoramic.step("forward")
            self.assertEqual(panoramic.snapshots[-1]["metrics"]["correct_coverage"],1)
            for _ in range(4):view,reward,terminated,truncated,info=env.step("hover")
            self.assertFalse(terminated);self.assertTrue(truncated);self.assertEqual(info["time_s"],4)
            again,_=env.reset(seed=7)
            np.testing.assert_array_equal(initial.labels,again.labels)

    def test_interruption_keeps_last_valid_replay(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            root=Path(temp);process=bundle(root/"p.zip",np.zeros((5,3,3),int))
            cfg=Config(action_s=1,sample_s=1,duration_s=4)
            original=Task.step
            def interrupted(task, action):
                if task.time_s==2:raise KeyboardInterrupt("test cancellation")
                return original(task,action)
            with patch.object(Task,"step",interrupted),patch.object(experiments,"OUTPUT",root):
                with self.assertRaises(KeyboardInterrupt):
                    experiments.run_episode(dict(id="test",path=str(process.path)),config=cfg,actions=["hover"]*4)
            import json
            files=list((root/"interrupted").glob("*/manifest.json"))
            self.assertEqual(len(files),1)
            manifest=json.loads(files[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"],"interrupted")
            self.assertEqual(manifest["last_valid_time_s"],2)


if __name__=="__main__":unittest.main()
