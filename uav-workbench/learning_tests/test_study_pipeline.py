from dataclasses import asdict,replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from test_core import TEST_TEMP,bundle
from test_world_model import Labels
from test_training import episode
from hyra_uav.core import Config
from hyra_learning.team_env import TeamEnv,TeamView
from hyra_learning.team_rules import TeamRules
from hyra_learning.networks import SearchAgent
from hyra_learning.training import EpisodeReplay
from hyra_learning.world_model import WorldLearner
from hyra_learning.sciml_study import actor_episode,checkpoint,restore,save,digest,read


class StudyPipelineTests(unittest.TestCase):
    def test_actor_execution_does_not_call_centralized_training_state(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            process=bundle(Path(temp)/'p.zip',np.zeros((5,5,9),dtype=int))
            config=Config(speed_m_s=1,angular_speed_deg_s=90,range_m=1.1,fov_deg=60,action_s=1,sample_s=1,duration_s=4)
            from hyra_learning.team_env import TeamConfig
            plan=dict(config=asdict(config),team=asdict(TeamConfig()))
            with patch.object(TeamEnv,'training_state',side_effect=AssertionError('Actor must not request training truth')):
                metrics,env=actor_episode(SearchAgent(),process,plan)
            self.assertEqual(len(env.actions),4)
            self.assertTrue(np.isfinite(metrics['return_']))

    def test_rule_actions_ignore_wet_content_and_keep_legal_safe_geometry(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            process=bundle(Path(temp)/'p.zip',np.zeros((5,5,9),dtype=int))
            config=Config(speed_m_s=1,angular_speed_deg_s=90,range_m=1.1,fov_deg=60,action_s=1,sample_s=1,duration_s=4)
            env=TeamEnv(process,config);view,_=env.reset()
            agents=[]
            for agent in view.agents:
                labels=agent.labels.copy();labels[labels>=0]=1-labels[labels>=0]
                agents.append(replace(agent,labels=labels))
            altered=TeamView(tuple(agents),view.bed)
            for name in ('R0','R1','R2'):
                a,b=TeamRules(name,view,config),TeamRules(name,altered,config)
                actions=a.choose(view)
                self.assertEqual(actions,b.choose(altered))
                self.assertTrue(a.safe(view.agents,actions))

    def test_replay_reference_checkpoint_resumes_and_detects_shard_corruption(self):
        from test_world_model import WorldModelTests
        fixture=WorldModelTests();fixture.setUp()
        plan=fixture.plan
        torch.manual_seed(29)
        learner=WorldLearner(2,20,plan,'cpu','Sur-QMIX+Physics',Labels(),29)
        replay=EpisodeReplay(8);generator=torch.Generator().manual_seed(31)
        rng=np.random.default_rng(37)
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            folder=Path(temp);references=[]
            for i in range(3):
                data=episode(3,generator);data['case_index']=torch.zeros(4,dtype=torch.long)
                replay.add(data);path=folder/f'episode-{i}.pt';save(path,dict(data=data))
                references.append(dict(path=path.name,sha256=digest(path)))
            learner.train_batch(replay.sample(2,rng,'cpu'))
            checkpoint(folder,learner,replay,rng,1,[],references,[])
            expected=learner.train_batch(replay.sample(2,rng,'cpu'))
            restored=WorldLearner(2,20,plan,'cpu','Sur-QMIX+Physics',Labels(),29)
            buffer,rng2=EpisodeReplay(8),np.random.default_rng(0)
            payload=restore(folder,restored,buffer,rng2)
            self.assertEqual(payload['episode'],1)
            actual=restored.train_batch(buffer.sample(2,rng2,'cpu'))
            self.assertEqual(expected,actual)
            for a,b in zip(learner.parameters,restored.parameters):torch.testing.assert_close(a,b,atol=0,rtol=0)
            with (folder/references[0]['path']).open('ab') as file:file.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Replay shard hash'):
                restore(folder,restored,buffer,rng2)


if __name__=='__main__':unittest.main()
