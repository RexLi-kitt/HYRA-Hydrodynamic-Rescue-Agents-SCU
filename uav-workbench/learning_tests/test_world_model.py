import io
import json
from pathlib import Path
import unittest

from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
import torch

from hyra_learning.training import EpisodeReplay
from hyra_learning.physics import SWEOperator
from hyra_learning.world_model import WorldLearner, clean_recurrent_context, conservation_loss
from test_training import episode


class Labels:
    def batch(self,case_indices,time_indices,device):
        n,nz,nx = len(case_indices),5,6
        current = torch.zeros(n,3,nz,nx,device=device)
        current[:,0] = .03
        return dict(current=current,future=current.clone(),rate=torch.zeros_like(current),
                    dt=torch.full((n,),.01,dtype=torch.float64,device=device),
                    bed=torch.zeros(n,nz,nx,dtype=torch.float64,device=device),source=torch.zeros(n,nz,nx,device=device),
                    drains=torch.zeros(n,nz,nx,device=device),manning=torch.full((n,nz,nx),.035,device=device),
                    solid=torch.zeros(n,nz,nx,dtype=torch.bool,device=device),
                    wall_x=torch.zeros(n,nz,nx+1,dtype=torch.bool,device=device),
                    wall_y=torch.zeros(n,nz+1,nx,dtype=torch.bool,device=device))


class WorldModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        root=Path(__file__).resolve().parents[1]
        self.plan=json.loads((root/'learning_configs/sur_preflight_20261025.json').read_text(encoding='utf-8'))
        self.plan['auxiliary']=json.loads((root/'learning_configs/sciml_round1_design_20261025.json').read_text(encoding='utf-8'))['auxiliary']
        self.replay=EpisodeReplay(8)
        generator=torch.Generator().manual_seed(47)
        for count in (3,4,2,4):
            data=episode(count,generator)
            data['case_index']=torch.zeros(count+1,dtype=torch.long)
            self.replay.add(data)

    def learner(self,method):
        torch.manual_seed(11)
        return WorldLearner(2,20,self.plan,'cpu',method,Labels(),11)

    def test_context_never_uses_future_observations(self):
        learner=self.learner('Sur-QMIX+Data')
        batch=self.replay.sample(2,np.random.default_rng(11),'cpu')
        baseline=clean_recurrent_context(learner.actor,batch)
        changed={key:value.clone() for key,value in batch.items()}
        changed['images'][:,3:]=100
        changed['poses'][:,3:]=100
        altered=clean_recurrent_context(learner.actor,changed)
        torch.testing.assert_close(baseline[:,:3],altered[:,:3],atol=0,rtol=0)

    def test_auxiliary_gradients_reach_shared_encoder_and_gru(self):
        for method in ('Sur-QMIX+Data','Sur-QMIX+Physics'):
            learner=self.learner(method)
            loss,diagnostics=learner.additional_loss(self.replay.sample(2,np.random.default_rng(11),'cpu'))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(learner.actor.image_embed.weight.grad.abs().sum()),0)
            self.assertGreater(float(learner.actor.gru.weight_ih.grad.abs().sum()),0)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in learner.parameters if p.grad is not None))
            if method.endswith('Physics'):
                self.assertGreater(diagnostics['auxiliary_physics_loss'],0)

    def test_data_and_physics_heads_match_and_resume_including_auxiliary_rng(self):
        a,b=self.learner('Sur-QMIX+Data'),self.learner('Sur-QMIX+Physics')
        self.assertEqual(sum(p.numel() for p in a.parameters),sum(p.numel() for p in b.parameters))
        for left,right in zip(a.world.parameters(),b.world.parameters()):
            torch.testing.assert_close(left,right,atol=0,rtol=0)
        for method in ('Sur-QMIX+Data','Sur-QMIX+Physics'):
            learner=self.learner(method)
            rng=np.random.default_rng(29)
            learner.train_batch(self.replay.sample(2,rng,'cpu'))
            stream=io.BytesIO()
            torch.save(learner.checkpoint(self.replay,rng,1,[]),stream)
            expected=learner.train_batch(self.replay.sample(2,rng,'cpu'))
            stream.seek(0)
            restored=self.learner(method)
            replay,rng2=EpisodeReplay(8),np.random.default_rng(0)
            restored.restore(torch.load(stream,weights_only=True),replay,rng2)
            actual=restored.train_batch(replay.sample(2,rng2,'cpu'))
            self.assertEqual(expected,actual)
            for left,right in zip(learner.parameters,restored.parameters):
                torch.testing.assert_close(left,right,atol=0,rtol=0)

    def test_constraint_is_zero_for_exact_discrete_rate_and_detects_corruption(self):
        labels=Labels().batch([0,0],[0,1],'cpu')
        labels['current'][:,0]=.2
        labels['source'][:,2,3]=.1
        current=labels['current']
        evolved=SWEOperator().step(current,labels['bed'],.01,source=labels['source'])
        rate=(evolved-current)/.01
        predictions=dict(current=current,rate=rate,future=current)
        self.assertEqual(float(conservation_loss(predictions,labels)),0.)
        changed=dict(predictions,rate=rate+.1)
        self.assertGreater(float(conservation_loss(changed,labels)),0)


if __name__ == '__main__':
    unittest.main()
