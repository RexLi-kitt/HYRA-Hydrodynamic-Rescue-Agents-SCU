import numpy as np
import torch
from hyra_validation.adapters.pymarl import components


def test_pymarl_episode_replay_preserves_values():
    EpisodeBatch,ReplayBuffer,_=components()
    scheme=dict(obs=dict(vshape=(3,),group="agents"),actions=dict(vshape=(1,),group="agents",dtype=torch.long))
    groups=dict(agents=2)
    batch=EpisodeBatch(scheme,groups,1,3)
    observations=np.array([[[1,2,3],[4,5,6]]],dtype=float)
    batch.update(dict(obs=observations,actions=[[[2],[5]]]),ts=0)
    buffer=ReplayBuffer(scheme,groups,2,3)
    buffer.insert_episode_batch(batch)
    recovered=buffer.sample(1)
    np.testing.assert_array_equal(recovered["obs"][0,0].numpy(),observations[0])
    assert recovered["actions"][0,0,:,0].tolist()==[2,5]
    assert recovered["filled"][0,:,0].tolist()==[1,0,0]
