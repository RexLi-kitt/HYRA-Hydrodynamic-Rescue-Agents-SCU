import numpy as np
import pytest
import torch
from hyra_validation.agents import SearchAgent


def test_training_enhancement_and_execution_guard():
    torch.manual_seed(1)
    agent=SearchAgent((33,5,5))
    obs=torch.randn(4,33,5,5)
    previous=torch.zeros(4,8);hidden=torch.zeros(4,64)
    agent.train()
    q,h=agent(obs,previous,0,hidden,enhance=True)
    q.square().mean().backward()
    assert torch.isfinite(agent.encoder.weight.grad).all()
    agent.eval()
    with pytest.raises(RuntimeError,match="training"):
        agent(obs,previous,0,hidden,enhance=True)
    with torch.no_grad():
        assert agent(obs,previous,0,hidden,enhance=False)[0].shape==(4,8)
