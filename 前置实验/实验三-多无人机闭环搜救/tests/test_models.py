import numpy as np
import pytest
import torch
from hyra_exp3.models import SharedForecastBackbone, graph_arrays, wrong_link_permutation, METHODS

def graph():
    mask = np.eye(16,dtype=bool)
    mask[:,0] = True
    return dict(E_mask=mask,E_lag_steps=np.zeros((16,16),int),E_score=mask.astype(float),
                P_mask=mask,P_lag_steps=np.zeros((16,16),int),P_score=mask.astype(float))

def test_wrong_links_preserve_degrees_and_parameter_capacity_is_identical():
    p = wrong_link_permutation()
    assert len(set(p)) == 16 and np.all(p != np.arange(16))
    original,_,_ = graph_arrays('P',graph())
    wrong,_,_ = graph_arrays('P_wrong_links',graph())
    assert np.array_equal(original.sum(1),wrong.sum(1))
    assert sorted(original.sum(0)) == sorted(wrong.sum(0))
    assert len({SharedForecastBackbone().parameter_count() for _ in METHODS}) == 1

@pytest.mark.parametrize('method',METHODS)
def test_variable_grid_shared_model_has_finite_predictions_and_trainable_gradients(method):
    torch.manual_seed(42)
    model=SharedForecastBackbone()
    history=torch.zeros(64,13,46)
    history[:,:,0]=.3
    history[:,:,1]=.2
    history[:,:,2]=torch.arange(12,-1,-1)/12
    history[:,:,3]=1
    static=torch.rand(64,4)
    ids=torch.arange(64)%16
    prediction=model(history,static,ids,torch.arange(8),torch.full((8,13),.2),torch.ones(8,13,dtype=torch.bool),graph(),method)
    assert prediction.shape == (8,13) and torch.isfinite(prediction).all()
    loss=(prediction-.4).square().mean()
    loss.backward()
    assert model.history.weight_ih_l0.grad is not None and torch.isfinite(model.history.weight_ih_l0.grad).all()

def test_uniform_time_ablation_equals_P_when_aligned_age_is_zero():
    torch.manual_seed(1)
    model=SharedForecastBackbone().eval()
    h=torch.zeros(32,13,46)
    h[:,-1,0]=.3; h[:,-1,1]=.2; h[:,-1,3]=1
    args=(h,torch.rand(32,4),torch.arange(32)%16,torch.arange(5),torch.full((5,13),.2),torch.ones(5,13,dtype=torch.bool),graph())
    with torch.no_grad():
        assert torch.allclose(model(*args,'P'),model(*args,'P_uniform_time'))
