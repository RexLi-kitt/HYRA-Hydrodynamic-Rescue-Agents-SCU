import numpy as np
import torch
from hyra_validation.metric_nn import PhysicalSourceTargetNetwork


def test_hybrid_has_no_cross_lead_physical_feature_access():
    torch.set_num_threads(1);torch.manual_seed(23)
    common=torch.randn(2,12,67);history=torch.randn(2,6,12,6)
    physical=torch.randn(2,12,26)*.05;global_component=torch.randn(2,12,26)*.01;prob=torch.rand(2)
    for kind in ('mlp','gru'):
        model=PhysicalSourceTargetNetwork(67,kind);model.eval()
        with torch.no_grad():
            original=model(common,history,physical,global_component,prob)
            changed=physical.clone();changed[:,:,11]+=.2
            other=model(common,history,changed,global_component,prob)
        keep=np.arange(26)!=11
        np.testing.assert_array_equal(original[:,:,keep].numpy(),other[:,:,keep].numpy())
        assert original.shape==(2,12,26) and torch.isfinite(original).all()
