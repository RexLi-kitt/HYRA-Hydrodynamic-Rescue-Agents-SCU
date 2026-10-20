from copy import deepcopy
import pytest
from hyra_exp3.training_audit import verify_model_budget


def test_training_audit_rejects_shortened_budget_and_outcome_selected_checkpoint():
    recipe=dict(steps=1500,checkpoint_steps=[500,1000,1500])
    checkpoints=[dict(step=s,validation_mse_m2=m) for s,m in ((500,.2),(1000,.1),(1500,.15))]
    row=dict(trained=True,test_accessed=False,distinct_training_families=24,validation_families=12,
        training_history=[dict(step=s,training_mse_m2=.1) for s in range(100,1501,100)],
        checkpoints=checkpoints,selected=checkpoints[1])
    verify_model_budget(row,recipe)
    incomplete=deepcopy(row);incomplete['training_history'].pop()
    with pytest.raises(AssertionError,match='optimization budget'):
        verify_model_budget(incomplete,recipe)
    wrong=deepcopy(row);wrong['selected']=wrong['checkpoints'][2]
    with pytest.raises(AssertionError,match='validation-only selection'):
        verify_model_budget(wrong,recipe)
