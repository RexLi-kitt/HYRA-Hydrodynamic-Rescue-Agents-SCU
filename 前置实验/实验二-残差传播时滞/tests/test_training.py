import torch
import pytest
from hyra_validation.training import RunLogger


def test_logger_accepts_scalar_tensor_and_rejects_nonfinite():
    logger=RunLogger()
    logger.log_stat("loss",torch.tensor(2.),1)
    assert logger.stats[0]["value"]==2.
    with pytest.raises(FloatingPointError):
        logger.log_stat("loss",float("nan"),1)
