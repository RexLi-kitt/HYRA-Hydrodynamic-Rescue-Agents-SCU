"""Load the final HYRA MAT inference checkpoint through the official SDK."""

import hashlib
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights" / "mat-contact1024-99131.pt"
PACKAGE = "hyra_mat_" + hashlib.sha256(str(HERE).encode("utf-8")).hexdigest()[:16]

if PACKAGE not in sys.modules:
    src = HERE / "src"
    spec = spec_from_file_location(PACKAGE, src / "__init__.py", submodule_search_locations=[str(src)])
    module = module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)


class Policy:
    def __init__(self, checkpoint=WEIGHTS, mode="sampled", device="cpu", seed=581000):
        if mode not in ("sampled", "greedy"):
            raise ValueError("mode must be sampled or greedy")
        torch.set_num_threads(1)
        ck = torch.load(checkpoint, map_location=device, weights_only=True)
        if ck.get("schema") != "HYRA_REACQUISITION_MAT_V1" or ck.get("config", {}).get("memory") is not True:
            raise ValueError("Expected the memory-enabled HYRA MAT inference checkpoint")

        model_type = import_module(f"{PACKAGE}.reacquisition_model").ReacquisitionMAT
        self.features = import_module(f"{PACKAGE}.features")
        self.device, self.mode = device, mode
        self.model = model_type(device, memory=True)
        self.model.load_state_dict(ck["model"], strict=True)
        self.model.eval()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.reset()

    def reset(self):
        self.hidden = self.model.initial(1)
        self.previous = np.full(4, 585, dtype=np.int64)

    def record_executed(self, moves, targets):
        moves, targets = np.asarray(moves, dtype=np.int64), np.asarray(targets, dtype=np.int64)
        if moves.shape != (4,) or targets.shape != (4,) or np.any(
            (moves < 0) | (moves > 8) | (targets < -1) | (targets > 63)
        ):
            raise ValueError("Invalid executed action")
        self.previous = moves * 65 + targets + 1

    def act(self, observation):
        data = self.features.features(observation)
        tensors = self.features.tensors({k: v[None] for k, v in data.items()}, self.device)
        previous = torch.as_tensor(self.previous[None], device=self.device)
        with torch.no_grad():
            action, _, _, self.hidden = self.model.act(
                **tensors, previous=previous, hidden=self.hidden,
                deterministic=self.mode == "greedy",
            )
        if not tensors["mask"][0, torch.arange(4, device=action.device), action[0]].all():
            raise ValueError("Model selected a masked action")
        packed = action.cpu().numpy().reshape(4)
        moves, targets = packed // 65, packed % 65 - 1
        self.record_executed(moves, targets)
        return moves, targets


def make_policy():
    return Policy(
        Path(os.environ.get("HYRA_CHECKPOINT", str(WEIGHTS))),
        os.environ.get("HYRA_MODE", "sampled"),
        os.environ.get("HYRA_DEVICE", "cpu"),
        int(os.environ.get("HYRA_SEED", "581000")),
    )
