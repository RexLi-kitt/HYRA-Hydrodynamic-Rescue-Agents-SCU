"""Environment checks only: imports, CUDA kernels and tiny solver smoke test.

Does not load research events or launch training/replay experiments.
Run with system Python, not a project virtual environment.
"""
import contextlib
import importlib
import importlib.metadata as metadata
import io
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = dict(checked_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        executable=sys.executable, python=sys.version, platform=platform.platform(),
        prefix=sys.prefix, base_prefix=sys.base_prefix, checks={})
    if sys.prefix != sys.base_prefix:
        raise RuntimeError("Use the system Python; virtual environments are not requested")
    destination = ROOT/"results/setup"
    destination.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("ANUGADATA",str(destination))
    def check(name, function):
        try:
            report["checks"][name] = dict(passed=True, details=function())
        except Exception as error:
            report["checks"][name] = dict(passed=False,error=f"{type(error).__name__}: {error}")
    for package in ["numpy","scipy","matplotlib","pytest","yaml","sacred","dill","netCDF4","meshpy","pyproj","xarray","tensorboard"]:
        def imported(p=package):
            module = importlib.import_module(p)
            return dict(version=str(getattr(module,"__version__","not exposed")),file=getattr(module,"__file__",None))
        check("import_"+package,imported)
    def cuda():
        import torch
        assert torch.cuda.is_available(), "CUDA unavailable"
        assert "sm_120" in torch.cuda.get_arch_list(), "No native RTX5060 architecture"
        torch.manual_seed(1)
        a = torch.randn(64,64,device="cuda",requires_grad=True)
        loss = (a@a.T).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(loss) and torch.isfinite(a.grad).all()
        # CPU kernel separately verifies base DLLs.
        assert (torch.tensor([1,2])*2).tolist()==[2,4]
        return dict(version=torch.__version__,cuda_runtime=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
                    architectures=torch.cuda.get_arch_list(),loss=float(loss.detach().cpu()))
    check("torch_cpu_cuda_forward_backward",cuda)
    def metis():
        import pymetis
        cut,parts = pymetis.part_graph(2,adjacency=[[1],[0,2],[1,3],[2]])
        assert len(parts)==4 and len(set(parts))==2
        return dict(version=metadata.version("pymetis"),edge_cuts=int(cut),parts=list(parts))
    check("pymetis_partition",metis)
    def solver():
        import anuga
        import numpy as np
        domain = anuga.rectangular_cross_domain(2,2,len1=2.,len2=2.)
        domain.set_store(False)
        domain.set_quantity("elevation",0.)
        domain.set_quantity("stage",.2)
        wall = anuga.Reflective_boundary(domain)
        domain.set_boundary(dict(left=wall,right=wall,top=wall,bottom=wall))
        times = list(domain.evolve(yieldstep=.1,finaltime=.2))
        values = domain.quantities["stage"].centroid_values
        assert np.isfinite(values).all() and np.allclose(values,.2,atol=1e-8)
        return dict(version=anuga.__version__,times=times,constant_water_depth_preserved=True)
    check("anuga_tiny_solver",solver)
    def pymarl():
        sys.path.insert(0,str(ROOT/"third_party/pymarl/src"))
        from components.episode_buffer import EpisodeBatch,ReplayBuffer
        from modules.mixers.qmix import QMixer
        from learners.q_learner import QLearner
        return dict(imports=[EpisodeBatch.__name__,ReplayBuffer.__name__,QMixer.__name__,QLearner.__name__],training_started=False)
    check("pymarl_core_imports",pymarl)
    result = subprocess.run([sys.executable,"-m","pip","check"],capture_output=True,text=True)
    report["checks"]["pip_check"] = dict(passed=result.returncode==0,details=result.stdout+result.stderr)
    versions = {d.metadata["Name"]:d.version for d in metadata.distributions() if d.metadata["Name"]}
    (destination/"system_packages_after.json").write_text(json.dumps(versions,indent=2,ensure_ascii=False),encoding="utf-8")
    report["all_passed"] = all(c["passed"] for c in report["checks"].values())
    (destination/"system_environment_check.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=True))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
