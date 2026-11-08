"""Official SDK entry point. No training libraries are required."""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/"engine"))
os.environ["HYRA_STUDY_RADIUS"]="1"
from delivery import DeliveryMission
from judge import DELIVERY_RADIUS_M
from scenario_io import load_trial, read, regenerate, write

VERSION="20261106.1"
OBS_FIELDS=("t","pos","heading","last","measured","seen_xy","seen_v","seen_at","visible","active","owner","new_cells","observed_cells")


def file_hash(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def view(env):
    raw=env.observation()
    fields={}
    for k in OBS_FIELDS:
        v=getattr(raw,k)
        if isinstance(v,np.ndarray):
            v=v.copy()
            v.flags.writeable=False
        fields[k]=v
    return SimpleNamespace(**fields)


def obs_hash(o):
    h=hashlib.sha256()
    for k in OBS_FIELDS:
        a=np.ascontiguousarray(np.asarray(getattr(o,k)))
        h.update(k.encode());h.update(a.dtype.str.encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def action_pair(result):
    actions, targets=map(np.asarray,result)
    for x,lo,hi in ((actions,0,8),(targets,-1,63)):
        if x.shape!=(4,) or x.dtype.kind not in "iu" or np.any(x<lo) or np.any(x>hi):
            raise ValueError("Expected four integer actions 0..8 and four integer person IDs -1..63")
    return actions,targets


def agent_factory(spec):
    filename,member=spec.rsplit(":",1)
    path=Path(filename).resolve()
    loader=importlib.util.spec_from_file_location("user_policy",path)
    if loader is None or loader.loader is None:raise ValueError(str(path))
    sys.path.insert(0,str(path.parent))
    mod=importlib.util.module_from_spec(loader);sys.modules[loader.name]=mod
    loader.loader.exec_module(mod)
    return getattr(mod,member)


def evaluate(folder,trial,factory=None,tape=None,golden=None,collect=False):
    water,pop=load_trial(folder,trial)
    assert DELIVERY_RADIUS_M==1.
    env=DeliveryMission(water,pop,(64,720,2400))
    agent=factory() if factory is not None else None
    history=[]
    while env.t<2400:
        t=env.t;step=t//5
        o=view(env)
        fingerprint=obs_hash(o) if golden is not None else None
        if golden is not None and fingerprint!=golden[step]:
            raise AssertionError(f"{trial['id']} observation differs before step {step}")
        actions,targets=action_pair(agent.act(o) if tape is None else (tape[step,:4],tape[step,4:]))
        env.step(actions,targets)
        if collect:history.append([*actions.tolist(),*targets.tolist()])
        if (env.supplied>=0).all() and (env.last>=0).all():break
    if tape is not None and len(tape)!=env.t//5:raise ValueError("Tape length differs from the mission")
    metrics=env.metrics()
    if golden is not None and obs_hash(view(env))!=golden[env.t//5]:
        raise AssertionError(f"{trial['id']} final observation differs")
    return metrics,history


def aggregate(rows):
    metrics=[r["metrics"] for r in rows];n=len(metrics)
    total=sum(m["total"] for m in metrics);delivered=sum(m["delivered"] for m in metrics)
    out=dict(trials=n,delivered=delivered,total=total,RCR=delivered/total,all_rate=sum(m["all"] for m in metrics)/n)
    for k in ("ST","ACC","coverage","FRT","CRT"):
        v=[m[k] for m in metrics if m[k] is not None]
        out[k]=sum(v)/len(v) if v else None
        out[k+"_valid"]=len(v)
    timed=[m["coverage"] for m in metrics if m["timeout"]]
    out.update(timeout_rate=len(timed)/n,timeout_coverage=sum(timed)/len(timed) if timed else None)
    return out


def manifest_checks(folder):
    manifest=read(folder/"SHA256.json")
    for name,digest in manifest.items():
        if file_hash(folder/name)!=digest:raise ValueError(f"SHA-256 mismatch: {name}")
    return len(manifest)


def verify():
    from contract_checks import check
    count=manifest_checks(HERE)
    folder=HERE/"examples"
    manifest_checks(folder)
    trials=read(folder/"panel.json")
    for case in trials:
        prefix=HERE/"golden"/case["id"]
        with np.load(prefix.with_suffix(".npz"),allow_pickle=False) as z:
            metrics,_=evaluate(folder,case,tape=z["tape"],golden=z["observations"])
        expected=read(prefix.with_suffix(".json"))
        if metrics!=expected:raise AssertionError(f"Metrics differ: {case['id']}")
        print(case["id"],metrics["ST"],"steps",metrics["delivered"],"/",metrics["total"])
    check()
    print("PASS: file digests, original action tapes, every observation, final per-person times, metrics and edge cases.",count)


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="cmd",required=True)
    sub.add_parser("verify")
    replay=sub.add_parser("replay")
    replay.add_argument("--panel",type=Path,default=HERE/"examples")
    replay.add_argument("--result",type=Path,required=True)
    g=sub.add_parser("regenerate")
    g.add_argument("--panel",type=Path,default=HERE/"examples")
    a=sub.add_parser("eval"); a.add_argument("--panel",type=Path,default=HERE/"examples")
    a.add_argument("--agent",default=str(HERE/"example_policy.py")+":make_policy")
    a.add_argument("--out",type=Path,required=True);a.add_argument("--save-actions",action="store_true")
    args=parser.parse_args()
    if args.cmd=="verify":verify();return
    folder=args.panel.resolve()
    manifest_checks(folder)
    trials=read(folder/"panel.json")
    if args.cmd=="regenerate":
        for case in trials:regenerate(folder,case);print(case["id"],"population identical")
        return
    if args.cmd=="replay":
        recorded=read(args.result)
        if (recorded["version"]!=VERSION or recorded["panel_sha"]!=file_hash(folder/"panel.json")
                or recorded["panel_files_sha"]!=file_hash(folder/"SHA256.json")):
            raise ValueError("Recorded release or panel differs")
        if [r["id"] for r in recorded["trials"]]!=[r["id"] for r in trials]:
            raise ValueError("Recorded trial list differs")
        for case,row in zip(trials,recorded["trials"]):
            m,_=evaluate(folder,case,tape=np.asarray(row["actions_and_targets"]))
            if m!=row["metrics"]:raise AssertionError(case["id"]+" replay differs")
        print("PASS: identical per-person delivery times and metrics for",len(trials),"action tapes")
        return
    if args.out.exists():raise FileExistsError(f"Choose a new output path: {args.out}")
    factory=agent_factory(args.agent);rows=[]
    for case in trials:
        m,h=evaluate(folder,case,factory=factory,collect=args.save_actions)
        row=dict(id=case["id"],map=case["map"],parent=case["parent"],family=case["family"],roster=case["roster"],metrics=m)
        if args.save_actions:row["actions_and_targets"]=h
        rows.append(row)
    out=dict(version=VERSION,panel_sha=file_hash(folder/"panel.json"),panel_files_sha=file_hash(folder/"SHA256.json"),
             policy_sha=file_hash(Path(args.agent.rsplit(":",1)[0]).resolve()),summary=aggregate(rows),trials=rows)
    write(args.out,out)
    print(json.dumps(out["summary"],indent=2))


if __name__=="__main__":
    main()
