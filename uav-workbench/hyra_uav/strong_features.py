"""Shared causal histories with alternative explicit evidence organisations."""
import numpy as np
from types import SimpleNamespace
from .core import Memory
from .diagnostic import node_features

GROUPS=("B1","B2","B3","A1","A2")


def latest_only(memory):
    """A strict change-evidence ablation: no earlier observed labels survive."""
    return SimpleNamespace(labels=memory.labels,
                           history=[[history[-1]] if history else [] for history in memory.history])


def representation(memory,now,graph,bed,cells):
    nodes=node_features(memory,now)
    pool=graph["pool"][cells]
    static=np.stack((graph["distance"][cells]/5,bed.ravel()[pool]-bed.ravel()[cells,None],graph["terrain_factor"][cells]),axis=-1)
    common=np.concatenate((nodes[cells],nodes[pool].reshape(len(cells),pool.shape[1]*19),static.reshape(len(cells),pool.shape[1]*3)),axis=1).astype(np.float32)
    tail={}
    for group in GROUPS:
        weights=np.ones(pool.shape) if group=="B1" else graph[{"B2":"B2_weight","A2":"A2_weight"}.get(group,"B3_weight")][cells]
        weights=weights/np.maximum(weights.sum(1,keepdims=True),1e-20)
        aggregate=np.einsum("nk,nkf->nf",weights,nodes[pool]).astype(np.float32)
        if group=="A1":aggregate[:,15:19]=0
        interaction=aggregate[:,[15,18]]*nodes[cells,13:14]
        tail[group]=np.concatenate((aggregate,interaction),axis=1)
    return common,tail


def batches(replay,graph,bed,cap=None,seed=20261025,strict_change_ablation=False):
    memory=Memory(tuple(replay["grid"]["shape"]),4);cursor=0;rng=np.random.default_rng(seed)
    for snapshot,truth in zip(replay["snapshots"],replay["truth"]):
        now=snapshot["time_s"];refreshed=np.zeros(memory.labels.size,bool)
        while cursor<len(replay["observations"]) and replay["observations"][cursor]["arrived_s"]<=now:
            obs=replay["observations"][cursor];memory.accept(obs,now)
            if obs["arrived_s"]==now:refreshed[obs["cells"]]=True
            cursor+=1
        if not 0<now<replay["config"]["duration_s"] or now%10:continue
        cells=np.flatnonzero((memory.labels.ravel()>=0)&~refreshed)
        if cap is not None and len(cells)>cap:cells=np.sort(rng.choice(cells,cap,replace=False))
        common,tail=representation(memory,now,graph,bed,cells)
        common_a1=None
        if strict_change_ablation:
            common_a1,latest_tail=representation(latest_only(memory),now,graph,bed,cells)
            tail["A1"]=latest_tail["A1"]
        # Hidden state is read only here to create the common target labels.
        y=(memory.labels.ravel()[cells]!=np.asarray(truth)[cells]).astype(np.uint8)
        yield dict(time_s=now,cells=cells,age_s=now-memory.observed_s.ravel()[cells],common=common,tail=tail,y=y,
                   common_A1=common_a1)


def cache(replay,graph,bed,path,cap=None,seed=20261025,strict_change_ablation=False):
    items=list(batches(replay,graph,bed,cap,seed,strict_change_ablation))
    data={"common":np.concatenate([i["common"] for i in items]),
          "y":np.concatenate([i["y"] for i in items]),"cells":np.concatenate([i["cells"] for i in items]),
          "time_s":np.concatenate([np.full(len(i["cells"]),i["time_s"]) for i in items]),
          "age_s":np.concatenate([i["age_s"] for i in items]),
          "decision_times_s":np.asarray([i["time_s"] for i in items])}
    for group in GROUPS:data[group]=np.concatenate([i["tail"][group] for i in items])
    if strict_change_ablation:data["common_A1"]=np.concatenate([i["common_A1"] for i in items])
    np.savez_compressed(path,**data)
    return dict(rows=len(data["y"]),positives=int(data["y"].sum()),common_dimension=data["common"].shape[1],tail_dimension=21,
                cap=cap,sampling_seed=seed,strict_change_ablation=strict_change_ablation)


def model_inputs(data,group):
    raw=data["common_A1"] if group=="A1" and "common_A1" in data else data["common"]
    return np.concatenate((raw,data[group]),axis=1)
