"""Training-only scripted demonstration of DQN Q rankings.

Search and rescue labels use public observations and the official transition.
The output is extra supervised training, not a pure DQN result.
"""
import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
import policy_core as p
import replay
import teachers

s=p.geometry.search
LOCAL=('policy_core.py','replay.py','teachers.py','demo.py')


def runtime():
    return dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,
                cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())


def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def source_lock():
    paths={str(p.SOURCE):p.sha(p.SOURCE),str(p.PYMARL/'modules'/'agents'/'rnn_agent.py'):p.sha(p.PYMARL/'modules'/'agents'/'rnn_agent.py')}
    for name in LOCAL:paths[str(p.HERE/name)]=p.sha(p.HERE/name)
    for path in (teachers.MAPPO_SEARCH/'core.py',teachers.MAPPO_SEARCH/'carpet.py',
                 teachers.MAPPO_RESCUE/'core.py',teachers.MAPPO_RESCUE/'micro.py',
                 teachers.MAPPO_RESCUE/'teacher_multistep.py'):
        paths[str(path)]=p.sha(path)
    paths.update(s.source_lock())
    return paths


def write_json(path,data):s.json_write(path,data)


def data_hash(examples):
    h=hashlib.sha256()
    for row in examples:
        state=row[0]
        for key in ('base_q','hidden','patch_bits','subcell','public','mask_bits','indices','people','actions'):
            z=np.ascontiguousarray(state[key]);h.update(key.encode());h.update(str(z.shape).encode());h.update(z.tobytes())
        for item in row[1:]:h.update(np.ascontiguousarray(np.asarray(item)).tobytes())
    return h.hexdigest()


def save(path,data):
    if path.exists():raise FileExistsError(path)
    with path.open('xb') as f:torch.save(data,f);f.flush();os.fsync(f.fileno())


def load_model(path,device):
    ck=torch.load(path,map_location=device,weights_only=False)
    if ck['schema']!='DQN_RESCUE_BC_V1':raise ValueError('Not a DQN demonstration checkpoint')
    s.check_lock(ck['source_lock'])
    model=p.RecurrentQ().to(device);model.load_state_dict(ck['model'])
    return model,ck


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('search','rescue'),required=True)
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--updates',type=int,required=True)
    ap.add_argument('--batch',type=int,default=32);ap.add_argument('--steps-per-map',type=int,default=480)
    ap.add_argument('--cases',type=int,default=512);ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--search-checkpoint',type=Path);ap.add_argument('--resume',type=Path)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out.resolve()
    if min(a.updates,a.batch,a.cases,a.steps_per_map)<=0 or a.steps_per_map>480:
        ap.error('positive counts and 1..480 map horizon required')
    if not out.is_relative_to(p.HERE) or out.exists():ap.error('out must be a new experiment directory')
    if a.phase=='rescue' and not a.search_checkpoint:ap.error('rescue phase requires a search checkpoint')
    torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    lock=source_lock()
    if a.search_checkpoint:lock[str(a.search_checkpoint.resolve())]=p.sha(a.search_checkpoint)
    resume_sha=p.sha(a.resume) if a.resume else None
    out.mkdir(parents=True);snapshot=out/'source_snapshot';snapshot.mkdir()
    for i,(path,digest) in enumerate(lock.items()):
        if Path(path).suffix=='.pt':continue
        dest=snapshot/f'{i:03d}_{Path(path).name}';shutil.copyfile(path,dest)
        if p.sha(dest)!=digest:raise ValueError('Source snapshot mismatch')
    write_json(out/'source_manifest.json',dict(inputs=lock,runtime=runtime(),
        parent_search_sha256=p.sha(a.search_checkpoint) if a.search_checkpoint else None))
    if a.phase=='search':model,parent=p.initialize(a.seed,a.device);del parent
    else:model,parent=load_model(a.search_checkpoint,a.device)
    if a.phase=='rescue' and (parent['phase']!='search' or parent['seed']!=a.seed):raise ValueError('Search checkpoint mismatch')
    reference=copy.deepcopy(model)
    for par in model.base.parameters():par.requires_grad_(False)
    for par in model.search.parameters():par.requires_grad_(a.phase=='search')
    for par in model.rescue.parameters():par.requires_grad_(a.phase=='rescue')
    trainable=model.search if a.phase=='search' else model.rescue
    optimizer=torch.optim.Adam(trainable.parameters(),lr=.001,eps=1e-5)
    if a.phase=='search':examples,records=teachers.search_examples(model,a.steps_per_map,a.device)
    else:examples,records=teachers.rescue_examples(model,a.cases,a.device)
    if not examples:raise ValueError('No demonstration examples')
    digest=data_hash(examples);previous=0
    if a.resume:
        saved=torch.load(a.resume,map_location=a.device,weights_only=False)
        if saved['schema']!='DQN_RESCUE_BC_V1' or saved['phase']!=a.phase or saved['seed']!=a.seed or saved['dataset_sha256']!=digest:
            raise ValueError('Resume dataset/phase mismatch')
        if saved['source_lock']!=lock:raise ValueError('Resume source changed')
        if saved['search_parent_sha256']!=(p.sha(a.search_checkpoint) if a.search_checkpoint else None):raise ValueError('Search parent changed')
        if saved['runtime']!=runtime() or saved['config']!=dict(batch=a.batch,steps_per_map=a.steps_per_map,cases=a.cases,device=a.device):
            raise ValueError('Resume runtime or sampling configuration changed')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);set_rng(saved['rng']);previous=saved['updates']
    losses=[];labels=np.array([int(row[2]) if a.phase=='rescue' else 1 for row in examples])
    positives=np.flatnonzero(labels>=0);negatives=np.flatnonzero(labels<0)
    for step in range(a.updates):
        if a.phase=='rescue' and len(negatives) and a.batch>1:
            count=max(1,a.batch//2);idx=np.r_[np.random.choice(positives,count),np.random.choice(negatives,a.batch-count)];np.random.shuffle(idx)
        else:idx=np.random.randint(len(examples),size=a.batch)
        samples=[examples[int(i)] for i in idx];q,mask=replay.qvalues(model,[r[0] for r in samples],a.device)
        if a.phase=='search':
            q=q.reshape(-1,9,65).masked_fill(~mask.reshape(-1,9,65),-torch.inf).amax(-1);legal=mask.reshape(-1,9,65).any(-1)
            targets=torch.as_tensor(np.concatenate([r[1] for r in samples]),device=a.device,dtype=torch.long)
            eligible=torch.as_tensor(np.concatenate([r[2] for r in samples]),device=a.device,dtype=torch.bool)
            if not bool(eligible.any()):continue
            loss=torch.nn.functional.cross_entropy(q[eligible].masked_fill(~legal[eligible],-torch.inf),targets[eligible])
        else:
            chosen=torch.as_tensor([4*i+r[1] for i,r in enumerate(samples)],device=a.device)
            q=q[chosen];mask=mask[chosen];truth=torch.as_tensor([r[2] for r in samples],device=a.device,dtype=torch.long)
            with torch.no_grad():ref,_=replay.qvalues(reference,[r[0] for r in samples],a.device);ref=ref[chosen]
            current=torch.log_softmax(q.masked_fill(~mask,-torch.inf),dim=-1)
            baseline=torch.log_softmax(ref.masked_fill(~mask,-torch.inf),dim=-1)
            probs=torch.softmax(ref.masked_fill(~mask,-torch.inf),dim=-1)
            kl=(probs*(torch.where(mask,baseline,0.)-torch.where(mask,current,0.))).sum(-1)
            positive=truth>=0;safe=torch.where(positive,truth,torch.full_like(truth,520))
            supervised=-current.gather(1,safe[:,None]).squeeze(1)
            loss=(torch.where(positive,supervised,torch.zeros_like(supervised))+torch.where(positive,.02*kl,kl)).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite demonstration loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(trainable.parameters(),10.);optimizer.step()
        losses.append(float(loss.detach()))
        if (step+1)%20==0:print(json.dumps(dict(phase=a.phase,update=previous+step+1,loss=losses[-1])),flush=True)
    s.check_lock(lock)
    if a.resume and p.sha(a.resume)!=resume_sha:raise ValueError('Resume checkpoint changed during run')
    ck=dict(schema='DQN_RESCUE_BC_V1',seed=a.seed,phase=a.phase,model=model.state_dict(),optimizer=optimizer.state_dict(),
        updates=previous+a.updates,dataset_sha256=digest,source_lock=lock,rng=rng_state(),
        search_parent_sha256=p.sha(a.search_checkpoint) if a.search_checkpoint else None,
        config=dict(batch=a.batch,steps_per_map=a.steps_per_map,cases=a.cases,device=a.device),
        runtime=runtime(),parent_sha256=p.SOURCE_SHA256)
    save(out/'checkpoint.pt',ck)
    write_json(out/'summary.json',dict(phase=a.phase,seed=a.seed,cases=a.cases if a.phase=='rescue' else None,
        search_collection_joint_steps=sum(r['joint_steps'] for r in records) if a.phase=='search' else 0,
        search_teacher_labels=sum(r['accepted_labels'] for r in records) if a.phase=='search' else None,
        updates=a.updates,optimization_steps=len(losses),batch=a.batch,example_draws=a.updates*a.batch,
        positive_examples=int((labels>=0).sum()),abstention_examples=int((labels<0).sum()),
        dataset_sha256=digest,cases_detail=records,losses=losses,checkpoint_sha256=p.sha(out/'checkpoint.pt'),
        limitations=['training-only scripted teacher, additional supervision, not pure DQN',
                     'old replay and optimizer not resumed; new residual and optimizer are a new branch']))


if __name__=='__main__':main()
