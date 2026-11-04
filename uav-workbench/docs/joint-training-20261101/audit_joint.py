"""Post-run integrity audit of the frozen joint-training experiment."""
import gc, hashlib, json, math
from collections import Counter
from pathlib import Path
import numpy as np
import torch

BASE=Path(__file__).parent
RUN=BASE/'run01'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def finite(value):
    if torch.is_tensor(value):return bool(torch.isfinite(value).all())
    if isinstance(value,np.ndarray):return bool(np.isfinite(value).all())
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
    if isinstance(value,(float,np.floating)):return math.isfinite(value)
    return True

torch.set_num_threads(2)
protocol=read(RUN/'protocol.json')
assert read(RUN/'queue-status.json')['status']=='completed'
assert sha(BASE/'joint_train.py')==protocol['source_sha256']
for parent in protocol['parents'].values():assert sha(Path(parent['path']))==parent['sha256']
assert read(RUN/'preflight.json')['status']=='passed'
results={};schedules=[];selections={}
for group,partner in [('hasac_mat','mat'),('hasac_hatrpo','hatrpo')]:
    folder=RUN/group
    rows=[json.loads(line) for line in (folder/'episodes.jsonl').read_text(encoding='utf8').splitlines()]
    assert [r['joint_episode'] for r in rows]==list(range(1,257))
    assert finite(rows)
    assert sum(r['loss'] is not None for r in rows)==128
    assert all((r['loss'] is not None)==(r['joint_episode']%2==0) for r in rows)
    allocations=Counter((r['case'],tuple(r['roles'])) for r in rows)
    assert len(allocations)==8 and set(allocations.values())=={32}
    schedules.append([(r['case'],r['layout'],r['roles']) for r in rows])
    validations=[]
    for ep in (0,64,128,192,256):
        validation=read(folder/f'validation-{ep:04d}.json')
        assert validation['n']==32 and len(validation['episodes'])==32 and finite(validation)
        keys={(r['assignment'],r['action_seed'],r['case'],r['layout']) for r in validation['episodes']}
        assert len(keys)==32
        validations.append(dict(joint_episode=ep,coverage=validation['summary']['coverage'],rescued_ratio=validation['summary']['RCR'],potential_conflicts_per_step=validation['summary']['ACC'],all_rescued=validation['summary']['CRT']['valid'],whole_task_complete=validation['whole_task_completed']))
    ck=torch.load(folder/'checkpoint-0256.pt',map_location='cpu',weights_only=False)
    assert ck['joint_episode']==256 and finite(ck)
    assert [m['algorithm'] for m in ck['models']]==['hasac',partner]
    assert [m['updates'] for m in ck['models']]==[512,128]
    assert len(ck['replay'])==4096
    latest=read(folder/'latest.json')
    assert latest['sha256']==sha(folder/'checkpoint-0256.pt')
    results[group]=dict(training_episodes=256,training_joint_steps=sum(r['ST'] for r in rows),training_seed=1001,case_role_counts=[dict(case=k[0],roles=list(k[1]),episodes=v) for k,v in sorted(allocations.items())],learner_update_calls=[512,128],checkpoint_finite=True,checkpoint_sha256=latest['sha256'],best_joint_episode=read(folder/'best.json')['joint_episode'],validation=validations)
    best=results[group]['best_joint_episode']
    chosen=next(v for v in validations if v['joint_episode']==best)
    selections[group]=dict(chosen_joint_episode=best,additional_joint_training_run=256,parent_training_episodes={'hasac':1024,partner:256 if partner=='mat' else 1024},checkpoint_relative_path=f'{group}/checkpoint-{best:04d}.pt' if best else None,checkpoint_sha256=sha(folder/f'checkpoint-{best:04d}.pt') if best else None,baseline=validations[0],chosen=chosen,latest=validations[-1],interpretation='development-validation selection only; baseline 0 reconstructed from unchanged parent weights and zero-initialized role adapter; no independent held-out result')
    del ck;gc.collect()
assert schedules[0]==schedules[1]
audit=dict(status='passed',parents_unchanged=True,training_source_unchanged=True,same_training_schedules=True,new_joint_episodes_total=512,scope='Integrity and finite-value audit only; not evidence of performance or real-world rescue validity.',groups=results)
(RUN/'integrity-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
(RUN/'selection-summary.json').write_text(json.dumps(selections,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(status=audit['status'],new_joint_episodes_total=512,best={k:v['chosen_joint_episode'] for k,v in selections.items()})))
