"""Additional fixed-action diagnostics, never used to select joint checkpoints."""
import json,sys,gc
from pathlib import Path
import numpy as np
import torch
from joint_train import ROOT,ROLES,LAYOUTS,restore,rollout,seed_all,aggregate,write,adapted
run=Path(__file__).parent/'run01'
assert json.loads((run/'queue-status.json').read_text(encoding='utf8'))['status']=='completed'
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
cases=[]
for c in json.loads((ROOT/'uav6-pilot-20261101/data/dataset.json').read_text(encoding='utf8'))['cases']:
 if c['split']=='validation':cases.append((Path(c['path']).name,np.load(c['path'],mmap_mode='r',allow_pickle=False)))
for group in ['hasac_mat','hasac_hatrpo']:
 best=json.loads((run/group/'best.json').read_text(encoding='utf8'))['joint_episode']
 for episode in sorted({0,best,256}):
  if episode==0:
   models=[adapted('hasac')[0],adapted(group.removeprefix('hasac_'))[0]]
  else:
   ck=torch.load(run/group/f'checkpoint-{episode:04d}.pt',map_location='cpu',weights_only=False);models,_=restore(ck);del ck;gc.collect()
  rows=[]
  for assignment,roles in enumerate(ROLES):
   for case,water in cases:
    for layout in LAYOUTS:
     seed_all(layout);metric,_=rollout(models,water,layout,roles,deterministic=True)
     rows.append(dict(assignment=assignment,case=case,layout=layout,**metric))
  result=dict(episodes=rows,summary=aggregate(rows),n=len(rows),whole_task_completed=sum(r['task_complete'] for r in rows),scope='post-training fixed-action diagnostic of baseline, selected and latest checkpoints; not used for model selection')
  write(run/group/f'validation-{episode:04d}-greedy.json',result);print(group,episode,json.dumps(result['summary']),flush=True)
  del models;gc.collect()
