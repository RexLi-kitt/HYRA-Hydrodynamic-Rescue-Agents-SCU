import argparse,hashlib,json,os,shutil,sys,time
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parent.parent;HERE=ROOT/'基线模型';V2=ROOT/'output/dqn-rescue-v1-20261107/frontier_v2';sys.path.insert(0,str(V2));import policy_core as original
s=original.geometry.search
ISO=Path(r'C:\Users\李\.codex\visualizations\2026\11\07\01a1b4c6-8a0c-70e0-8b5c-129c6faa9a31')/'hyra_baseline_relocation_20261108'
configs={'MAPPO':(ROOT/'output/mappo-rescue-v1-20261107/continuation_32768/frozen','', [77126,77127,77128]),'DQN':(V2/'retained_demo_v4','hard_',[88126,88127,88128]),'VDN':(V2/'vdn_v1_20261108','hard_',[88126,88127,88128]),'QMIX':(V2/'qmix_v1_20261108','plain_',[88126,88127,88128]),'QPLEX':(V2/'qplex_v1_20261108','hard_',[88126,88127,88128])}
def sha(p):
 with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=list(configs));ap.add_argument('--device',default='cuda:0');a=ap.parse_args();torch.set_num_threads(1)
 ISO.mkdir(parents=True,exist_ok=True);ds=s.Dataset();records=[]
 for name,(root,prefix,seeds) in configs.items():
  if a.model and name!=a.model:continue
  target=ISO/('relocated_and_renamed_'+name.lower())
  if target.exists():raise FileExistsError(target)
  shutil.copytree(HERE/name,target)
  for seed in seeds:
   path=(root/f'seed_{seed}'/'eval_032768.json') if name=='MAPPO' else root/f'eval_{prefix}{seed}.json'
   ref=read(path);checkpoint=target/'weights'/f'{seed}.pt';manifest=read(HERE/name/'manifest.json');assert sha(checkpoint)==manifest['seeds'][str(seed)]['export_sha256']
   for row in ref['full']:
    if row['parent']!=41004:continue
    assert len(row['trace'])==480
    os.environ.update(HYRA_MODEL_SEED=str(seed),HYRA_MODE=row['mode'],HYRA_DEVICE=a.device,HYRA_SEED=str(row['evaluation_seed']))
    os.environ.pop('HYRA_CHECKPOINT',None)
    factory=s.sdk.agent_factory(str(target/'policy.py')+':make_policy');policy=factory()
    b=ds.bridge(41004)
    for i,r in enumerate(row['trace']):
     obs=b.observation();moves,targets=policy.act(obs)
     if not np.array_equal(moves,r['moves']) or not np.array_equal(targets,r['targets']):raise AssertionError(f'{name} {seed} {row["mode"]} step {i}')
     if not np.array_equal(obs.pos,r['position']):raise AssertionError('observation mismatch')
     b.step(moves,targets,local_action_order=False)
    assert b.done and b.env.metrics()==row['metrics']
    key='hyra_pack_'+name.lower()+'_'
    modules=[Path(m.__file__).resolve() for k,m in sys.modules.items() if k.startswith(key) and getattr(m,'__file__',None)]
    assert modules and all(x.is_relative_to(target) for x in modules)
    rec=dict(model=name,seed=seed,mode=row['mode'],steps=480,evaluation_seed=row['evaluation_seed'],device=a.device,checkpoint_sha256=sha(checkpoint),source_result_sha256=sha(path),actions_exact=True,metrics_exact=True,package_paths_isolated=True)
    records.append(rec);print(json.dumps(rec),flush=True)
  assert sum(x['model']==name for x in records)==6
 out=HERE/('validation_'+a.model.lower()+'.json' if a.model else 'validation_portability.json')
 out.write_text(json.dumps(dict(status='PASS',sdk_sha256=sha(s.SDK/'run.py'),validation_script_sha256=sha(Path(__file__)),records=records),indent=2)+'\n',encoding='utf8')
if __name__=='__main__':main()
