"""Portable continuation layer: verifies original bytes before path relocation.

One algorithm per process avoids conflicting legacy imports. Full checkpoints are
trusted torch pickle files and must come from the published SHA-verified assets.
"""
import argparse,copy,hashlib,importlib.util,json,os,re,shutil,sys,time
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
MANIFEST=HERE/'original_manifest.json'
# Pin the exact original-input inventory. A changed inventory requires a code review.
MANIFEST_SHA256='a2a814dfd4fc874b5b4e80b00471c56abd12810a9fee04c2c5688a0174dc8ede'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load_inventory():
    if sha(MANIFEST)!=MANIFEST_SHA256:raise ValueError('Input inventory changed')
    return json.loads(MANIFEST.read_text(encoding='utf8'))

def verify_inputs(base,inventory):
    for path,digest in inventory['all_files_sha256'].items():
        f=(base/path).resolve()
        if not f.is_relative_to(base.resolve()) or not f.is_file() or sha(f)!=digest:
            raise ValueError('Original input missing or changed: '+path)

def verify_lock(lock,base,inventory):
    for old,digest in lock.items():
        key=hashlib.sha256(old.encode('utf8')).hexdigest();entry=inventory['legacy_path_sha256_mapping'].get(key)
        if not entry or entry['sha256']!=digest:raise ValueError('Unmapped original locked source')
        f=base/entry['path']
        if sha(f)!=digest:raise ValueError('Original locked input changed: '+entry['path'])

def copytree_links(src,dst):
    for f in src.rglob('*'):
        q=dst/f.relative_to(src)
        if f.is_dir():q.mkdir(parents=True,exist_ok=True)
        elif f.is_file():
            q.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(f,q)
            except OSError:shutil.copyfile(f,q)

def patch_files(stage):
    # Only runtime-local path constants change. Their original bytes remain in inputs.
    loc={'ALIGN':stage/'alignment','SDK':stage/'sdk','RULE':stage/'rule/policy.py',
         'UPSTREAM':stage/'upstream/on-policy','PYMARL':stage/'upstream/pymarl/src'}
    files=[stage/'project/output/mappo-search-v1-20261107/core.py',stage/'alignment/mappo_short_seed2_v1/trainer.py',stage/'project/output/dqn-rescue-v1-20261107/frontier_v2/policy_core.py']
    changed={}
    for path in files:
        text=path.read_text(encoding='utf8');before=sha(path)
        for key,value in loc.items():
            expression=r'(?m)^'+key+r'=Path\(r[\'\"]([^\'\"]*)[\'\"]\)'
            text=re.sub(expression,lambda m,k=key:'%s=Path(os.environ[%r])'%(k,'HYRA_RUNTIME_'+k),text)
        if 'import os\n' not in text:text='import os\n'+text
        tmp=path.with_name(path.name+'.portable.tmp');tmp.write_text(text,encoding='utf8',newline='\n');os.replace(tmp,path)
        changed[path.relative_to(stage).as_posix()]={'original_sha256':before,'portable_sha256':sha(path)}
    for key,value in loc.items():os.environ['HYRA_RUNTIME_'+key]=str(value)
    return changed

def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def make_dataset(s,base,inventory):
    # Original water/population paths in panel.json are historical. Use a
    # verified four-case mapping; keep the original referee entirely unchanged.
    cases={x['parent']:x for x in inventory['cases']}
    class Dataset:
        def __init__(self):self.cache=OrderedDict()
        def load(self,parent):
            if parent in self.cache:self.cache.move_to_end(parent);return self.cache[parent]
            if parent not in cases:raise ValueError('Only the four registered training cases are bundled')
            row={'water':str(base/cases[parent]['files']['water']['path']),'population':str(base/cases[parent]['files']['population']['path'])}
            for k in ('water','population'):
                if sha(row[k])!=cases[parent]['files'][k]['sha256']:raise ValueError('Training case changed')
            pair=s.sdk.load_trial(base,row);self.cache[parent]=pair
            while len(self.cache)>4:self.cache.popitem(last=False)
            return pair
        def bridge(self,parent):return s.RefereeBridge(s.sdk.DeliveryMission(*self.load(parent),(64,720,2400)))
    return Dataset()

def make_run(model,seed,stage,base,inventory,source,device,allow_runtime_change=False):
    repo=stage/'project/output';v=repo/'dqn-rescue-v1-20261107/frontier_v2';m=repo/'mappo-rescue-v1-20261107'
    if model=='MAPPO':
        sys.path.insert(0,str(m));c=__import__('core');s=c.search
        s.check_lock=lambda lock:verify_lock(lock,base,inventory)
        mod=__import__('train');ds=make_dataset(s,base,inventory);run=mod.Run(ds,seed,device)
        if source.get('paired_continuation'):
            meta=source['paired_continuation'];verify_lock(meta['source_lock'],base,inventory)
            if meta['arm']!='frozen':raise ValueError('Wrong MAPPO arm')
        if source.get('stable_ppo_v2'):verify_lock(source['stable_ppo_v2']['source_lock'],base,inventory)
        run.restore(source)
        for n,p in run.policy.actor.named_parameters():p.requires_grad_(n.startswith('rescue.'))
        for p in run.policy.critic.parameters():p.requires_grad_(True)
        for g in run.policy.actor_optimizer.param_groups:g['lr']=.0001
        for g in run.policy.critic_optimizer.param_groups:g['lr']=.0005
        return run,s,None
    selected={'DQN':('retained_demo_v4','train_v4.py','hard'),'VDN':('vdn_v1_20261108','train_vdn.py','hard'),'QMIX':('qmix_v1_20261108','train_qmix.py','plain'),'QPLEX':('qplex_v1_20261108','train_qplex.py','hard')}[model]
    folder=v/selected[0];sys.path.insert(0,str(folder));sys.path.insert(0,str(v));p=__import__('policy_core');s=p.geometry.search
    s.check_lock=lambda lock:verify_lock(lock,base,inventory)
    # Override the legacy dataset factory for the four verified training scenes.
    s.Dataset=lambda:make_dataset(s,base,inventory)
    mod=import_file('portable_training_algorithm',folder/selected[1]);bc=v/('pilot_rescue_1000_88126' if seed==88126 else f'formal/rescue_{seed}')/'checkpoint.pt'
    verify_lock(source['source_lock'],base,inventory)
    run=mod.Run(seed,bc,selected[2],device)
    if allow_runtime_change:
        source=dict(source);source['config']=dict(source['config']);source['config']['device']=device
        import demo
        source['runtime']=demo.runtime()
    run.restore(source,source['source_lock'])
    return run,s,source['source_lock']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=('MAPPO','DQN','VDN','QMIX','QPLEX'),required=True);ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--start',choices=('parent','checkpoint'),default='checkpoint');ap.add_argument('--resume',type=Path);ap.add_argument('--resume-sha256');ap.add_argument('--steps',type=int,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--inputs',type=Path,default=HERE/'original');ap.add_argument('--device',default='cuda:0');ap.add_argument('--allow-runtime-change',action='store_true');a=ap.parse_args()
    seeds=(77126,77127,77128) if a.model=='MAPPO' else (88126,88127,88128)
    if a.seed not in seeds or a.steps<=0 or (a.model=='MAPPO' and a.steps%128):ap.error('invalid seed/steps; MAPPO steps must be divisible by 128')
    output=a.output.resolve();base=a.inputs.resolve()
    if output.exists() or output.is_relative_to(base) or base.is_relative_to(output):raise ValueError('Output must be new and separate from inputs')
    torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    inventory=load_inventory();verify_inputs(base,inventory)
    checkpoint=PACKAGE/a.model/'training/checkpoints'/f'{a.seed}.pt'
    if a.resume:
        checkpoint=a.resume.resolve()
        if not a.resume_sha256 or not re.fullmatch(r'[0-9a-fA-F]{64}',a.resume_sha256) or sha(checkpoint)!=a.resume_sha256.lower():raise ValueError('Provide matching --resume-sha256 for the checkpoint')
    elif a.start=='parent':
        checkpoint=base/('project/output/mappo-rescue-v1-20261107/stable_ppo_v2/seed_%s/checkpoint-008192.pt'%a.seed if a.model=='MAPPO' else 'project/output/dqn-rescue-v1-20261107/frontier_v2/'+('pilot_rescue_1000_88126' if a.seed==88126 else f'formal/rescue_{a.seed}')+'/checkpoint.pt')
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    if not a.resume and a.start=='checkpoint':
        recorded=json.loads((PACKAGE/a.model/'training/checkpoints.json').read_text(encoding='utf8'))['seeds'][str(a.seed)]['sha256']
        if sha(checkpoint)!=recorded:raise ValueError('Checkpoint SHA mismatch')
    state=torch.load(checkpoint,map_location=a.device,weights_only=False)
    portable=state.get('portable_continuation')
    if portable:
        if portable['model']!=a.model or portable['seed']!=a.seed or portable['inventory_sha256']!=MANIFEST_SHA256:raise ValueError('Wrong portable resume')
        source=state['state']
    else:source=state
    current_runtime=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version())
    original_runtime=(portable or {}).get('runtime_original',source.get('runtime'))
    original_device=(portable or {}).get('device_original',source.get('config',{}).get('device',a.device))
    if a.allow_runtime_change and source.get('rng',{}).get('cuda') is not None:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=len(source['rng']['cuda']):raise ValueError('CUDA RNG restoration requires the original CUDA device count')
    output.mkdir(parents=True);stage=output/'runtime_files';copytree_links(base,stage);patched=patch_files(stage)
    if a.model!='MAPPO' and a.start=='parent' and not a.resume:
        # A parent starts an entirely new run using the same registered config.
        repo=stage/'project/output/dqn-rescue-v1-20261107/frontier_v2';folder,script,arm={'DQN':('retained_demo_v4','train_v4.py','hard'),'VDN':('vdn_v1_20261108','train_vdn.py','hard'),'QMIX':('qmix_v1_20261108','train_qmix.py','plain'),'QPLEX':('qplex_v1_20261108','train_qplex.py','hard')}[a.model]
        sys.path.insert(0,str(repo/folder));sys.path.insert(0,str(repo));p=__import__('policy_core');s=p.geometry.search;s.check_lock=lambda lock:verify_lock(lock,base,inventory);s.Dataset=lambda:make_dataset(s,base,inventory)
        mod=import_file('portable_training_algorithm',repo/folder/script);bc=repo/('pilot_rescue_1000_88126' if a.seed==88126 else f'formal/rescue_{a.seed}')/'checkpoint.pt';run=mod.Run(a.seed,bc,arm,a.device)
        # Original checkpoint of this branch carries the exact original lock.
        template_path=PACKAGE/a.model/'training/checkpoints'/f'{a.seed}.pt';expected=json.loads((PACKAGE/a.model/'training/checkpoints.json').read_text(encoding='utf8'))['seeds'][str(a.seed)]['sha256'];assert sha(template_path)==expected
        template=torch.load(template_path,map_location='cpu',weights_only=False);lock=template['source_lock'];verify_lock(lock,base,inventory);del template
    else:run,s,lock=make_run(a.model,a.seed,stage,base,inventory,source,a.device,a.allow_runtime_change)
    begin=run.joint;updates=[];trajectory=[]
    if a.model=='MAPPO':
        for _ in range(a.steps//128):data,trace=run.collect(128);trajectory.extend(trace);stats=run.update(data,epochs=5);updates.append({'joint':run.joint,'epochs':len(stats),'delivered':sum(x['delivered'] for x in trace)})
        snapshot=run.snapshot()
    else:
        for _ in range(a.steps):
            trajectory.append(run.step())
            if run.joint>=64 and run.joint%32==0:updates.append(run.update())
        snapshot=run.snapshot(lock)
    meta={'model':a.model,'seed':a.seed,'inventory_sha256':MANIFEST_SHA256,'source_sha256':sha(checkpoint),'source_start_joint':begin,'end_joint':run.joint,'patched_sources':patched,'device':a.device,'device_original':original_device,'runtime_original':original_runtime,'runtime_current':current_runtime,'allow_runtime_change':a.allow_runtime_change,'bitwise_reproducibility':'not_assured' if a.allow_runtime_change else 'requires_matching_environment'}
    (output/'trajectory.json').write_text(json.dumps(trajectory,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')
    target=output/'checkpoint.pt';torch.save({'portable_continuation':meta,'state':snapshot},target)
    (output/'result.json').write_text(json.dumps({'status':'COMPLETED','metadata':meta,'updates':updates,'checkpoint_sha256':sha(target)},ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps({'status':'COMPLETED','model':a.model,'seed':a.seed,'start':begin,'end':run.joint,'updates':len(updates),'checkpoint_sha256':sha(target)}),flush=True)
if __name__=='__main__':main()



