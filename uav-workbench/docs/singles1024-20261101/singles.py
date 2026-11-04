"""Six single-learner controls on the frozen all-pairs environment and data."""
import argparse, ctypes, gc, hashlib, importlib.util, inspect, json, os, shutil, sys, time, traceback
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
PAIR = ROOT/'uav-all-pairs-20261101/run01'
LEGACY = ROOT/'uav-linked-training-20261101'
RUN = BASE/'run01'
OUT = Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
REPO = Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031')
ARCHIVE = OUT/'uav-singles1024-20261101'
DOC = REPO/'uav-workbench/docs/singles1024-20261101'
ALGOS = ('happo','hatrpo','hasac','maac','mat','coma')
ROLE = (0,0,0,0)
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');os.replace(tmp,p)
def emit(status,**kw):
    value=dict(status=status,pid=os.getpid(),updated=time.strftime('%Y-%m-%d %H:%M:%S'),**kw)
    write(RUN/'queue-status.json',value)
    for root in (ARCHIVE,DOC):write(root/'live-run.json',dict(**value,live_path=str(RUN)))
    print(json.dumps(value,ensure_ascii=False),flush=True)
def stopped():return any((p/'STOP_AFTER_CHECKPOINT').exists() for p in (RUN,ARCHIVE))

def register():
    RUN.mkdir(parents=True,exist_ok=False)
    pair=read(PAIR/'protocol.json')
    sources=dict(pair['source_hashes'])
    sources.update({str(p):sha(p) for p in (Path(__file__),BASE/'protocol.md')})
    protocol=dict(date='2026-11-01',authorization='User adds singles to1024 and permits2048 under existing requirements; list all single/pair data.',
      algorithms=ALGOS,prohibited=['mappo','qmix','vdn','ippo','dqn','ddpg'],parents=pair['parents'],source_hashes=sources,
      pair_protocol_sha256=sha(PAIR/'protocol.json'),dataset_sha256=sha(PAIR/'dataset.json'),train_seed=1001,threads=2,
      schedule='All6 singles reach1024 NEW adaptation episodes. Then each eligible single continues latest full state to2048. Parent256/512 is reported separately. No performance early stop before1024.',
      controls=pair['controls'],dataset=pair['dataset'],ranking=pair['ranking'],extend2048=pair['extend2048'],
      validation='Same6 parents,2 layouts,2 stochastic seeds+1fixed =36 unique episodes. One learner controls all4 UAVs; no duplicate role assignment. Pairs have72 with2roles. Compare rates and matched strata, not raw completed counts.',
      learning='One Learner checkpoint, all4 owned actors. Reuse frozen mixed_core on/off updates; two fresh episodes per on-policy update; HASAC/MAAC4 batches64 per block, replay1024. Sampler sees two references to SAME learner for interface compatibility; never two models/optimizers. MAT retains causal joint-action decoder. Fresh replay/optimizers, same parentweights and zero role padding as pairs.',
      checkpoint='Every64episodes; evaluation0/64/128/192/256 then every128; no-cover0/256/1024 and2048 ifeligible. Best and latest separately; best0 explicitly means no new learning gain.',
      budget='One heavy process after all-pairs completed AND integrity audit passed. Up to6*2048=12288 new common environment episodes.8h active wall limit at checkpoint boundaries;8GiB free; cooperative stop; no automatic failure retry.',
      limitations='Synthetic16train/6validation fields,1trainingseed;50sealedtests untouched. No field rescue or battery/communication/building/weather validation. Different ownership/update count is intrinsic; compare environmentepisodes and also report optimizerupdates. Historical4train/2validation rows remain separate.')
    write(RUN/'protocol.json',protocol);shutil.copy2(PAIR/'dataset.json',RUN/'dataset.json')
    for root in (ARCHIVE,DOC):
        root.mkdir(parents=True,exist_ok=True)
        for p in (Path(__file__),BASE/'protocol.md',RUN/'protocol.json',RUN/'dataset.json'):shutil.copy2(p,root/p.name)
    emit('waiting_for_pairs',predecessor=str(PAIR),required='completed + integrity-audit passed')

def wait_for_pairs():
    state=read(PAIR/'queue-status.json');kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x00100000,False,state.get('pid',0));started=time.monotonic()
    try:
        while handle:
            if stopped():raise InterruptedError('Manual stop while waiting')
            result=kernel.WaitForSingleObject(handle,30000)
            if result==0:break
            if result!=258:raise RuntimeError(f'Predecessor process wait failed:{result}')
            if time.monotonic()-started>6*3600:raise TimeoutError('Predecessor wait exceeded6h; no retry')
    finally:
        if handle:kernel.CloseHandle(handle)
    state=read(PAIR/'queue-status.json')
    assert state['status']=='completed',f'Predecessor did not complete: {state}'
    assert read(PAIR/'integrity-audit.json')['status']=='passed','Predecessor integrity audit missing/failed'
    assert not (PAIR/'STOP_AFTER_CHECKPOINT').exists(),'Predecessor manual stop marker retained'

def bootstrap():
    global e,np,torch,update_on,update_off
    import numpy as np
    import torch
    sys.path[:0]=[str(LEGACY),str(ROOT/'uav-all-pairs-20261101')]
    spec=importlib.util.spec_from_file_location('single_engine',LEGACY/'linked_train.py')
    e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
    from mixed_core import install
    e.ARCHIVE=ARCHIVE;e.DOC=DOC;e.ROLES=(ROLE,)
    for a,p in read(RUN/'protocol.json')['parents'].items():e.PARENTS[a]=Path(p['path'])
    install(e)
    local=inspect.getclosurevars(e.update_block).nonlocals
    update_on,update_off=local['update_on'],local['update_off']
    Mixed=e.Controller
    class SingleController(Mixed):
        def decide(self,models,obs,boundary,roles,deterministic=False):
            assert tuple(roles)==ROLE
            if models is None:return super().decide(None,obs,boundary,roles,deterministic)
            assert len(models)==1
            return super().decide([models[0],models[0]],obs,boundary,roles,deterministic)
    def single_update(models,rows,replay):
        assert len(models)==1 and all(np.array_equal(r['obs'][0,-4:],ROLE) for r in rows)
        m=models[0];loss={}
        if m.algorithm in ('happo','hatrpo','mat','coma'):loss[m.algorithm]=update_on(m,rows,0)
        replay.extend(rows);del replay[:-e.REPLAY_SIZE]
        if m.algorithm in ('hasac','maac'):
            loss[m.algorithm]=[update_off([m,m],0,e.random.sample(replay,min(64,len(replay)))) for _ in range(4)]
        return loss
    e.Controller=SingleController;e.update_block=single_update
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)

def inputs():
    proto=read(RUN/'protocol.json')
    assert sha(RUN/'dataset.json')==proto['dataset_sha256']
    for p,h in proto['source_hashes'].items():assert sha(p)==h,p
    for p in proto['parents'].values():assert sha(p['path'])==p['sha256'],p['path']
    cases={'train':[],'validation':[]}
    for r in read(RUN/'dataset.json')['cases']:
        p=Path(r['path']);assert sha(p)==r['sha256']
        assert r['finite_nonnegative'] and abs(r['mass_residual_m3'])<=max(1e-6,abs(r['source_volume_m3'])*1e-9)
        water=np.load(p,mmap_mode='r',allow_pickle=False)
        if r['split']=='train':cases['train'].append((p.stem,water))
        else:cases['validation'].append(('legacy' if r['seed'] in (2101,2201) else 'new',p.stem,water))
    assert len(cases['train'])==16 and len(cases['validation'])==6
    return cases

def preflight(cases):
    checks=[];water=cases['train'][0][1];dest=RUN/'preflight';dest.mkdir()
    for a in ALGOS:
        e.seed_all(41);m,ck=e.adapted(a);old=e.Learner(a,e.OLD_DIM);old.load_state_dict(ck['model'])
        obs=e.Env(water,9001).observe();mask=np.ones((4,9),bool)
        e.seed_all(73);x,l,v=old.act(obs,mask);e.seed_all(73);y,k,w=m.act(e.augment(obs,ROLE),mask)
        assert np.array_equal(x,y) and np.allclose(l,k,atol=3e-6) and abs(v-w)<3e-6,a
        e.seed_all(91);models=[m];_,rows=e.rollout(models,water,990001,ROLE,True)
        assert any((r['mask'].sum(-1)>1).any() for r in rows),a
        assert all(np.abs(r['logp'][r['mask'].sum(-1)==1]).max(initial=0)<1e-6 for r in rows),a
        before=e.model_hash(models);replay=[];e.update_block(models,rows,replay)
        assert e.model_hash(models)!=before and m.updates>0,a
        p=dest/f'{a}.pt';e.save(p,e.checkpoint(models,replay,2,{}))
        loss=e.update_block(models,rows,replay);expected=e.model_hash(models)
        saved=torch.load(p,map_location='cpu',weights_only=False);assert len(saved['models'])==1
        restored,rp=e.restore(saved);loss2=e.update_block(restored,rows,rp)
        assert e.model_hash(restored)==expected and loss==loss2,a
        checks.append(dict(algorithm=a,parent_equivalence=True,one_learner=True,masked_forced_logp=True,learns=True,exact_serialized_resume=True))
        write(RUN/'preflight-progress.json',dict(completed=len(checks),total=6,checks=checks))
        print(json.dumps(dict(preflight=a,passed=True)),flush=True)
        del m,old,models,restored,replay,rp,saved;gc.collect()
    write(RUN/'preflight.json',dict(status='passed',checks=checks))

def mirror(cid=None):
    folders=[RUN]+([RUN/cid] if cid else [])
    for folder in folders:
        for p in folder.iterdir():
            if p.is_file() and p.suffix in ('.json','.jsonl','.md'):
                for root in (ARCHIVE,DOC):
                    dest=root/p.relative_to(RUN);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    if cid:
        for p in (RUN/cid).glob('eval-*/*.npz'):
            for root in (ARCHIVE,DOC):
                dest=root/p.relative_to(RUN)
                if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)

def budget(start):
    if stopped():raise InterruptedError('STOP_AFTER_CHECKPOINT requested')
    if shutil.disk_usage(RUN).free<8*1024**3:raise RuntimeError('Less than8GiB free; checkpoint preserved')
    if time.monotonic()-start>8*3600:raise TimeoutError('Active8h budget reached at checkpoint; preserved')

def stage(cid,cases,target,identity,start):
    folder=RUN/cid;folder.mkdir(exist_ok=True);emit('training',current=cid,stage=target)
    if (folder/'latest.json').exists():
        latest=read(folder/'latest.json');p=folder/latest['file'];assert sha(p)==latest['sha256']
        ck=torch.load(p,map_location='cpu',weights_only=False);assert ck['identity']==identity
        models,replay=e.restore(ck);begin=ck['joint_episode'];bi=read(folder/'best.json');best=read(folder/bi['evaluation']);best_ep=bi['joint_episode']
    else:
        e.seed_all(e.TRAIN_SEED);models=[e.adapted(cid)[0]];replay=[];begin=0;best_ep=0
        best=e.evaluate(models,cases['validation'],folder,0)
        e.save(folder/'checkpoint-0000.pt',e.checkpoint(models,replay,0,identity))
        write(folder/'best.json',dict(joint_episode=0,evaluation='eval-0000.json',file='checkpoint-0000.pt',quality=e.quality(best)))
        e.evaluate(models,cases['validation'],folder,0,coverage=False)
        mirror(cid)
    pending=[];started=time.perf_counter()
    for ep in range(begin+1,target+1):
        name,water=cases['train'][(ep-1)%len(cases['train'])]
        metric,rows=e.rollout(models,water,18010000+ep,ROLE,True);pending.extend(rows);loss=None
        if ep%2==0:loss=e.update_block(models,pending,replay);pending=[]
        record=dict(joint_episode=ep,case=name,layout=18010000+ep,roles=ROLE,loss=loss,optimizer_updates=models[0].updates,elapsed_stage_s=time.perf_counter()-started,**metric)
        with (folder/'episodes.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        write(folder/'status.json',dict(status='training',joint_episode=ep,target=target,best_joint_episode=best_ep,optimizer_updates=models[0].updates))
        check=ep%64==0 or (ep%2==0 and stopped())
        if check:
            p=folder/f'checkpoint-{ep:04d}.pt';e.save(p,e.checkpoint(models,replay,ep,identity))
            write(folder/'latest.json',dict(joint_episode=ep,file=p.name,sha256=sha(p)))
            print(json.dumps(dict(candidate=cid,episode=ep,target=target)),flush=True)
        if (ep<=256 and ep%64==0) or (ep>256 and ep%128==0):
            val=e.evaluate(models,cases['validation'],folder,ep)
            if e.quality(val)>e.quality(best):best,best_ep=val,ep
            write(folder/'best.json',dict(joint_episode=best_ep,evaluation=f'eval-{best_ep:04d}.json',file=f'checkpoint-{best_ep:04d}.pt',quality=e.quality(best)))
            if ep in (256,1024,2048):e.evaluate(models,cases['validation'],folder,ep,coverage=False)
        if check:mirror(cid);budget(start)
    write(folder/'status.json',dict(status='stage_completed',joint_episode=target,target=target,best_joint_episode=best_ep,optimizer_updates=models[0].updates,best_summary=best['summary']))
    mirror(cid);del models,replay;gc.collect()

def main():
    register();wait_for_pairs();emit('preflight');bootstrap();cases=inputs();active=time.monotonic();preflight(cases);mirror()
    identity=dict(protocol_sha256=sha(RUN/'protocol.json'),dataset_sha256=sha(RUN/'dataset.json'))
    for cid in ALGOS:budget(active);stage(cid,cases,1024,identity,active)
    decisions={cid:e.trend2048(RUN/cid) for cid in ALGOS};write(RUN/'extension-1024.json',decisions);mirror()
    for cid,d in decisions.items():
        if d['eligible']:budget(active);stage(cid,cases,2048,identity,active)
    inputs();audit={cid:e.validate_records(RUN/cid) for cid in ALGOS}
    write(RUN/'integrity-audit.json',dict(status='passed',records=audit,source_and_parents_unchanged=True))
    emit('completed',elapsed_s=time.monotonic()-active,extension=decisions,completed={cid:read(RUN/cid/'status.json')['joint_episode'] for cid in ALGOS});mirror()
    # Reporting is a separate light post-processing job; it cannot change selection/training.
    import subprocess
    report=BASE/'refresh.py'
    if report.exists():subprocess.run([sys.executable,str(report),'--workbook'],check=True)

if __name__=='__main__':
    try:main()
    except BaseException as ex:
        write(RUN/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()))
        emit('manual_stopped' if isinstance(ex,InterruptedError) else 'failed',error=repr(ex));mirror()
        raise
