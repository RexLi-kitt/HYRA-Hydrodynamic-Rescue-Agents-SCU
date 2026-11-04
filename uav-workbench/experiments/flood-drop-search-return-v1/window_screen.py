"""Paired deadline screen, followed by untouched confirmation if >=85% on DEV."""
import copy,time,contextlib,traceback,json
import numpy as np
import torch
from train import ROOT,REPO,HERE,BASE,PREVIOUS,OLD_MODEL,read,write,sha,MemoryGRU,FineEnv,predictor,FlowFogEnv,Config,make_scenario,generate
OUT=ROOT/'uav-window85-20261104/run01'
PRIOR=ROOT/'uav-search-return-train-20261104/run01'
METHODS=['coverage100','flow_coverage','fine142-e16']

def run():
    OUT.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);start=time.monotonic();rows=[];audits=[]
    cases=read(PRIOR/'scenarios.json')['test'];sources={str(p):sha(p) for folder in (HERE,BASE,HERE.parent/'flood-drop-fine-memory-v1',HERE.parent/'flood-drop-search-redesign-v1') for p in folder.glob('*.py')}
    sources.update({str(p):sha(p) for p in (REPO/'terrain-workbench/terrain-lab').glob('flood_*.py')})
    inputs={str(p):sha(p) for p in (PRIOR/'scenarios.json',PRIOR/'input-lock.json',OLD_MODEL)}
    oldlock=read(PRIOR/'input-lock.json');assert sha(PRIOR/'scenarios.json')==oldlock['scenarios_sha256']
    protocol=dict(date='2026-11-04',windows=[420,540,660],duration=2400,methods=METHODS,checkpoint=str(OLD_MODEL),source_sha256=sources,input_sha256=inputs,
      development='All6 previously inspected TEST parents1104801..06 are now DEV, not new test. Same40 people,fall times,initial coordinates,exposure IDs,traps. Change deadline only, common2400s horizon for all windows.',
      water='Regenerate same seed with2400s physical SWE integration; assert all421 old frames exactly equal before using. Never pad/repeat final frame. New mass/finite checks inherited from generate.',
      selection='Smallest registered window with frozen fine142-e16 DEV supplied>=85%; otherwise no confirmation or automatic new tuning. Record all baselines and all windows; reaching85 via deadline relaxation is difficulty change, not model improvement.',
      confirmation='If selected:6untouched parents1104901..06,scenario seeds2026111600..05. Generate base420s roster with fall_end1680,then same roster selected window. Three methods atselected window plus same GRU420 control if different. Freeze chosen window before generating/evaluating. No confirmation retuning.',
      unchanged='20m sensor,4UAV,40people,10s service,2m/1mps thresholds,random refloat,physical flight/separation; no new training or policy modification. Prior weights trained at420s/2100s, transferred inference domain shift disclosed.',
      acceptance='85% means people supplied, not whole-scene success. Report full/95% scenes and15pp scene-gap separately. Six confirmation parents small; no claim universal/field accuracy.',
      budget='One CPU4 workload,54DEV and atmost24confirmation runs;1800s/STOP guard;no remote push.')
    write(OUT/'protocol.json',protocol)
    def guard():
        if (OUT/'STOP').exists():raise InterruptedError('STOP requested')
        if time.monotonic()-start>1800:raise TimeoutError('1800s cap')
    def status(state,**kw):write(OUT/'status.json',dict(state=state,elapsed_s=time.monotonic()-start,evaluations=len(rows),**kw))
    ck=torch.load(OLD_MODEL,map_location='cpu',weights_only=True);m=MemoryGRU(ck['mean'],ck['coarse']);m.load_state_dict(ck['state']);m.eval()
    pred=predictor(m,np.array(ck['mu'],np.float32),np.array(ck['sd'],np.float32))
    def evaluate(w,c,window,method,phase):
        guard();status('evaluating',phase=phase,case=c['id'],window=window,method=method)
        sc=copy.deepcopy(c['scenario']);sc['config']['duration']=2400;sc['config']['window']=window
        for ev,old in zip(sc['events'],c['scenario']['events']):
            ev['deadline']=ev['fall']+window
            assert {k:v for k,v in ev.items() if k!='deadline'}=={k:v for k,v in old.items() if k!='deadline'}
            assert ev['deadline']<=2400
        assert sc['traps']==c['scenario']['traps']
        e=FineEnv(w,sc,pred,.25) if method=='fine142-e16' else FlowFogEnv(w,sc,method)
        while True:
            if e.t%120==0:guard()
            if e.step(e.choices()[1])[2]:break
        persons=[dict(id=j,fall=int(e.fall[j]),deadline=int(e.deadline[j]),first_seen=float(e.first_seen[j]),supplied_at=float(e.delivered_at[j]),status=int(e.status[j])) for j in range(e.cfg.people)]
        assert np.isfinite(e.pos).all() and np.isfinite(e.person).all() and (e.status>=2).all()
        for p in persons:
            if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
        row=dict(phase=phase,case=c['id'],parent=c['field']['parent'],window=window,method=method,people=e.cfg.people,supplied=int((e.status==2).sum()),all_scene=bool((e.status==2).all()),p95_scene=bool((e.status==2).sum()>=38),
          not_seen_before_deadline=sum(p['status']!=2 and (p['first_seen']<0 or p['first_seen']>=p['deadline']) for p in persons),refloats=int(e.refloat_counts.sum()),persons=persons)
        rows.append(row)
        with (OUT/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
    allcases=dict(development=[],confirmation=[]);waterhash={}
    def field(seed,split):
        status('generating_water',seed=seed)
        with (OUT/'generation.log').open('a',encoding='utf-8') as f,contextlib.redirect_stdout(f):r=generate(OUT/'fields',REPO/'terrain-workbench/terrain-lab',seed,split,duration=2400)
        waterhash[r['path']]=sha(r['path']);return r,np.load(r['path'],mmap_mode='r')
    def aggregate(phase):
        result=[]
        for window in protocol['windows']:
            for method in METHODS:
                rr=[r for r in rows if r['phase']==phase and r['window']==window and r['method']==method]
                if rr:result.append(dict(window=window,method=method,scenes=len(rr),people=sum(r['people'] for r in rr),supplied=sum(r['supplied'] for r in rr),all_scenes=sum(r['all_scene'] for r in rr),p95_scenes=sum(r['p95_scene'] for r in rr),unseen_failures=sum(r['not_seen_before_deadline'] for r in rr),refloats=sum(r['refloats'] for r in rr)))
        return result
    try:
        for c in cases:
            guard();assert sha(c['field']['path'])==oldlock['water_sha256'][c['field']['path']]
            f,w=field(c['field']['seed'],'development');oldw=np.load(c['field']['path'],mmap_mode='r')
            assert np.array_equal(w[:len(oldw)],oldw),'Water prefix changed; do not claim deadline-only comparison'
            audits.append(dict(parent=f['parent'],old_water_prefix_exact=True,mass_residual=f['mass_residual_m3'],finite_nonnegative=f['finite_nonnegative']))
            cc=dict(id=c['id'],field=f,scenario=c['scenario']);allcases['development'].append(cc);write(OUT/'scenarios.json',allcases);write(OUT/'audits.json',audits)
            for window in protocol['windows']:
                for method in METHODS:evaluate(w,cc,window,method,'development')
            print(json.dumps(dict(case=c['id'],evaluations=len(rows))),flush=True)
        dev=aggregate('development');eligible=[r['window'] for r in dev if r['method']=='fine142-e16' and r['supplied']/r['people']>=.85]
        chosen=min(eligible) if eligible else None
        write(OUT/'selection.json',dict(chosen_window=chosen,development=dev,rule=protocol['selection'],before_confirmation=True))
        print(json.dumps(dict(selection=chosen,development=dev)),flush=True)
        if chosen is not None:
            cfg=Config(**cases[0]['scenario']['config']);cfg=Config(**{**cfg.__dict__,'duration':2400,'window':420})
            for i,seed in enumerate(range(1104901,1104907)):
                guard();f,w=field(seed,'confirmation');sc=make_scenario(w,2026111600+i,cfg,fall_end=1680,pool_size=120)
                c=dict(id=f'confirmation-{i}',field=f,scenario=sc);allcases['confirmation'].append(c);write(OUT/'scenarios.json',allcases)
                for method in METHODS:evaluate(w,c,chosen,method,'confirmation')
                if chosen!=420:evaluate(w,c,420,'fine142-e16','confirmation')
                print(json.dumps(dict(case=c['id'],evaluations=len(rows))),flush=True)
        for p,h in {**sources,**inputs,**waterhash}.items():assert sha(p)==h,p
        write(OUT/'input-lock.json',dict(water_sha256=waterhash,scenarios_sha256=sha(OUT/'scenarios.json')))
        write(OUT/'summary.json',dict(chosen_window=chosen,development=dev,confirmation=aggregate('confirmation'),elapsed_s=time.monotonic()-start,evaluations=len(rows),trained=False,source_water_unchanged=True))
        status('completed')
    except BaseException as ex:
        write(OUT/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':run()
