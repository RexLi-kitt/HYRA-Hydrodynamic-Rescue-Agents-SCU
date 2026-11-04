"""TRAIN-only parameter screening; shared patch fog and observed-flow forecast."""
from pathlib import Path
import sys,json,time,copy,traceback,hashlib,argparse
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'flood-drop-regional-refloat-v1'))
from experiment import Model,History,features,pool,SearchEnv,read,write,sha
from refloat_environment import sample_water,make_scenario,Config

def transport(demand,u,v,seconds):
    """Forward bilinear transport; mass leaving square is lost, not edge-clipped."""
    rr,cc=np.mgrid[:10,:10];xx=cc+u*seconds/100;yy=rr+v*seconds/100
    x0=np.floor(xx).astype(int);y0=np.floor(yy).astype(int);fx=xx-x0;fy=yy-y0
    moved=np.zeros((10,10))
    for dx,dy,w in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)):
        x=x0+dx;y=y0+dy;valid=(x>=0)&(x<10)&(y>=0)&(y<10)
        np.add.at(moved,(y[valid],x[valid]),(demand*w)[valid])
    assert np.isfinite(moved).all() and (moved>=0).all()
    return moved

class FlowFogEnv(SearchEnv):
    def __init__(self,water,scenario,method,predict=None):
        self.method=method;self.predict=predict;self.history=History();self.prediction=None;self.predicted_at=None
        self.patch=bool(scenario.get('patch_fog',False));self.fog_orientation=int(scenario['seed'])%2
        self.sensor_radii=None;self.fog_uav_observations=0;self.observation_calls=0
        super().__init__(water,scenario,'coverage100')

    def radii(self):
        # Synthetic central 400m band, alternating orientation by seed. Current local visibility is public.
        if not self.patch:return np.full(self.cfg.uavs,self.cfg.sensor)
        q=self.pos[:,self.fog_orientation];dense=(q>=300)&(q<=700)
        return np.where(dense,min(20.,self.cfg.sensor),self.cfg.sensor)

    def observe_world(self):
        if not self.patch:return super().observe_world()
        before=int(self.known.sum());found=int(self.discovered.sum());r=self.radii();self.sensor_radii=r
        self.fog_uav_observations+=int((r<self.cfg.sensor).sum());self.observation_calls+=1
        mask=np.zeros((100,100),bool)
        for p,radius in zip(self.pos,r):mask|=np.sum((self.cells-p)**2,axis=-1)<=radius**2
        self.known|=mask;self.last[mask]=self.t
        a=self.t//5;b=min(a+1,300);alpha=self.t%5/5
        self.measured[:,mask]=(1-alpha)*self.water[a][:,mask]+alpha*self.water[b][:,mask]
        distances=np.linalg.norm(self.person[:,None]-self.pos[None],axis=-1)
        self.visible=(self.status>0)&(self.status!=4)&(distances<=r[None]).any(1)
        ids=np.flatnonzero(self.visible);self.first_seen[ids[self.first_seen[ids]<0]]=self.t
        self.discovered[ids]=True;self.seen_xy[ids]=self.person[ids];self.seen_at[ids]=self.t
        if len(ids):
            self.seen_v[ids]=sample_water(self.water,self.t,self.person[ids])[:,1:]
            self.seen_v[ids[self.stuck[ids]]]=0
        self.belief[ids]=np.where(self.status[ids]==1,1,2)
        return int(self.known.sum())-before,int(self.discovered.sum())-found

    def search_score(self,stale,risk):
        if self.method=='coverage100':return super().search_score(stale,risk)
        base=stale*(1+risk)
        prior=np.where(self.known,np.where(self.measured[0]>=self.cfg.wet_depth,2.,.5),1.)
        cover=pool(np.where(self.known,np.minimum(stale,2.),2.)*prior)
        fresh=self.known&(self.t-self.last<=120)
        u=pool(np.where(fresh,self.measured[1],0));v=pool(np.where(fresh,self.measured[2],0))
        if self.method=='flow_coverage':target=.5*cover+.5*transport(cover,u,v,60)
        else:
            x=self.history.get(self)
            if self.prediction is None or self.predicted_at!=self.history.time:
                self.prediction=np.maximum(0,self.predict(x)).reshape(10,10);self.predicted_at=self.history.time
            target=self.prediction
            if self.method.endswith('_flowblend'):
                target=.5*target+.5*transport(target,u,v,60)
                # Normalize only spatial emphasis; keep coverage half of target, no truth-derived scale.
                target=.5*cover/(cover.mean()+1e-6)+.5*target/(target.mean()+1e-6)
            else:return base*(1+8*np.repeat(np.repeat(target,10,0),10,1))
        return base*(.25+np.repeat(np.repeat(target,10,0),10,1))

METHODS=['coverage100','flow_coverage','linear-16','gru-16','gru-32','linear-16_flowblend','gru-32_flowblend']
PARAMS=[dict(name=f'view60-w{w}',sensor=60,window=w,patch=False) for w in (240,360,480)]+[
    dict(name=f'view{v}-w360',sensor=v,window=360,patch=False) for v in (20,40,100)]+[
    dict(name='patch20-60-w360',sensor=60,window=360,patch=True)]

def aggregate(rows):
    n=sum(r['people'] for r in rows);s=sum(r['supplied'] for r in rows)
    return dict(cases=len(rows),people=n,supplied=s,people_rate=s/n,all_rate=sum(r['supplied']==r['people'] for r in rows)/len(rows),
        p95_rate=sum(r['supplied']>=int(np.ceil(.95*r['people'])) for r in rows)/len(rows),
        unseen=sum(r['unseen_expired'] for r in rows),refloats=sum(r['refloat_events'] for r in rows))

def variant(c,p):
    c=copy.deepcopy(c);s=c['scenario'];s['events']=s['events'][:40]
    s['config'].update(people=40,duration=1500,window=p['window'],sensor=float(p['sensor']))
    for e in s['events']:e['deadline']=e['fall']+p['window']
    assert max(e['deadline'] for e in s['events'])<=1500
    s['patch_fog']=p['patch'];c['id']+='-'+p['name'];return c

def run(out,previous,weights):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=[];done=0
    torch.set_num_threads(4)
    models={}
    for name in ('linear-16','gru-16','gru-32'):
        ck=torch.load(weights/(name+'.pt'),map_location='cpu',weights_only=True);m=Model(ck['kind'],ck['mean']);m.load_state_dict(ck['state']);m.eval()
        mu=np.array(ck['mu'],np.float32);sd=np.array(ck['sd'],np.float32)
        def pred(x,m=m,mu=mu,sd=sd):
            with torch.no_grad():return m(torch.tensor((x-mu)/sd)).numpy()
        models[name]=pred
    archive=read(previous/'scenarios.json');cal=read(weights/'protocol.json')['calibration_parents']
    selected=[min([c for c in archive['train'] if c['field']['parent']==p],key=lambda c:c['scenario']['seed']) for p in cal]
    assert len(selected)==4
    sources={str(p):sha(p) for folder in ('flood-drop-flow-fog-v1','flood-drop-regional-refloat-v1','flood-drop-refloat-v1','flood-drop-search-probe-v1') for p in (HERE.parent/folder).glob('*.py')}
    water_hashes={c['field']['path']:sha(c['field']['path']) for c in selected+archive['development']['view60']}
    protocol=dict(version='flow-fog-train-screen-v1',parameters=PARAMS,methods=METHODS,source_hashes=sources,water_hashes=water_hashes,
        weights={str(weights/(n+'.pt')):sha(weights/(n+'.pt')) for n in models},scenario_sha256=sha(previous/'scenarios.json'),
        train_screen='4 previous calibration hydraulic parents, lowest-seed roster each, first40 independently shuffled IDs; same persons across all7 conditions. T1500,end960,w240/360/480; accepted events retained, no favorable person deletion.',
        selection='TRAIN only: target85..95% people, >=15pp p95 scene improvement over BOTH nonneural baselines, no people regression against either. Prefer candidates meeting goal; then smaller distance to85..95, higher people, higher p95 gap, stable declared method/parameter order. Allscene gap also reported. Four scenes too small for proof.',
        validation='12 fresh event seeds on6 previously exposed DEV hydraulic parents; NOT untouched-map final test. Freeze selected parameter/model before generating seeds. Evaluate all7 methods, report preselected only as candidate. No further selection from this holdout.',
        fog='Shared per-UAV current location central400m band; x/y orientation seed%2; local radius20 vs background parameter sensor. No known map preview; observations outside disks absent. No weather flight-risk claim.',
        flow='60second coarse forward bilinear advection using observed flow age<=120s; unknown flow0, exiting masslost. Coverage gets same forecast. Learned fusion halfnormalizedcoverage/halfnormalizedprediction; halfnow/halfforecast. Known-person existing30s velocity extrapolation unchanged.',
        scope='Frozen weights no training; selection16/32 now by TRAIN taskpeople, not reusedDEV. Fog and20m/window480 shift from original training; failures retained. No final physical accuracy claim. All fallthresholds2m/1mps,drop10s unchanged. No production edits or remote push.',
        target='People85..95%; >=15 percentage points in at-least95%-people scenes; allpeople scenes separate. Finite exploratory budget, not guaranteed achievement.',
        budget='196 TRAIN screening +84 fresh-event validation=280 rollouts; one serialCPU worker4torchthreads, no new optimization epochs; stopfile STOP, 45min ceiling.')
    write(out/'protocol.json',protocol)
    def status(state,**kw):write(out/'status.json',dict(status=state,elapsed_s=time.monotonic()-start,rollouts=done,**kw))
    def evaluate(c,method,phase,param):
        nonlocal done
        if (out/'STOP').exists():raise InterruptedError('STOP requested')
        if time.monotonic()-start>2700:raise TimeoutError('45min ceiling')
        status(phase,method=method,case=c['id'],parameter=param)
        predictor=models.get(method.removesuffix('_flowblend'))
        e=FlowFogEnv(np.load(c['field']['path'],mmap_mode='r'),c['scenario'],method,predictor)
        while True:
            _,_,finished,_=e.step(e.choices()[1])
            if finished:break
        persons=[dict(id=j,fall=int(e.fall[j]),deadline=int(e.deadline[j]),first_seen=float(e.first_seen[j]),supplied_at=float(e.delivered_at[j]),status=int(e.status[j])) for j in range(e.cfg.people)]
        supplied=int((e.status==2).sum());expired=int((e.status==3).sum());outside=int((e.status==4).sum())
        assert supplied+expired+outside==e.cfg.people
        assert np.isfinite(e.pos).all() and np.isfinite(e.person).all()
        for p in persons:
            if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
        row=dict(phase=phase,parameter=param,method=method,case=c['id'],parent=c['field']['parent'],people=e.cfg.people,supplied=supplied,expired=expired,outside=outside,
            unseen_expired=sum(p['status']==3 and (p['first_seen']<0 or p['first_seen']>p['deadline']) for p in persons),refloat_events=int(e.refloat_counts.sum()),fog_uav_observations=e.fog_uav_observations,persons=persons)
        with (out/'results.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        rows.append(row);done+=1
    try:
        scenarios=[dict(parameter=p['name'],**variant(c,p)) for p in PARAMS for c in selected];write(out/'train-scenarios.json',scenarios)
        for p in PARAMS:
            for method in METHODS:
                for c in [c for c in scenarios if c['parameter']==p['name']]:evaluate(c,method,'train_screen',p['name'])
        summary={p['name']:{m:aggregate([r for r in rows if r['parameter']==p['name'] and r['method']==m]) for m in METHODS} for p in PARAMS}
        candidates=[]
        for pi,p in enumerate(PARAMS):
            q=summary[p['name']];bp=max(q[m]['people_rate'] for m in METHODS[:2]);bs=max(q[m]['p95_rate'] for m in METHODS[:2])
            for mi,m in enumerate(METHODS[2:]):
                a=q[m];distance=max(.85-a['people_rate'],0,a['people_rate']-.95);gap=a['p95_rate']-bs
                met=distance==0 and gap>=.15 and a['people_rate']>=bp
                candidates.append(dict(parameter=p,method=m,goal_met=met,people_rate=a['people_rate'],gap95=gap,people_gap=a['people_rate']-bp,
                    rank=[int(met),-distance,a['people_rate'],gap,-pi,-mi]))
        chosen=max(candidates,key=lambda c:c['rank']);write(out/'train-summary.json',summary);write(out/'selection.json',dict(chosen=chosen,all_candidates=candidates,selected_before_validation=True))
        # Freeze complete fresh event set after selection, with deterministic seeds declared by formula.
        fields={c['field']['parent']:c['field'] for c in archive['development']['view60']};hold=[]
        assert not set(fields)&set(cal)
        for i,(parent,field) in enumerate(sorted(fields.items())):
            for repeat in range(2):
                template=copy.deepcopy(archive['development']['view60'][0]['scenario']['config']);template.update(duration=1500,window=chosen['parameter']['window'],sensor=float(chosen['parameter']['sensor']))
                s=make_scenario(np.load(field['path'],mmap_mode='r'),2026110300+2*i+repeat,Config(**template),fall_end=960,pool_size=120);s['patch_fog']=chosen['parameter']['patch']
                hold.append(dict(id=f'fresh-{i}-{repeat}',field=field,scenario=s))
        write(out/'validation-scenarios.json',hold)
        for method in METHODS:
            for c in hold:evaluate(c,method,'fresh_event_validation',chosen['parameter']['name'])
        result={m:aggregate([r for r in rows if r['phase']=='fresh_event_validation' and r['method']==m]) for m in METHODS}
        a=result[chosen['method']];bp=max(result[m]['people_rate'] for m in METHODS[:2]);bs=max(result[m]['p95_rate'] for m in METHODS[:2]);ba=max(result[m]['all_rate'] for m in METHODS[:2])
        met=.85<=a['people_rate']<=.95 and a['people_rate']>=bp and a['p95_rate']-bs>=.15
        for p,h in {**sources,**water_hashes,**protocol['weights']}.items():assert sha(p)==h,p
        write(out/'summary.json',dict(status='completed',rollouts=done,selected=chosen,validation=result,goal_met=met,p95_gap=a['p95_rate']-bs,all_gap=a['all_rate']-ba,people_gap=a['people_rate']-bp,source_hashes_unchanged=True,not_final_map_holdout=True))
        status('completed',goal_met=met)
    except BaseException as ex:
        write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);a=p.parse_args();run(a.output,a.previous,a.weights)
