"""Frozen comparison models, independently generated terrain, explicit mechanism fixtures."""
import copy,contextlib,time,os
import numpy as np
import torch
from torch import nn
from train import ROOT,REPO,HERE,read,write,sha,generate,make_scenario,Config
from window_adapter import load,normalized
from count_coupling import CountCoupledEnv
from joint_avoidance import SafeFlowEnv,swept_distances
from memory import FineHistory,public_features
OUT=ROOT/'uav-frozen-rigor-20261104/run01'
OLD=ROOT/'uav-fine-memory-20261104/run01'
METHODS=['coverage100','flow_coverage','statistic','linear','gru-frozen']
MODELS={}

class LinearCount(nn.Module):
    def __init__(self,mean):
        super().__init__();self.head=nn.Linear(256,1)
        nn.init.zeros_(self.head.weight);nn.init.constant_(self.head.bias,float(np.log(np.expm1(max(mean,1e-5)))))
    def forward(self,x):
        b,t,c,h,w=x.shape;z=x.permute(0,3,4,1,2).reshape(b,h,w,t*c)
        return nn.functional.softplus(self.head(z).squeeze(-1))+1e-6

def statistic_shape(x):
    a=x[:,-1] if x.ndim==5 else x[-1][None]
    # Public last observed footprint/age/wetness, no coordinates or hidden people.
    return (.1+a[:,1])*(.25+2*a[:,5]+1-a[:,0])

def initialize_worker():
    torch.set_num_threads(1)
    if (OUT/'predictor-lock.json').exists():
        ck=torch.load(OUT/'linear.pt',map_location='cpu',weights_only=True)
        m=LinearCount(ck['mean']);m.load_state_dict(ck['state']);m.eval()
        mu=np.array(ck['mu'],np.float32);sd=np.array(ck['sd'],np.float32)
        def lin(x):
            with torch.inference_mode():return m(torch.from_numpy(normalized(x,mu,sd,True))[None])[0].numpy().ravel()
        scale=read(OUT/'statistic.json')['scale']
        MODELS.update({'linear':lin,'statistic':lambda x:(scale*statistic_shape(x)[0]).ravel(),'gru-frozen':load(202611142,True)[0]})

def guard():
    if (OUT/'STOP').exists():raise InterruptedError('User STOP')

def generate_case(job):
    family,seed,scenario_seed,cfgdict=job;guard();folder=OUT/'fields';folder.mkdir(exist_ok=True);cfg=Config(**cfgdict)
    with (OUT/f'generation-{seed}.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):
        if family=='original':meta=generate(folder,REPO/'terrain-workbench/terrain-lab',seed,'original',duration=2400)
        else:meta=generate_variant(folder,family,seed)
        sc=make_scenario(np.load(meta['path'],mmap_mode='r'),scenario_seed,cfg,fall_end=1680,pool_size=120)
    return dict(id=f'{family}-{seed}',family=family,field=meta,scenario=sc)

def generate_variant(folder,family,seed):
    import sys
    terrain=REPO/'terrain-workbench/terrain-lab';sys.path.insert(0,str(terrain));os.environ['HYRA_FLOOD_BACKEND']='numpy'
    from flood_solver import FloodSolver
    rng=np.random.default_rng(seed);yy,xx=np.indices((100,100),dtype=float);x=(xx+.5)*10;y=(yy+.5)*10
    if family=='two_valleys':
        c1=260+40*np.sin(y/190);c2=730+45*np.sin(y/210+1)
        bed=.0002*(1000-y)+.000002*np.minimum((x-c1)**2,(x-c2)**2)
        inlets=[[[15,int(c[15,0]/10)+k] for k in range(-2,3)] for c in (c1,c2)]
    elif family=='diagonal':
        along=(x+y)/np.sqrt(2);cross=(x-y)/np.sqrt(2)
        bed=.00015*(1414-along)+.000002*(cross-40*np.sin(along/220))**2
        inlets=[[[15,15+k] for k in range(-2,3)]]
    else:raise ValueError(family)
    for _ in range(4):
        cx,cy=rng.uniform(50,950,2);bed+=rng.uniform(-.1,.2)*np.exp(-((x-cx)**2+(y-cy)**2)/rng.uniform(15000,60000))
    bed-=bed.min();h=np.maximum(0,rng.uniform(.03,.08)-bed)
    solver=FloodSolver(bed,h,10.,boundary='closed',max_timestep=2.5,spatial_order=1,time_integrator='euler',manning=.035)
    peak=float(rng.uniform(8,18));graph=[[0.,1.],[300.,peak],[750.,peak*.5],[1200.,.5],[1500.,0.],[2400.,0.]]
    solver.set_source_schedule([dict(cells=cells,hydrograph=[[t,q/len(inlets)] for t,q in graph]) for cells in inlets])
    path=folder/f'{family}-{seed}.npy';assert not path.exists()
    arr=np.lib.format.open_memmap(path,mode='w+',dtype='<f4',shape=(481,3,100,100));initial=float(h.sum()*100)
    for k,t in enumerate(range(0,2401,5)):
        guard();solver.advance(float(t));u,v=solver.velocity();f=np.stack([solver.h,u,v]);assert np.isfinite(f).all() and (f[0]>=0).all();arr[k]=f
    arr.flush();del arr
    final=float(solver.h.sum()*100);source=float(solver.balance['scheduled_inflow_m3']);residual=final-initial-source
    assert abs(residual)<=max(1e-6,abs(source)*1e-9)
    meta=dict(seed=seed,parent=f'synthetic-{family}-{seed}',family=family,path=str(path),sha256=sha(path),scope='Uncalibrated synthetic SWE structure shift, not field truth',finite_nonnegative=True,mass_residual_m3=residual,source_volume_m3=source,initial_volume_m3=initial,final_volume_m3=final,numerics=solver.numerical_config(),hydrograph=graph,source_cells=inlets)
    write(path.with_suffix('.json'),meta);return meta

def mechanism_cases(cfg):
    """Prescribed h/u/v forcing, not a solved/validated hydrodynamic event."""
    folder=OUT/'fields';folder.mkdir(exist_ok=True)
    levels=np.full(481,2.2,np.float32)
    for lo,hi in [(480,520),(1080,1120),(1680,1720)]:levels[lo//5:hi//5+1]=0
    path=folder/'manufactured-rewet.npy';assert not path.exists()
    a=np.lib.format.open_memmap(path,mode='w+',dtype='<f4',shape=(481,3,100,100));a[:,0]=levels[:,None,None];a[:,1]=np.where(levels>0,.25,0)[:,None,None];a[:,2]=0;a.flush();del a
    cases=[]
    for i in range(4):
        seed=2026112600+i;rng=np.random.default_rng(seed);ev=[]
        for j in range(40):
            fall=int([240,840,1440][j%3]+rng.integers(-20,21));xy=rng.uniform(200,800,2)
            ev.append(dict(fall=fall,deadline=fall+660,xy=xy.tolist(),trigger='threshold',exposure_id=j,field=[2.2,.25,0]))
        rng.shuffle(ev)
        sc=dict(seed=seed,config=cfg.__dict__,events=ev,traps=rng.uniform(200,800,(12,2)).tolist(),conditioning='Manufactured 40 forced-depth falls, same service/window; current/future truth unavailable to policies')
        cases.append(dict(id=f'manufactured-{i}',family='manufactured_rewet',field=dict(parent=f'manufactured-layout-{i}',path=str(path),sha256=sha(path),scope='One shared prescribed field, four layouts; NOT four independent hydraulic maps'),scenario=sc))
    write(OUT/'mechanism-field.json',dict(scope='Prescribed uniform drying/rewetting with .25m/s drift. Not a SWE validation case; volume change is externally prescribed, no momentum-accuracy claim.',dry_intervals=[[480,520],[1080,1120],[1680,1720]],maximum_depth=2.2,speed=.25,sha256=sha(path)))
    return cases

def episode(job):
    c,method,collect=job;guard();w=np.load(c['field']['path'],mmap_mode='r')
    e=SafeFlowEnv(w,c['scenario'],method) if method in ('coverage100','flow_coverage') else CountCoupledEnv(w,c['scenario'],MODELS[method],.25)
    hist=FineHistory();xx=[];yy=[];minimum=float('inf');checks=0
    if not collect:
        def forbid(*args,**kwargs):raise AssertionError('Hidden counts requested by deployable policy')
        e.sector_counts=forbid
    while True:
        if e.t%120==0:guard()
        if collect and e.t%30==0:xx.append(hist.get(e));yy.append(e.sector_counts(100).astype(np.float32))
        if e.t in (0,600):
            other=copy.copy(e);other.person=np.full_like(e.person,999);other.status=np.full_like(e.status,3);other.water=None;other.fall=e.fall+100;other.deadline=e.deadline+100
            np.testing.assert_array_equal(public_features(e),public_features(other));checks+=1
        p=e.pos.copy();done=e.step(e.choices()[1])[2];minimum=min(minimum,float(swept_distances(p,e.pos).min()));assert minimum>=5-1e-8
        if done:break
    pp=[dict(id=j,fall=int(e.fall[j]),deadline=int(e.deadline[j]),first_seen=float(e.first_seen[j]),supplied_at=float(e.delivered_at[j]),status=int(e.status[j]),refloats=int(e.refloat_counts[j])) for j in range(e.cfg.people)]
    for p in pp:
        if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
    assert np.isfinite(e.pos).all() and np.isfinite(e.person).all() and (e.status>=2).all()
    row=dict(case=c['id'],parent=c['field']['parent'],family=c.get('family','calibration'),method=method,supplied=int((e.status==2).sum()),people=e.cfg.people,p95_scene=bool((e.status==2).sum()>=.95*e.cfg.people),all_scene=bool((e.status==2).all()),min_separation=minimum,refloats=int(e.refloat_counts.sum()),persons=pp,hidden_checks=checks)
    if collect:
        path=OUT/f'cal-{c["id"]}.npz';np.savez_compressed(path,x=np.stack(xx),y=np.stack(yy));row['data']=str(path)
    return row
