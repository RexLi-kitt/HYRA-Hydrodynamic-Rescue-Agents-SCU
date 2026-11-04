import json
from copy import copy
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch
import numpy as np
from dynamic_drop_rebuild import sim, planner
from dynamic_drop_rebuild.learning import predictor, SEEDS
from release_utils import fingerprints
from patrol_aimpoints import policy
from release_io import read,save
from .scenario import make,OUT,maps,ORIGINAL_DATA


def decide(m,p,t,age,wet,dm,arm,meta):
    a,end,zone=planner.plan(m,p,t,age,wet,dm,.5)
    announcements=[(at,zs) for at,zs in meta['reports'] if at<=t]
    if not announcements:
        end,_=policy.select('K1',m.stamp,p,t,zone,end,40)
        return a,end,zone
    bonus=np.zeros(100); earliest=np.full(100,999); reported=np.zeros(100,bool)
    for at,zs in announcements:
        for z in zs+(meta['corridor'] if at==0 else []):
            bonus[z]=max(bonus[z],np.mean(m.stamp[policy.CELLS[z]]<at))
            reported[z]=True;earliest[z]=min(earliest[z],at)
    zone[:]=-1
    for k in range(4):
        i=(t//20+k)%4
        if a[i]>=0:continue
        dist=np.linalg.norm(planner.CENTERS-p[i],axis=1)
        if arm=='H':
            ids=np.flatnonzero(reported)
            z=min(ids,key=lambda j:(-bonus[j],earliest[j],dist[j],j))
        else:
            valid=dist>=12
            raw=(.12+age)*(1+3*wet)/(1+dist/180)
            used=zone[zone>=0]
            if len(used):raw/=1+2*np.exp(-planner.DIST[:,used]**2).sum(1)
            raw[~valid]=0
            if np.ptp(dm)>1e-8:
                raw=.5*planner.norms(raw)+.5*planner.norms(dm)/(1+dist/180)
            score=np.full(100,-np.inf)
            score[valid]=planner.norms(raw[valid])+.5*bonus[valid]
            z=score.argmax()
        zone[i]=z
        delta=planner.CENTERS[z]-p[i]
        end[i]=p[i]+delta*min(1,50/max(dist[z],1e-10))
    end,_=policy.select('K1',m.stamp,p,t,zone,end,40)
    return a,end,zone


def run(data,arm,seed,switch=None,probe=False):
    scene,f,meta,tags=data
    public={k:meta[k] for k in ('reports','corridor')}
    steps=[];div=[]
    def act(m,p,t,age,wet,dm,alpha):
        if probe or t==switch:
            alt=predictor('T0',seed)(m.sequence())
            other=decide(copy(m),p,t,age,wet,alt,'T',public)
        a,e,z=decide(m,p,t,age,wet,dm,arm,public)
        if probe and np.max(abs(other[1]-e))>1e-9:
            div.append(dict(t=t,tzones=other[2].tolist(),gzones=z.tolist(),te=other[1].tolist(),ge=e.tolist()))
        if t==switch:
            a,e,z=other;m.owner=a;m.last_zone=z
        steps.append(dict(t=t,p=p.tolist(),a=a.tolist(),end=e.tolist(),z=z.tolist()))
        return a,e,z
    model={'H':'B0','T':'T0','G':'TG'}[arm]
    with ExitStack() as stack:
        for mod,key,value in [(sim,'T',300),(policy,'T',300),(sim,'fields',lambda _:f),(sim,'scenario',lambda *args:scene),(sim,'plan',act)]:
            stack.enter_context(patch.object(mod,key,value))
        rec=sim.run(meta['parent'],meta['layout'],40,arm=model,predictor=None if arm=='H' else predictor(model,seed),alpha=.5)[0]
    rec.pop('choices');rec['row'].update(arm=arm,seed=seed)
    return dict(**rec,trace=steps,div=div)


def development(job):
    p,n=job;data=make(p,n)
    scene,f,meta,tags=data
    b,xy,st,*_=scene
    assert all((st[:v,i]==0).all() and st[v,i]!=0 for i,v in enumerate(b))
    assert all(np.all(xy[v:,i]==xy[v,i]) for i,v in enumerate(b) if i>=16)
    assert all((xy[:,i][st[:,i]==3]==xy[:,i][st[:,i]==3][0]).all() for i in range(16) if (st[:,i]==3).any())
    base=run(data,'G',11)
    changed=xy.copy();changed[0,b>0]=345
    f2=f.copy();f2[1:]=0
    rec=run(((b,changed,st,scene[3],0),f2,meta,tags),'G',11)
    assert rec['trace'][0]==base['trace'][0]
    # At 60s at least one still-pending person in each of two separate reported blocks.
    pending=set()
    for i,v in enumerate(base['persons']):
        if b[i]<=60<b[i]+240 and st[60,i]!=4 and (v['saved']<0 or v['saved']>60):
            pending.add(tuple(meta['reports'][0 if i<16 else 1+(i-16)//8][1]))
    return dict(parent=p,layout=n,cross=sum(v['crossed'] and v['displacement']>=20 for v in tags),
                snag=sum(v['snag'] and v['displacement']>0 for v in tags),concurrent=len(pending)>=2,hash=scene[3])


def map_job(p):
    maps.DATA=OUT/'water';return maps.make_map(p)


def case(job):
    p,n=job;path=OUT/'cases'/f'{p}-{n}.gz'
    if path.exists():return
    maps.DATA=OUT/'water'
    data=make(p,n);scene,_,meta,tags=data
    records=[run(data,'H',0)];branches=[]
    for seed in SEEDS:
        records.append(run(data,'T',seed))
        g=run(data,'G',seed,probe=True);records.append(g)
        if g['div']:
            t=g['div'][0]['t'];branch=run(data,'G',seed,switch=t)
            assert g['trace'][:t//5]==branch['trace'][:t//5] and branch['trace'][t//5]['end']==g['div'][0]['te']
            branches.append(dict(seed=seed,t=t,alternative=branch))
    assert len({r['row']['scenario_hash'] for r in records})==1
    save(path,dict(meta=meta,tags=tags,records=records,branches=branches,hash=scene[3]))


def main():
    from .report import report
    OUT.mkdir(exist_ok=True)
    code=fingerprints()
    code.update({p.name:maps.digest(p) for p in Path(__file__).parent.glob('*.py')})
    seal=OUT/'seal.json'
    if not seal.exists():
        from .checks import checks
        checks()
        with ProcessPoolExecutor(6) as pool:dev=list(pool.map(development,[(p,n) for p in range(31001,31007) for n in range(4)]))
        gates={k:len({v['parent'] for v in dev if v[k]}) for k in ('cross','snag','concurrent')}
        maps.save_json(OUT/'development.json',dict(layouts=dev,gates=gates))
        if min(gates.values())<4:
            print('Development gate stopped:',gates);return
        maps.save_json(seal,code)
    assert json.loads(seal.read_text())==code
    with ProcessPoolExecutor(4) as pool:manifest=list(pool.map(map_job,range(184001,184101)))
    maps.save_json(OUT/'maps.json',manifest)
    with ProcessPoolExecutor(8) as pool:list(pool.map(case,[(p,n) for p in range(184001,184101) for n in range(2)]))
    report()

