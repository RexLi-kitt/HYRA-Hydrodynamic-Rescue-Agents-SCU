import argparse,csv,hashlib,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
import numpy as np
from flood_dqn import maps
from flood_dqn.env import Environment,layout
from release_io import save,read

OUT=maps.ROOT/'generated/v1-rescue95-v1'
DEV=range(31001,31007)
TEST=range(189001,189051)

def public(e):
    return SimpleNamespace(p=e.p,heading=e.heading,step_no=e.step_no,stamps=e.stamps,water=e.water,
        covered=e.covered,found=e.found,rescued=e.rescued,owner=e.owner,
        targets=np.where(e.found[:,None],e.targets,np.nan),cells=np.where(e.found,e.cells,-1))


def evaluate(job):
    stage,arm,p,n=job
    from .rescue import Rescue
    if arm=='sweep':
        from .sweep import Sweep
        planner=Sweep()
    elif arm=='search':
        from .search import Search
        planner=Search()
    else:
        from importlib import import_module
        name={'teamroute':'TeamRoute'}.get(arm,arm.capitalize())
        planner=getattr(import_module('.'+arm,__package__),name)()
    path=OUT/stage/arm/f'{p}-{n}.gz'
    ps=(2000000 if stage=='dev' else 31000000)+p*4+n
    location=(maps.DATA if stage=='dev' else OUT/'water')/'maps'/f'{p}.npz'
    with np.load(location) as f:e=Environment(f['fields'],f['safe'],layout(ps))
    controller=Rescue();trace=[];events=[]
    found=np.where(e.found,0,-1);saved=np.full(20,-1)
    while True:
        state=public(e)
        fixed=controller.actions(state)
        actions=np.asarray(planner.actions(state,fixed),int)
        assert np.all(actions[fixed>=0]==fixed[fixed>=0]) and np.all((actions>=0)&(actions<9))
        _,done=e.step(actions)
        trace.append(actions.tolist())
        found[(found<0)&e.found]=e.step_no;saved[(saved<0)&e.rescued]=e.step_no
        if done:break
    if hasattr(planner,'close'):planner.close()
    for j,c in enumerate(e.cells):
        seq=e.safe[:,c//100,c%100];ok=seq[1:]&seq[:-1]
        events.append(dict(id=j,cell=int(c),found=int(found[j]),saved=int(saved[j]),possible=bool(ok.any()),
                           window_after_discovery=bool(found[j]>=0 and ok[found[j]:].any())))
    row=dict(parent=p,layout=n,layout_seed=ps,arm=arm,discovered=int(e.found.sum()),rescued=int(e.rescued.sum()),**e.metrics())
    save(path,dict(row=row,people=events,actions=trace))
    return row


def batch(stage,arms,parents,ns):
    jobs=[(stage,a,p,n) for p in parents for n in ns for a in arms]
    with ProcessPoolExecutor(8) as pool:return list(pool.map(evaluate,jobs))


def summarize(rows,stage):
    def write(name,rr):
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rr[0]);w.writeheader();w.writerows(rr)
    people=[]
    for r in rows:
        d=read(OUT/stage/r['arm']/f"{r['parent']}-{r['layout']}.gz")
        people.extend(dict(parent=r['parent'],layout=r['layout'],arm=r['arm'],**v) for v in d['people'])
    write(stage+'-episodes.csv',rows);write(stage+'-people.csv',people)
    result=[]
    for a in sorted({r['arm'] for r in rows}):
        rr=[r for r in rows if r['arm']==a]
        x=np.array([np.mean([r['RCR'] for r in rr if r['parent']==p]) for p in sorted({r['parent'] for r in rr})])
        lo,hi=np.quantile(np.random.default_rng(20261104).choice(x,(10000,len(x))).mean(1),[.025,.975])
        row=dict(arm=a,n=len(rr),rescued=sum(r['rescued'] for r in rr),denominator=20*len(rr),lo=lo,hi=hi)
        for k in ('RCR','ST','ACC','FRT','CRT','coverage','timeout','discovered'):
            v=[r[k] for r in rr if r[k] is not None]
            row[k]=np.mean(v) if v else None
            if k in ('FRT','CRT'):row[k+'_n']=len(v)
        result.append(row)
    write(stage+'-summary.csv',result);print(result,flush=True)
    return result


def generate(p):
    maps.DATA=OUT/'water'
    return maps.make_map(p)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['dev','test'])
    ap.add_argument('arms',nargs='*',default=[])
    args=ap.parse_args();OUT.mkdir(exist_ok=True)
    if args.stage=='dev':
        summarize(batch('dev',args.arms or ['search','sweep'],DEV,range(4)),'dev');return
    with (OUT/'dev-summary.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    best=max(rows,key=lambda r:(float(r['RCR']),float(r['coverage']),-float(r['ACC'])))['arm']
    source=Path(__file__).parent
    files=[*source.glob('*.py'),*source.glob('*.js'),*source.glob('*.npz'),*source.glob('*.java'),Path(maps.__file__),maps.ROOT/'flood_dqn/env.py',maps.TERRAIN/'flood_solver.py',
           maps.ROOT/'coordinated_search_dqn/planner.py',maps.ROOT/'rescue_assignment_dqn/controller.py']
    seal=dict(best=best,parents=[TEST.start,TEST.stop-1],layouts=4,criterion='RCR > .95',files={str(f):maps.digest(f) for f in files})
    path=OUT/'seal.json'
    if path.exists():assert json.loads(path.read_text())==seal
    else:maps.save_json(path,seal)
    with ProcessPoolExecutor(4) as pool:manifest=list(pool.map(generate,TEST))
    maps.save_json(OUT/'maps.json',manifest)
    summarize(batch('test',[best],TEST,range(4)),'test')


if __name__=='__main__':main()

