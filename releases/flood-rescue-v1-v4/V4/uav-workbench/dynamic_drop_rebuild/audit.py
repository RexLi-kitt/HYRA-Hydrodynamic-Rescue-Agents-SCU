import json,sys
from concurrent.futures import ProcessPoolExecutor
from .sim import *
def jobs():
    return [(p,n,r) for p in range(31001,31007) for r in (60,40,100) for n in range(2 if r==60 else 1)]
def worker(job):
    p,n,r=job;out={}
    for re in (False,True):
        for arm in ('B0','oracle'):
            rec,_,_=run(p,n,r,re,arm,alpha=.5)
            out[f'{re}-{arm}']=rec
    maps.save_json(OUT/'audit'/f'{p}-{n}-{r}.json',out)
def audit():
    dest=OUT/'audit';dest.mkdir(exist_ok=True,parents=True)
    with ProcessPoolExecutor(4) as pool:list(pool.map(worker,[j for j in jobs() if not (dest/('%d-%d-%d.json'%j)).exists()]))
    sums={}
    for path in dest.glob('*.json'):
        for k,d in json.loads(path.read_text()).items():
            key=k+'-'+str(d['row']['radius']);sums[key]=sums.get(key,0)+d['row']['saved']
    assert len(list(dest.glob('*.json')))==24
    maps.save_json(OUT/'A-gate.json',dict(totals=sums,passed=sums['True-oracle-60']>sums['True-B0-60']))
if __name__=='__main__':audit()
