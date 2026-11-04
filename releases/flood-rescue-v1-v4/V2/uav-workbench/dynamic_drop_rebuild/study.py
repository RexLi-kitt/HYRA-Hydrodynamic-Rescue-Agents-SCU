from .learning import *
ALPHAS=(0,.25,.5,1.)
def seeds(arm):
    return SEEDS if arm.startswith('T') else (11,)
def case(job):
    stage,p,n,r=job;directory=OUT/stage;directory.mkdir(exist_ok=True);file=directory/f'{p}-{n}-{r}.json'
    if file.exists():return
    b,_,_=run(p,n,r)
    rows=[dict(**b,alpha=0,seed=0)]
    choice={} if stage=='calibration' else json.loads((OUT/'selection.json').read_text())['alpha']
    for arm in ARMS:
        for seed in seeds(arm):
            model=predictor(arm,seed)
            alphas=ALPHAS if not choice else sorted(set([.5,choice[arm]]))
            for alpha in alphas:
                if alpha==0:result=dict(b,row=dict(b['row'],arm=arm))
                else:result,_,_=run(p,n,r,arm=arm,predictor=model,alpha=alpha)
                rows.append(dict(**result,alpha=alpha,seed=seed))
    assert len({r['row']['scenario_hash'] for r in rows})==1
    maps.save_json(file,rows)
def batch(stage,jobs):
    with ProcessPoolExecutor(4) as pool:list(pool.map(case,[(stage,*j) for j in jobs]))
def records(stage):
    return [r for f in sorted((OUT/stage).glob('*.json')) for r in json.loads(f.read_text())]
def choose():
    rr=records('calibration');alpha={};scores={}
    for a in ARMS:
        means=[np.mean([r['row']['saved'] for r in rr if r['row']['arm']==a and r['alpha']==v]) for v in ALPHAS]
        idx=int(np.argmax(means));alpha[a]=ALPHAS[idx];scores[a]=means[idx]
    winner=max(ARMS,key=lambda arm:scores[arm])
    maps.save_json(OUT/'selection.json',dict(alpha=alpha,calibration_scores=scores,winner=winner))
def eval_stages():
    batch('calibration',[(p,n,60) for p in range(21013,21017) for n in range(4)])
    choose()
    batch('development',[(p,n,r) for p in range(31001,31007) for r in (60,40,100) for n in range(2 if r==60 else 1)])
def summarize(stage):
    choice=json.loads((OUT/'selection.json').read_text());alpha=choice['alpha'];winner=choice['winner']
    rr=records(stage)
    rows=[];people=[];matrix=[];delta=[]
    for c in rr:
        r=c['row'];key=dict(**r,seed=c['seed'],alpha=c['alpha'],selected_alpha=c['alpha']==alpha.get(r['arm'],0))
        rows.append(key)
        if key['selected_alpha']:people.extend(dict(parent=r['parent'],layout=r['layout'],radius=r['radius'],arm=r['arm'],seed=c['seed'],**v) for v in c['persons'])
    write_rows(OUT/(stage+'-runs.csv'),rows);write_rows(OUT/(stage+'-people.csv'),people)
    for rad in (40,60,100):
        for mode in ('chosen','fixed_half'):
            pp={}
            for a in ('B0',*ARMS):
                sub=[v for v in rows if v['arm']==a and v['radius']==rad and (v['selected_alpha'] if mode=='chosen' else v['alpha']==(.5 if a!='B0' else 0))]
                ids=sorted({v['parent'] for v in sub})
                pp[a]=np.array([np.mean([r['saved'] for r in sub if r['parent']==p]) for p in ids])
                matrix.append(dict(radius=rad,mode=mode,arm=a,layouts=len({(r['layout'],r['parent']) for r in sub}),seed_runs=len(sub),**{k:float(np.mean([r[k] for r in sub])) for k in ('saved','found','near','all40','ms','conflict_stops')}))
            for a,b in [(winner,'B0'),('LG','L0'),('TG','T0'),('TG','LG'),('T0','L0')]:
                diff=pp[a]-pp[b];rng=np.random.default_rng(20261103);ci=np.quantile(diff[rng.integers(len(diff),size=(10000,len(diff)))].mean(1),[.025,.975])
                delta.append(dict(radius=rad,mode=mode,a=a,b=b,people=float(diff.mean()),positive_parents=int((diff>0).sum()),lo=float(ci[0]),hi=float(ci[1]),parents=len(diff)))
    write_rows(OUT/(stage+'-summary.csv'),matrix);write_rows(OUT/(stage+'-paired.csv'),delta)
    primary=next(v for v in delta if v['radius']==60 and v['mode']=='chosen' and v['b']=='B0')
    passed=primary['people']>0 and primary['positive_parents']>=4 if stage=='development' else primary['lo']>0
    maps.save_json(OUT/(stage+'-gate.json'),dict(passed=bool(passed),**primary))
    return passed
def seal():
    return dict(choice=json.loads((OUT/'selection.json').read_text()),models={p.name:maps.digest(p) for p in OUT.glob('*.pt')},code={p.name:maps.digest(p) for p in Path(__file__).parent.glob('*.py')},test_parents=[181001,181050])
def new_map(p):
    maps.DATA=OUT/'new-water';return maps.make_map(p)
def test():
    assert json.loads((OUT/'development-gate.json').read_text())['passed'] and json.loads((OUT/'seal.json').read_text())==seal()
    with ProcessPoolExecutor(3) as pool:manifest=list(pool.map(new_map,range(181001,181051)))
    maps.save_json(OUT/'test-water.json',manifest)
    batch('test',[(p,n,r) for p in range(181001,181051) for n in range(2) for r in (60,40,100)])
    summarize('test')
if __name__=='__main__':
    eval_stages()
    if summarize('development'):
        maps.save_json(OUT/'seal.json',seal());test()
