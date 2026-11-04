from pathlib import Path
import collections, hashlib, json, os, subprocess, time
import numpy as np

OUT=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
R=OUT/'round77'
def write(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def split(obs,mod):
    return 'both_unresolved' if obs and mod else 'observation_only' if obs else 'model_only' if mod else 'neither'

def unknown_audit():
    source=OUT/'round76/qualitative_request/result.json'
    a=json.loads(source.read_text('utf-8'))
    result={'source':str(source),'sha256':digest(source),'old_scores_unchanged':True,'okushiri':{},'merewether':{}}
    rows=[]
    for row in a['okushiri']['all_trend_rows']:
        if row['comparison']=='unknown':
            obs=row['observed']['label']=='unresolved';mod=row['model']['label']=='unresolved'
            assert obs or mod
            rows.append(dict(row,unknown_cause=split(obs,mod)))
    result['okushiri']={'counts':dict(collections.Counter(r['unknown_cause'] for r in rows)), 'rows':rows,
        'original_trend_summary':a['okushiri']['trend'], 'original_opposite_rows':[x for x in a['okushiri']['all_trend_rows'] if x['comparison']=='inconsistent']}
    assert len(rows)==34
    for name,v in a['merewether'].items():
        dirs=[];pairs=[]
        for row in v['all_direction_rows']:
            if row['comparison']=='unknown':
                obs=row['observation_speed']<.05;mod=row['model_speed']<.05
                assert obs or mod
                dirs.append(dict(row,unknown_cause=split(obs,mod),known_model_flow_failure=row['id']=='19-J'))
        for row in v['all_relative_speed_pairs']:
            if row['comparison']=='unknown':
                obs=abs(row['observed_difference'])<=1e-12;mod=abs(row['model_difference'])<=1e-12
                assert obs or mod
                pairs.append(dict(row,unknown_cause=split(obs,mod)))
        result['merewether'][name]={'direction_counts':dict(collections.Counter(x['unknown_cause'] for x in dirs)),
            'direction_rows':dirs,'speed_pair_counts':dict(collections.Counter(x['unknown_cause'] for x in pairs)),
            'speed_pair_rows':pairs,'original_direction_summary':v['direction'],'original_speed_summary':v['relative_speed']}
        assert len(dirs)==v['direction']['unknown'] and len(pairs)==v['relative_speed']['unknown']
    write(R/'unknown-causes.json',result)
    print(json.dumps({k:v['counts'] for k,v in result.items() if isinstance(v,dict) and 'counts' in v},ensure_ascii=False))
    for k,v in result['merewether'].items(): print(k,v['direction_counts'],v['speed_pair_counts'])

def make_case(case,name):
    case.mkdir(parents=True,exist_ok=False)
    nx,ny=20,12
    header=f'ncols {nx}\nnrows {ny}\nxllcorner 0\nyllcorner 0\ncellsize 1\nNODATA_value -9999\n'
    def raster(n,value):
        (case/n).write_text(header+(' '.join([str(value)]*nx)+'\n')*ny)
    initial=.2 if name=='inflow_west' else 1.
    for n in ['bed.asc','bed.asc1x','bed.asc1y','water.asc1x','water.asc1y']:raster(n,0)
    raster('water.asc',initial)
    side=name[-1] if name.startswith('drain_') else 'E'
    bci=f'{side} 0 {ny if side in ["W","E"] else nx} HFIX .2\n'
    if name=='inflow_west':bci='W 0 12 HFIX 1.5\n'
    if name=='closed':bci=''
    if name=='point_and_outlet':bci+='P 10.5 6.5 QFIX .01\n'
    (case/'input.bci').write_text(bci)
    (case/'case.par').write_text('DEMfile bed.asc\nstartfile water.asc\nbcifile input.bci\ndirroot results\nresroot check\ndg2\nsim_time .5\ninitial_tstep .02\nCFL .3\nmassint .1\nsaveint .5\nfpfric 0\noutput_precision 15\n')
    return initial*nx*ny

def run_case(exe,case,initial_volume):
    start=time.perf_counter()
    with (case/'run.log').open('w') as log:
        p=subprocess.run([str(exe),'-v','case.par'],cwd=case,env=dict(os.environ,OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,timeout=30)
    result={'exit_code':p.returncode,'wall_s':time.perf_counter()-start,'exe_sha256':digest(exe)}
    if p.returncode==0:
        h=np.loadtxt(case/'results/check-0000.wd',skiprows=6)
        m=np.loadtxt(case/'results/check.mass',skiprows=1,ndmin=2)
        steps=np.loadtxt(case/'accepted_timesteps.txt',ndmin=2)
        result.update(initial_volume_m3=initial_volume,final_volume_m3=float(h.sum()),min_depth_m=float(h.min()),
            finite_nonnegative=bool(np.isfinite(h).all() and (h>=0).all()),steps=len(steps),end_s=float(steps[-1].sum()),
            total_ledger_residual_m3=float(m[:,10].sum()),max_interval_residual_m3=float(np.abs(m[:,10]).max()),negative_trace_present=(case/'negative-depth.jsonl').exists())
    write(case/'result.json',result)
    return result

if __name__=='__main__':
    R.mkdir(exist_ok=False)
    write(R/'protocol.json',{
      'date':'2026-10-30','baseline':'round76 isolated CPU64 DG2 integrated-source executable',
      'unknown_task':'Split existing unknowns by observation/model/both without changing old labels, denominators, scope or claiming improvement. 19-J remains a known flow/wetting failure, not a missing-observation excuse.',
      'hypothesis':'DG2 updates H with half RK1+half RK2 mass flux, but existing boundary ledger records only RK2; reproduce before changing code.',
      'cases':['closed','drain_E','drain_W','drain_N','drain_S','inflow_west','point_and_outlet'],
      'fixed_config':'20x12 cells, dx=1m, flatbed, no friction, initial H1m (inflow:0.2m), HFIX .2m (inflow:1.5m), .5s simulation, dtmax.02, massint.1, CPU1; point adds .01m2/s.',
      'criteria':['Reproduce nonzero exchange with ledger residual >1e-8m3 before fix','Candidate fixes ledger within1e-8m3 for all cases','Candidate preserves all emitted hydrodynamic fields and accepted timesteps bitwise','All final depths finite/nonnegative and no negative-RK trace','Closed case unchanged; mixed-source/outlet counts point exactly once'],
      'change_if_reproduced':'Only weight boundary rate accounting half at each RK stage; do not change physical flux, source, H or momentum update, boundaries or time selection.',
      'cost':'One small sequential suite, at most one isolated CPU compile, then same-suite single-factor comparison; no60s/800s rerun or complete field run.',
      'rejection':'Do not relax tolerances, alter observations or adopt production. Stop candidate on field/time differences, bad depth or residual. Results only support internal bookkeeping, not field fidelity.'})
    unknown_audit()
    exe=OUT/'round76/build-integrated-cpu64/Release/lisflood.exe'
    results={}
    for name in ['closed','drain_E','drain_W','drain_N','drain_S','inflow_west','point_and_outlet']:
        case=R/'baseline'/name;iv=make_case(case,name)
        results[name]=run_case(exe,case,iv)
    write(R/'baseline-results.json',results)
    print(json.dumps(results,ensure_ascii=False),flush=True)
