"""Global family-weighted selection on development data; no confirmation claims."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from hyra_validation.artifacts import new_run,write_json
from hyra_validation.metric_dataset import MetricFamilyData
from hyra_validation.metric_candidates import MetricCandidates
from hyra_validation.metric_vector import metric_vector_scores
from hyra_validation.metric_compiled import CompiledMetricCases
from hyra_validation.local_service import load_design
from hyra_validation.lead import TARGETS


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',type=Path);parser.add_argument('--split',choices=['train','validation'],default='validation')
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--count',type=int,default=12);parser.add_argument('--batch-size',type=int,default=12)
    args=parser.parse_args()
    modelrun=args.models or Path(json.loads((ROOT/'results/latest_metric_model_training.json').read_text(encoding='utf-8'))['run'])
    modelconfig=json.loads((modelrun/'config.json').read_text(encoding='utf-8'))
    if args.split=='train' and args.start<modelconfig['train_count']:raise ValueError('Training diagnostics must hold out whole training families')
    if args.split=='validation' and modelconfig['train_count']!=24:raise ValueError('Complete planned training before formal global validation')
    design=json.loads((ROOT/'configs/metric_development.design.v1.json').read_text(encoding='utf-8'))
    cache=Path(json.loads((ROOT/'results/latest_metric_feature_cache.json').read_text(encoding='utf-8'))['folder'])
    paths=[cache/f'metric_v2_{args.split}_{i:03}.npz' for i in range(args.start,args.start+args.count)]
    for path in paths:assert path.exists() and path.with_suffix('.json').exists(),str(path)
    config,_=load_design(ROOT);registry=json.loads((modelrun/'registry.json').read_text(encoding='utf-8'))
    candidates=MetricCandidates(registry,config['models']);nmodel=len(candidates.specs)
    run=new_run('metric_model_development_selection',dict(model_run=str(modelrun),cache=str(cache),split=args.split,
        family_start=args.start,family_count=args.count,primary_delay_s=0,primary_population='frozen_nominal_critical',
        rank='family_macro_minimum_gain_over_nominal_across_both_reference_meshes',confirmation_claim=False))
    for rel in json.loads((run/'provenance.json').read_text(encoding='utf-8'))['files']:
        target=run/'frozen_source'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/rel).read_bytes())
    write_json(run/'candidates.json',candidates.specs);print('RUN '+str(run),flush=True);allstats=[]
    for path in paths:
        receipt=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'));assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['cache_sha256']
        data=MetricFamilyData(receipt['manifest_path'],design)
        with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        stats={str(mesh):dict(total=0.,services=np.zeros(nmodel),window_error=np.zeros(nmodel),mae_6_12=np.zeros(nmodel),oracle_services=0.,blocks=0) for mesh in (80,40)}
        truthcache={};tasks={anchor:data.cases(anchor) for anchor in design['anchors_s']};compiled={}
        for anchor,task in tasks.items():
            selected=[task['cases'][i] for i in np.flatnonzero(task['critical'])]
            if selected:compiled[anchor]=CompiledMetricCases(selected,task['row_times'],task['row_targets'])
        for block,m in enumerate(a['metadata']):
            mesh,anchor,condition,center,phase,noise,delay,replicate,weight=m
            if delay!=0:continue
            mesh=int(mesh);anchor=int(anchor);task=tasks[anchor];indices=np.flatnonzero(task['critical'])
            if not len(indices):continue
            cases=[task['cases'][i] for i in indices];cachekey=(mesh,anchor,int(condition),int(center))
            if cachekey not in truthcache:
                truth=a['truth'][block].reshape(72,26)
                score=compiled[anchor].scores(truth)
                lookup={(float(t),int(r)):i for i,(t,r) in enumerate(zip(task['row_times'],task['row_targets']))}
                pairrows=np.array([[lookup[float(c['now_s']),r] for r in c['targets']] for c in cases])
                truthcache[cachekey]=(truth,score,pairrows)
            truth,actual,pairrows=truthcache[cachekey];s=stats[str(mesh)]
            s['total']+=weight*len(cases);s['oracle_services']+=weight*actual['count'][0].max(axis=1).sum();s['blocks']+=1
            public=candidates.prepare(a['common'][block],a['history'][block],a['physical'][block],a['global_component'][block],a['event_probability'][block])
            for begin in range(0,nmodel,args.batch_size):
                specs=candidates.specs[begin:begin+args.batch_size]
                predictions=np.array([a['base'][block]+candidates.predict(spec,public) for spec in specs]).reshape(len(specs),72,26)
                score=compiled[anchor].scores(predictions)
                chosen=score['chosen'];services=actual['count'][0][np.arange(len(cases))[None,:],chosen]
                error=(score['common_maxima']<=.8)!=(actual['common_maxima']<=.8)
                lead=(abs(predictions[:,:,3]-truth[None,:,3])+abs(predictions[:,:,6]-truth[None,:,6]))/2
                mae=lead[:,pairrows].mean(axis=-1)
                sl=slice(begin,begin+len(specs));s['services'][sl]+=weight*services.sum(axis=1)
                s['window_error'][sl]+=weight*error.mean(axis=(2,3)).sum(axis=1);s['mae_6_12'][sl]+=weight*mae.sum(axis=1)
            if s['blocks']%16==0:print(f'EVAL {path.stem} mesh={mesh} blocks={s["blocks"]}',flush=True)
        result=dict(family=path.stem,split=args.split,meshes={})
        for mesh,s in stats.items():
            den=s['total'];result['meshes'][mesh]={k:((v/den).tolist() if isinstance(v,np.ndarray) else v/den) if den else None for k,v in s.items() if k not in ('total','blocks')}
            result['meshes'][mesh].update(total_weighted_cases=den,blocks=s['blocks'],active=den>0)
        write_json(run/(path.stem+'.json'),result);allstats.append(result);print('FAMILY '+path.stem,flush=True)
    active=[f for f in allstats if all(m['active'] for m in f['meshes'].values())];selection=[]
    for i,spec in enumerate(candidates.specs):
        gains=[min(f['meshes'][mesh]['services'][i]-f['meshes'][mesh]['services'][0] for mesh in ('80','40')) for f in active]
        selection.append(dict(name=spec['name'],group=spec['group'],robust_gain_over_nominal=float(np.mean(gains)) if gains else None))
    winners={}
    if active:
        for group in ('P0','P1','P2','P3','P4','P5'):
            eligible=[s for s in selection if s['group']==group];winners[group]=max(eligible,key=lambda s:(s['robust_gain_over_nominal'],s['name']))
        winners['B_star']=max([s for s in selection if s['group'] in ('P0','P1','P2','P3')],key=lambda s:(s['robust_gain_over_nominal'],s['name']))
    result=dict(completed=True,independent_families=len(allstats),active_families=len(active),coverage=len(active)/len(allstats),
        split=args.split,training_diagnostic=args.split=='train',confirmation_claim=False,winners=winners,candidate_selection=selection)
    write_json(run/'selection.json',result);write_json(ROOT/'results/latest_metric_development_selection.json',dict(run=str(run),**{k:result[k] for k in ('completed','training_diagnostic','independent_families','active_families','winners')}))
    print(json.dumps({k:result[k] for k in ('independent_families','active_families','winners')},ensure_ascii=True),flush=True)


if __name__=='__main__':main()
