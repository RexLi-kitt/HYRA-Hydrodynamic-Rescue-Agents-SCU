"""Fixed E1 environment, paired E1/E0 arrived inputs; replay actual service yield."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from hyra_validation.artifacts import write_json
from hyra_validation.metric_dataset import MetricFamilyData
from hyra_validation.metric_candidates import MetricCandidates
from hyra_validation.metric_compiled import CompiledMetricCases
from hyra_validation.local_service import load_design


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--selection',type=Path,required=True);args=parser.parse_args();run=args.selection.resolve()
    selection=read(run/'selection.json');config=read(run/'config.json');cache=Path(config['cache']);all_specs=read(run/'candidates.json')
    winner_names={selection['winners'][g]['name'] for g in ('P4','P5','B_star')}|{'P0_nominal'}
    selected_p4=selection['winners']['P4']['name'];winner_names.add(selected_p4.replace('P4_','P5_',1))
    specs=[s for s in all_specs if s['name'] in winner_names]
    ordinary,_=load_design(ROOT);predictor=MetricCandidates([],ordinary['models']);design=read(ROOT/'configs/metric_development.design.v1.json')
    families=[]
    for index in range(config['family_start'],config['family_start']+config['family_count']):
        path=cache/f'metric_v2_{config["split"]}_{index:03}.npz';receipt=read(path.with_suffix('.json'))
        assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['cache_sha256'];data=MetricFamilyData(receipt['manifest_path'],design)
        with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
        E0={tuple(m[[0,1,4,5,6,7]]):i for i,m in enumerate(a['metadata']) if m[2]==0};compiled={};e0pred={};stats={}
        for anchor in design['anchors_s']:
            task=data.cases(anchor);cases=[c for i,c in enumerate(task['cases']) if task['critical'][i]]
            if cases:compiled[anchor]=(CompiledMetricCases(cases,task['row_times'],task['row_targets']),np.array([c['decision_offset_s'] for c in cases]))
        for block,m in enumerate(a['metadata']):
            mesh,anchor,condition,center,phase,noise,delay,replicate,weight=m
            if condition!=1 or delay!=0 or int(anchor) not in compiled:continue
            pair=E0[tuple(m[[0,1,4,5,6,7]])];np.testing.assert_array_equal(a['common'][block],a['common'][pair])
            scorer,offsets=compiled[int(anchor)];truth=scorer.scores(a['truth'][block].reshape(72,26));actual=truth['count'][0]
            public=predictor.prepare(a['common'][block],a['history'][block],a['physical'][block],a['global_component'][block],a['event_probability'][block])
            if pair not in e0pred:
                null_public=predictor.prepare(a['common'][pair],a['history'][pair],a['physical'][pair],a['global_component'][pair],a['event_probability'][pair])
                e0pred[pair]={s['name']:scorer.scores((a['base'][pair]+predictor.predict(s,null_public)).reshape(72,26))['chosen'][0] for s in specs}
            for spec in specs:
                name=spec['name'];chosen=scorer.scores((a['base'][block]+predictor.predict(spec,public)).reshape(72,26))['chosen'][0]
                null=e0pred[pair][name];rows=np.arange(len(chosen));gain=actual[rows,chosen]-actual[rows,null];changed=chosen!=null
                for offset in (None,*design['decision_offsets_s']):
                    take=np.ones(len(chosen),bool) if offset is None else offsets==offset
                    if not take.any():continue
                    key=(int(mesh),name,str(offset));s=stats.setdefault(key,dict(total=0.,gain=0.,changed=0.))
                    s['total']+=weight*int(take.sum());s['gain']+=weight*float(gain[take].sum());s['changed']+=weight*int(changed[take].sum())
                    if name=='P0_nominal':assert not changed.any() and np.all(gain==0)
                    if offset==0:assert np.all(gain[take]==0), 'Action effect before any pulse-specific information'
        rows=[]
        for (mesh,name,offset),s in stats.items():rows.append(dict(mesh=mesh,name=name,decision_offset_s=None if offset=='None' else int(offset),
            completed_service_gain=s['gain']/s['total'],action_change_fraction=s['changed']/s['total'],weighted_cases=s['total']))
        result=dict(family=path.stem,rows=rows);families.append(result);write_json(run/(path.stem+'_information_action.json'),result)
        print('PAIRED_INFORMATION '+path.stem,flush=True)
    totals=[]
    for spec in specs:
        for offset in (None,*design['decision_offsets_s']):
            values=[];changes=[]
            for family in families:
                rows=[r for r in family['rows'] if r['name']==spec['name'] and r['decision_offset_s']==offset]
                if len(rows)==2:values.append(min(r['completed_service_gain'] for r in rows));changes.append(min(r['action_change_fraction'] for r in rows))
            if values:totals.append(dict(name=spec['name'],decision_offset_s=offset,independent_families=len(values),
                robust_completed_service_gain=float(np.mean(values)),minimum_mesh_action_change_fraction=float(np.mean(changes))))
    result=dict(completed=True,intervention='Same E1 truth; replace arrived inputs by paired E0 inputs with identical noise; recompute predictions and initial order; same actual six-second abort feedback',
        primary_delay_s=0,E1_condition_only=True,training_diagnostic=selection['training_diagnostic'],confirmation_claim=False,summary=totals)
    write_json(run/'information_action_check.json',result);print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
