"""Reuse completed hydraulic fields for the preregistered all-pairs metric task screen."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from diagnose_numerical_stability import read,load,sha,csv_write
from calibrate_physical_scale import summarize,source_visibility,ScaleFields
from hyra_validation.artifacts import new_run,write_json
from hyra_validation.metric_tasks import metric_cases,metric_scores,choose_metric


def arrays(scores):
    return {name:np.array([[order[name] for order in row] for row in scores])
            for name in ('count','margin','starts','maxima')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path)
    args=parser.parse_args();design=read(ROOT/'configs/metric_task_geometry.design.v1.json')
    parent=Path(design['hydraulic_parent']);assert read(parent/'completion.json')['completed']
    assert read(parent/'independent_audit.json')['passed']
    hydraulic_design=read(parent/'config.json')['design'];registry=read(parent/'field_registry.json')
    run=args.run.resolve() if args.run else new_run('metric_all_pairs_training_screen',dict(
        design=design,hydraulic_parent=str(parent),hydraulic_registry_sha256=sha(parent/'field_registry.json')))
    if args.run:
        for rel,checksum in read(run/'provenance.json')['files'].items():assert sha(ROOT/rel)==checksum,rel
    print('RUN '+str(run),flush=True)
    helper_design=dict(hydraulic_design,domain_lengths_x_m=design['candidate_lengths_x_m'],
                       truth_pulse_amplitudes_m=design['candidate_amplitudes_m'])
    cache={};results=[];geometries={};started=time.perf_counter()
    additional=ScaleFields(run,dict(hydraulic_design,maximum_new_integrations=24))
    source_parent=Path(hydraulic_design['source_family_parent'])
    for family in hydraulic_design['family_ids']:
        meta=read(source_parent/'families'/(family+'.json'))
        for length in design['candidate_lengths_x_m']:
            info=registry[f'{family}_L{length}_n40_nominal_base_a0_y0_mm0']
            assert sha(info['raw_path'])==info['raw_sha256']
            raw=load(info['raw_path'])
            for release in design['additional_actual_forecast_releases_s']:
                index=int(np.flatnonzero(raw['time_s']==release)[0])
                restart=dict(state={k:raw[k][index] for k in ('stage','xmomentum','ymomentum')},
                    source_kind='same_mesh_independent_nominal_base',source_path=info['raw_path'],
                    source_sha256=info['raw_sha256'],state_time_s=release)
                additional.get(meta,length,'nominal_forecast',anchor=release,restart=restart)
    registry.update(additional.registry)
    def field(family,length,kind,anchor=0,center=0,amp=0.):
        key=f'{family}_L{length}_n40_{kind}_a{anchor}_y{center}_mm{round(amp*1000)}'
        if key not in cache:
            info=registry[key];assert sha(info['projected_path'])==info['projected_sha256']
            cache[key]=load(info['projected_path'])
        return cache[key]
    for length in design['candidate_lengths_x_m']:
        cases,coverage=metric_cases(length,0,design['origin_ids'],design['flight_speed_m_per_s'],design['decision_offsets_s'])
        geometries[str(length)]=dict(cases=cases,coverage=coverage)
    write_json(run/'geometry.json',geometries)
    for family in hydraulic_design['family_ids']:
        for length in design['candidate_lengths_x_m']:
            rows=[];strata=[];dumps={};base=field(family,length,'truth_base')
            geometry=geometries[str(length)];cases0=geometry['cases'];case_count=len(cases0)
            for anchor in design['anchors_s']:
                cases=[dict(c,now_s=c['now_s']+anchor) for c in cases0]
                base_scores=arrays(metric_scores(base['time_s'],base['linear'],cases,anchor))
                af=field(family,length,'nominal_forecast',anchor)
                anchor_scores=arrays(metric_scores(af['time_s'],af['linear'],cases,anchor))
                critical=np.min(abs(.8-anchor_scores['maxima']),axis=(1,2))<=.05
                current=[None]*case_count
                for offset in design['decision_offsets_s']:
                    indices=[j for j,c in enumerate(cases) if c['decision_offset_s']==offset]
                    release=int((anchor+offset)//10*10);forecast=field(family,length,'nominal_forecast',release)
                    scored=metric_scores(forecast['time_s'],forecast['linear'],[cases[j] for j in indices],anchor)
                    for j,row in zip(indices,scored):current[j]=row
                chosen=choose_metric(current,cases);ix=np.arange(case_count)
                base_gap=base_scores['count'][ix,1-chosen]-base_scores['count'][ix,chosen]
                prefix=f'a{anchor}'
                dumps[prefix+'_nominal_actions']=chosen;dumps[prefix+'_critical']=critical
                dumps[prefix+'_anchor_nominal_maxima']=anchor_scores['maxima']
                for name,data in base_scores.items():dumps[prefix+'_E0_'+name]=data
                for center in design['physical_centers_y_m']:
                    for amplitude in design['candidate_amplitudes_m']:
                        pulse=field(family,length,'truth_pulse',anchor,center,amplitude)
                        scored=arrays(metric_scores(pulse['time_s'],pulse['linear'],cases,anchor))
                        gap=scored['count'][ix,1-chosen]-scored['count'][ix,chosen]
                        specific=(gap>0)&(base_gap<=0)
                        availability={}
                        for offset in design['decision_offsets_s']:
                            availability[offset]=np.array([[source_visibility(pulse['time_s'],pulse['linear']-base['linear'],
                                anchor+offset,phase,delay) for delay in (0,10)] for phase in (0,.25,.5,.75)])
                        flags=np.array([availability[c['decision_offset_s']] for c in cases]) & specific[:,None,None]
                        pp=prefix+f'_y{center}_mm{round(amplitude*1000)}'
                        for name,data in scored.items():dumps[pp+'_E1_'+name]=data
                        dumps[pp+'_pulse_specific']=specific;dumps[pp+'_visible']=flags
                        for population,keep in [('general',np.ones(case_count,bool)),('critical',critical)]:
                            for condition,evaluated in [('E0',base_scores),('E1',scored)]:
                                actual=evaluated['count'][ix,chosen];oracle=evaluated['count'].max(axis=1)
                                record=dict(family=family,length_x_m=length,amplitude_m=amplitude,anchor_s=anchor,
                                    center_y_m=center,population=population,condition=condition,cases=int(keep.sum()),
                                    available_cases=case_count,nominal_sum=int(actual[keep].sum()),oracle_sum=int(oracle[keep].sum()),
                                    headroom_sum=int((oracle-actual)[keep].sum()),
                                    pulse_specific_gain_sum=int(np.maximum(gap,0)[keep & specific].sum()) if condition=='E1' else 0,
                                    visible_D0_sum=int((flags[:,:,0]*np.maximum(gap,0)[:,None])[keep].sum()) if condition=='E1' else 0,
                                    visible_D10_sum=int((flags[:,:,1]*np.maximum(gap,0)[:,None])[keep].sum()) if condition=='E1' else 0)
                                rows.append(record)
                                for label,mask in [('same_half',np.array([not c['cross_half'] for c in cases])),
                                                   ('cross_half',np.array([c['cross_half'] for c in cases])),
                                                   ('speed5',np.array([c['speed_m_per_s']==5 for c in cases])),
                                                   ('speed10',np.array([c['speed_m_per_s']==10 for c in cases]))]:
                                    selected=keep & mask
                                    strata.append(dict(record,stratum=label,cases=int(selected.sum()),
                                        nominal_sum=int(actual[selected].sum()),oracle_sum=int(oracle[selected].sum()),
                                        headroom_sum=int((oracle-actual)[selected].sum()),
                                        pulse_specific_gain_sum=int(np.maximum(gap,0)[selected & specific].sum()) if condition=='E1' else 0,
                                        visible_D0_sum=int((flags[:,:,0]*np.maximum(gap,0)[:,None])[selected].sum()) if condition=='E1' else 0,
                                        visible_D10_sum=int((flags[:,:,1]*np.maximum(gap,0)[:,None])[selected].sum()) if condition=='E1' else 0))
            path=run/(family+f'_L{length}_scores.npz');np.savez_compressed(path,**dumps)
            result=dict(family=family,length_x_m=length,rows=rows,strata=strata,scores_path=str(path),scores_sha256=sha(path))
            write_json(run/(family+f'_L{length}.json'),result);results.append(result)
            write_json(run/'progress.json',dict(completed=False,completed_contexts=len(results),last_context=family+f'_L{length}'))
            print('CONTEXT_DONE '+family+f' L{length}',flush=True)
    summary=summarize(helper_design,results)
    summary.update(task_design='all66_pairs_metric_flight_fixed5_and10mps_complete12s',new_reference_or_probe_integrations=0,
                   new_nominal_forecast_integrations=len(additional.registry),
                   seconds=time.perf_counter()-started,geometry_coverage={k:v['coverage'] for k,v in geometries.items()})
    write_json(run/'summary.json',summary)
    csv_write(run/'opportunity_table.csv',[r for x in results for r in x['rows']])
    csv_write(run/'strata.csv',[r for x in results for r in x['strata']])
    write_json(run/'completion.json',dict(completed=True,contexts=len(results),new_reference_or_probe_integrations=0,
                                         new_nominal_forecast_integrations=len(additional.registry)))
    write_json(run/'progress.json',dict(completed=True,completed_contexts=len(results)))
    write_json(ROOT/'results/latest_metric_task_screen.json',dict(run=str(run),completed=True,audit_pending=True,
               selected_for_next_stage=summary['selected_for_next_stage']))
    print('SUMMARY '+str(summary),flush=True)


if __name__=='__main__':main()
