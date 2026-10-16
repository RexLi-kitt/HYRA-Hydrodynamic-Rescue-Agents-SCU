"""Verify full-state pulse forks against saved independent full-length integrations."""
from pathlib import Path
import argparse
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from diagnose_numerical_stability import read,load,sha
from hyra_validation.artifacts import new_run,write_json
from hyra_validation.adapters.anuga import simulate
from hyra_validation.scaled_hydraulics import scaled_domain,sampling_xy
from hyra_validation.numerical_stability import point_weights,read_points,boundary_footprint


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path)
    args=parser.parse_args();design=read(ROOT/'configs/hydraulic_fork_equivalence.design.v1.json')
    run=args.run.resolve() if args.run else new_run('hydraulic_state_fork_equivalence',dict(design=design))
    if args.run:
        for rel,checksum in read(run/'provenance.json')['files'].items():assert sha(ROOT/rel)==checksum,rel
    print('RUN '+str(run),flush=True);(run/'forks').mkdir(exist_ok=True)
    rows=[];started=time.perf_counter()
    for resolution in ('coarse','fine'):
        parent=Path(design['scale_parent'] if resolution=='coarse' else design['refinement_parent'])
        registry=read(parent/'field_registry.json');config=read(parent/'config.json')
        source_parent=Path(config['design']['source_family_parent'] if resolution=='coarse' else config['source_family_parent'])
        mesh=design[resolution+'_mesh']
        for family in design[resolution+'_families']:
            meta_path=source_parent/'families'/(family+'.json');meta=read(meta_path);assert meta['split']=='train'
            if resolution=='coarse':base_key=f'{family}_L100_n40_truth_base_a0_y0_mm0'
            else:base_key=f'{family}_n80_dt2_truth_base_a0_y0'
            base_info=registry[base_key];assert sha(base_info['raw_path'])==base_info['raw_sha256']
            base=load(base_info['raw_path']);weights=point_weights(base['centroid_xy_m'],sampling_xy(100))
            for anchor in design[resolution+'_anchors_s']:
                index=int(np.flatnonzero(base['time_s']==anchor)[0])
                initial={name:base[name][index] for name in design['evolved_fields']}
                for center in design['centers_y_m']:
                    for amplitude in design[resolution+'_amplitudes_m']:
                        if resolution=='coarse':key=f'{family}_L100_n40_truth_pulse_a{anchor}_y{center}_mm{round(amplitude*1000)}'
                        else:key=f'{family}_n80_dt2_truth_pulse_a{anchor}_y{center}'
                        info=registry[key];assert sha(info['raw_path'])==info['raw_sha256']
                        reference=load(info['raw_path']);ri=int(np.flatnonzero(reference['time_s']==anchor)[0])
                        for name in design['evolved_fields']:
                            np.testing.assert_allclose(initial[name],reference[name][ri],atol=1e-12,rtol=0)
                        result_path=run/'forks'/(key+'.json');fork_path=result_path.with_suffix('.npz')
                        if result_path.exists():
                            row=read(result_path);assert sha(fork_path)==row['fork_sha256'];rows.append(row);continue
                        domain=scaled_domain(meta['params'],mesh,100,initial=initial,start=anchor,
                            amplitude_m=amplitude,anchor_s=anchor,center_y_m=center)
                        assert boundary_footprint(domain)==[center-10,center+10]
                        end=min(anchor+design['post_fork_horizon_s'],float(reference['time_s'][-1]))
                        clock=time.perf_counter();times,depth,states,_=simulate(domain,end,sampling_xy(100),2)
                        arrays={name:np.array([state[name] for state in states]) for name in design['evolved_fields']}
                        fork_seconds=time.perf_counter()-clock
                        indices=np.searchsorted(reference['time_s'],times)
                        np.testing.assert_array_equal(reference['time_s'][indices],times)
                        errors={name:float(np.max(abs(arrays[name]-reference[name][indices]))) for name in arrays}
                        sampled=read_points(arrays['stage'],base['bed_m'],weights)['linear']
                        old=read_points(reference['stage'][indices],reference['bed_m'],weights)['linear']
                        depth_error=float(np.max(abs(sampled-old)))
                        np.savez_compressed(fork_path,time_s=times,**arrays,linear_depth_m=sampled)
                        row=dict(key=key,mesh=mesh,family=family,anchor_s=anchor,center_y_m=center,amplitude_m=amplitude,
                            restart_source_kind='reference_base_environment_only_not_forecast',
                            restart_source_path=base_info['raw_path'],restart_source_sha256=base_info['raw_sha256'],
                            reference_path=info['raw_path'],reference_sha256=info['raw_sha256'],
                            metadata_path=str(meta_path),metadata_sha256=sha(meta_path),fork_path=str(fork_path),fork_sha256=sha(fork_path),
                            maximum_absolute_errors=errors,fixed_point_max_depth_difference_m=depth_error,
                            compared_state_scalars=sum(value.size for value in arrays.values()),fork_seconds=fork_seconds,
                            reference_full_run_seconds=info.get('seconds'),
                            passed=all(error<=design['maximum_absolute_field_difference'] for error in errors.values())
                                and depth_error<=design['maximum_absolute_fixed_point_depth_difference_m'])
                        write_json(result_path,row);rows.append(row)
                        write_json(run/'progress.json',dict(completed=False,comparisons=len(rows),last_key=key))
                        print('FORK '+key+' max='+str(max(errors.values())),flush=True)
    assert len(rows)==design['maximum_fork_integrations']==64
    summary=dict(comparisons=len(rows),all_passed=all(r['passed'] for r in rows),
        maximum_absolute_errors={name:max(r['maximum_absolute_errors'][name] for r in rows) for name in design['evolved_fields']},
        maximum_fixed_point_depth_difference_m=max(r['fixed_point_max_depth_difference_m'] for r in rows),
        compared_state_scalars=sum(r['compared_state_scalars'] for r in rows),
        seconds=time.perf_counter()-started,old_confirmation_labels_read=0,learned_models=0,
        interpretation='Saved-state branch equivalence for reference generation only; no physical model advantage or task efficacy claim.')
    write_json(run/'comparisons.json',rows);write_json(run/'summary.json',summary)
    write_json(run/'completion.json',dict(completed=True,all_passed=summary['all_passed']))
    write_json(run/'progress.json',dict(completed=True,comparisons=len(rows)))
    write_json(ROOT/'results/latest_hydraulic_fork_equivalence.json',dict(run=str(run),completed=True,
        all_passed=summary['all_passed'],audit_pending=True))
    print('SUMMARY '+str(summary),flush=True)


if __name__=='__main__':main()
