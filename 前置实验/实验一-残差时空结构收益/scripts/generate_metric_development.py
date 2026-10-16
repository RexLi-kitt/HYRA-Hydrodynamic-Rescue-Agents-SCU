"""Generate fixed fresh train/validation splits. Pilot and confirmation stay sealed."""
from pathlib import Path
import argparse
import copy
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from diagnose_numerical_stability import read,load,sha
from calibrate_physical_scale import ScaleFields
from hyra_validation.artifacts import new_run,write_json
from hyra_validation.metric_family import draw_metric_family,independent_restart,verify_manifest


def field_design(design,mesh,final,horizon):
    return dict(mesh=mesh,output_interval_s=design['output_interval_s'],final_time_s=final,
                forecast_horizon_s=horizon,maximum_new_integrations=64)


def generate_family(run,design,split,index):
    meta=draw_metric_family(design,split,index);family=meta['family_id'];folder=run/'families'/family
    folder.mkdir(parents=True,exist_ok=True);manifest_path=folder/'manifest.json'
    if (folder/'completion.json').exists():
        complete=read(folder/'completion.json');assert sha(manifest_path)==complete['manifest_sha256']
        return verify_manifest(manifest_path)
    if (folder/'metadata.json').exists():assert read(folder/'metadata.json')==meta
    else:write_json(folder/'metadata.json',meta)
    all_fields={};references={}
    for mesh in design['reference_meshes']:
        storage=folder/f'reference{mesh}';storage.mkdir(exist_ok=True)
        fields=ScaleFields(storage,field_design(design,mesh,390,90))
        base=fields.get(meta,100,'truth_base');raw=load(base['info']['raw_path'])
        branches={}
        for anchor in design['anchors_s']:
            restart=independent_restart(base['info'],raw,anchor,'reference_base_environment_only_not_forecast')
            for center in design['centers_y_m']:
                pulse=fields.get(meta,100,'truth_pulse',anchor,center,design['pulse_amplitude_m'],restart=restart)
                np.testing.assert_allclose(pulse['linear'][0],base['linear'][np.flatnonzero(base['time_s']==anchor)[0]],atol=1e-10,rtol=0)
                branches[f'a{anchor}_y{center}']=pulse['info']['key']
        references[str(mesh)]=dict(base=base['info']['key'],branches=branches)
        all_fields.update(fields.registry)
        write_json(run/'progress.json',dict(completed=False,current_family=family,current_stage=f'reference{mesh}_done'))
    (folder/'nominal40').mkdir(exist_ok=True)
    nominal=ScaleFields(folder/'nominal40',field_design(design,40,450,150))
    base=nominal.get(meta,100,'nominal_base');raw=load(base['info']['raw_path'])
    sensitivities={}
    for label,sign in [('plus',1),('minus',-1)]:
        perturbed=copy.deepcopy(meta)
        perturbed['nominal_specifications'][0]['factor']*=1+sign*design['global_sensitivity_relative_step']
        field=nominal.get(perturbed,100,'nominal_global_'+label)
        sensitivities[label]=field['info']['key']
    calibration={};versions={}
    for anchor in design['anchors_s']:
        restart=independent_restart(base['info'],raw,anchor,'same_mesh_independent_nominal_base')
        zero=nominal.get(meta,100,'nominal_calibration_base',anchor=anchor,restart=restart)
        probes={}
        for center in design['centers_y_m']:
            probe=nominal.get(meta,100,'nominal_probe',anchor,center,design['probe_amplitude_m'],restart=restart)
            np.testing.assert_allclose(probe['linear'][0],zero['linear'][0],atol=1e-10,rtol=0)
            probes[str(center)]=probe['info']['key']
        calibration[str(anchor)]=dict(zero=zero['info']['key'],probes=probes)
        versions[str(anchor)]=dict(field_key=zero['info']['key'],published_horizon_s=60,published_rows=31,
                                  actual_release_s=anchor,shared_with_unperturbed_nominal_calibration=True)
    (folder/'forecasts40').mkdir(exist_ok=True)
    forecasts=ScaleFields(folder/'forecasts40',field_design(design,40,450,60))
    a,b,step=design['forecast_release_grid']
    for release in range(a,b+1,step):
        if str(release) in versions:continue
        restart=independent_restart(base['info'],raw,release,'same_mesh_independent_nominal_base')
        f=forecasts.get(meta,100,'nominal_forecast',anchor=release,restart=restart)
        versions[str(release)]=dict(field_key=f['info']['key'],published_horizon_s=60,published_rows=31,
                                  actual_release_s=release,shared_with_unperturbed_nominal_calibration=False)
    all_fields.update(nominal.registry);all_fields.update(forecasts.registry)
    manifest=dict(family=family,split=split,index=index,metadata=meta,fields=all_fields,references=references,
        nominal_base=base['info']['key'],nominal_sensitivities=sensitivities,nominal_calibration=calibration,
        forecast_versions={key:versions[key] for key in sorted(versions,key=int)},
        physical_library_uses_only_nominal_states=True,model_target_observations_allowed=False)
    assert len(all_fields)==47 and len(versions)==24
    write_json(manifest_path,manifest)
    write_json(folder/'completion.json',dict(completed=True,manifest_sha256=sha(manifest_path),fields=len(all_fields)))
    print('FAMILY_DONE '+family,flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path)
    args=parser.parse_args();design=read(ROOT/'configs/metric_development.design.v1.json')
    run=args.run.resolve() if args.run else new_run('fresh_metric_development_data',dict(design=design,splits=['train','validation']))
    if args.run:
        for rel,checksum in read(run/'provenance.json')['files'].items():assert sha(ROOT/rel)==checksum,rel
    print('RUN '+str(run),flush=True);started=time.perf_counter();families=[]
    write_json(ROOT/'results/latest_metric_development_data.json',dict(run=str(run),completed=False,audit_pending=True))
    for split in ('train','validation'):
        for index in range(design['planned_family_counts'][split]):
            manifest=generate_family(run,design,split,index);families.append(dict(
                family=manifest['family'],split=split,manifest_path=str(run/'families'/manifest['family']/'manifest.json')))
            write_json(run/'family_index.json',families)
            write_json(run/'progress.json',dict(completed=False,completed_families=len(families),last_family=manifest['family']))
    write_json(run/'completion.json',dict(completed=True,families=len(families),seconds=time.perf_counter()-started,
        fields_per_family=47,total_fields=len(families)*47,pilot_or_confirmation_opened=False))
    write_json(run/'progress.json',dict(completed=True,completed_families=len(families)))
    write_json(ROOT/'results/latest_metric_development_data.json',dict(run=str(run),completed=True,audit_pending=True))
    print('COMPLETED '+str(run),flush=True)


if __name__=='__main__':main()
