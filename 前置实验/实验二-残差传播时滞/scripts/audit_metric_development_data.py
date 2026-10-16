"""Independent saved-state, projection, seed, and release-contract audit."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.dont_write_bytecode=True
from hyra_validation.artifacts import write_json
from hyra_validation.metric_family import draw_metric_family


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as z:return {key:z[key] for key in z.files}


def audit_family(entry,design):
    import anuga
    path=Path(entry['manifest_path']);manifest=read(path);complete=read(path.parent/'completion.json')
    assert digest(path)==complete['manifest_sha256']
    assert manifest['metadata']==draw_metric_family(design,manifest['split'],manifest['index'])
    assert manifest['split'] in ('train','validation') and len(manifest['fields'])==47
    assert manifest['metadata']['seed_identity'][2]==design['split_ids'][manifest['split']]
    assert not manifest['model_target_observations_allowed'];bases={};fields=manifest['fields'];scalars=0;restarts=0
    for key,info in fields.items():
        if info['kind'] not in ('truth_base','nominal_base'):continue
        assert digest(info['raw_path'])==info['raw_sha256'];bases[info['raw_path']]=load(info['raw_path'])
    f0=manifest['metadata']['nominal_specifications'][0]['factor'];params=manifest['metadata']['params']
    expected_xy=np.array([(5+10*x,5+10*y) for y in range(10) for x in range(10)]);mesh_vertices={}
    for key,info in fields.items():
        assert info['family']==manifest['family'] and info['split']==manifest['split']
        assert digest(info['raw_path'])==info['raw_sha256'] and digest(info['projected_path'])==info['projected_sha256']
        raw=bases.get(info['raw_path']);raw=load(info['raw_path']) if raw is None else raw
        projected=load(info['projected_path']);vertices=projected['vertices'];weights=projected['weights']
        assert np.min(weights)>=0;np.testing.assert_allclose(weights.sum(axis=1),1.,atol=1e-12,rtol=0)
        np.testing.assert_allclose((raw['centroid_xy_m'][vertices]*weights[:,:,None]).sum(axis=1),expected_xy,atol=1e-10,rtol=0)
        if info['mesh'] not in mesh_vertices:
            points,triangles,_=anuga.rectangular_cross(info['mesh'],info['mesh'],len1=100.,len2=100.)
            mesh_vertices[info['mesh']]=points[triangles]
        coordinates=mesh_vertices[info['mesh']]
        np.testing.assert_array_equal(coordinates.mean(axis=1),raw['centroid_xy_m'])
        # ANUGA's callable quantity is evaluated at triangle vertices, then averaged.
        bed=(-params['slope']*coordinates[:,:,0]+params['ridge']*np.exp(-((coordinates[:,:,1]-50)/12)**2)).mean(axis=1)
        np.testing.assert_allclose(raw['bed_m'],bed,atol=1e-12,rtol=0)
        depth=np.maximum(raw['stage']-raw['bed_m'],0.);manual=sum(depth[:,vertices[:,j]]*weights[:,j] for j in range(3))
        np.testing.assert_allclose(manual,projected['linear'],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(raw['time_s'],projected['time_s']);scalars+=manual.size
        nominal=info['kind'].startswith('nominal');factor=f0 if nominal else 1.
        if info['kind']=='nominal_global_plus':factor*=1.02
        if info['kind']=='nominal_global_minus':factor*=.98
        assert abs(factor-info['factor'])<1e-15
        restart=info['restart']
        if restart:
            expected='same_mesh_independent_nominal_base' if nominal else 'reference_base_environment_only_not_forecast'
            assert restart['source_kind']==expected and restart['source_path'] in bases
            baseinfo=next(v for v in fields.values() if v['raw_path']==restart['source_path'])
            assert baseinfo['mesh']==info['mesh'];assert baseinfo['kind']==('nominal_base' if nominal else 'truth_base')
            assert digest(restart['source_path'])==restart['source_sha256']
            base=bases[restart['source_path']];i=np.flatnonzero(base['time_s']==restart['state_time_s'])[0]
            for name in ('stage','xmomentum','ymomentum'):np.testing.assert_allclose(raw[name][0],base[name][i],atol=1e-12,rtol=0)
            restarts+=1
        start=info['anchor_s'] if restart else 0
        if not restart:end=450 if nominal else 390
        else:end=start+(90 if not nominal else (60 if info['kind']=='nominal_forecast' else 150))
        np.testing.assert_array_equal(raw['time_s'],np.arange(start,end+1,2))
        expected_amp=.2 if info['kind']=='truth_pulse' else (.1 if info['kind']=='nominal_probe' else 0.)
        assert info['pulse_amplitude_m']==expected_amp
        if expected_amp:assert info['physical_interval_y_m']==[info['center_y_m']-10.,info['center_y_m']+10.]
    assert restarts==42
    releases=manifest['forecast_versions'];assert sorted(map(int,releases))==list(range(100,331,10))
    for release,spec in releases.items():
        info=fields[spec['field_key']];point=load(info['projected_path'])
        assert spec['actual_release_s']==int(release) and spec['published_rows']==31 and spec['published_horizon_s']==60
        assert info['kind'] in ('nominal_forecast','nominal_calibration_base')
        np.testing.assert_array_equal(point['time_s'][:31],np.arange(int(release),int(release)+61,2))
        assert spec['shared_with_unperturbed_nominal_calibration']==(int(release) in design['anchors_s'])
    return dict(passed=True,family=manifest['family'],split=manifest['split'],manifest_sha256=digest(path),fields=47,
        projection_scalars=scalars,independent_restarts=restarts,actual_published_versions=24,audit_source_sha256=digest(__file__),
        scope='Saved simulated fields and fixed causal release contract; no model efficacy or real-data validation claim')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');parser.add_argument('--limit',type=int,default=36);args=parser.parse_args()
    run=Path(read(ROOT/'results/latest_metric_development_data.json')['run']);design=read(run/'config.json')['design']
    folder=run/'data_audit';folder.mkdir(exist_ok=True);done={}
    print('AUDIT '+str(run),flush=True)
    while True:
        for rel,checksum in read(run/'provenance.json')['files'].items():assert digest(ROOT/rel)==checksum,rel
        for entry in read(run/'family_index.json')[:args.limit]:
            family=entry['family'];output=folder/(family+'.json')
            if family in done:continue
            if output.exists():
                result=read(output);assert result['manifest_sha256']==digest(entry['manifest_path'])
            else:result=audit_family(entry,design);write_json(output,result)
            done[family]=result;write_json(folder/'progress.json',dict(passed=True,audited_families=len(done),families=sorted(done),
                fields=sum(v['fields'] for v in done.values()),projection_scalars=sum(v['projection_scalars'] for v in done.values())))
            print('AUDITED '+family,flush=True)
        if len(done)>=args.limit or not args.watch:break
        time.sleep(20)
    full=bool(len(done)==36 and (run/'completion.json').exists())
    write_json(folder/'completion.json',dict(passed=True,all_development_data_audited=full,audited_families=len(done)))
    if full:
        pointer=read(ROOT/'results/latest_metric_development_data.json');pointer.update(audit_pending=False,independent_audit_passed=True);write_json(ROOT/'results/latest_metric_development_data.json',pointer)
    print('DONE_AUDIT '+str(len(done)),flush=True)


if __name__=='__main__':main()
