"""Recompute saved full-centroid differences for all reference-state forks."""
from pathlib import Path
import argparse
import sys
import numpy as np

from audit_pulse_service_alignment import read,load,digest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from hyra_validation.artifacts import write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    run=parser.parse_args().run.resolve();design=read(run/'config.json')['design']
    assert read(run/'completion.json')['completed']
    for rel,checksum in read(run/'provenance.json')['files'].items():assert digest(ROOT/rel)==checksum,rel
    registries={40:read(Path(design['scale_parent'])/'field_registry.json'),
                80:read(Path(design['refinement_parent'])/'field_registry.json')}
    rows=read(run/'comparisons.json');assert len(rows)==64
    maxima={name:0. for name in design['evolved_fields']};depth_maximum=0.;scalars=0
    for row in rows:
        assert row['restart_source_kind']=='reference_base_environment_only_not_forecast'
        assert 'truth_base' in Path(row['restart_source_path']).name
        for name,checksum in [('fork_path','fork_sha256'),('reference_path','reference_sha256'),
                              ('restart_source_path','restart_source_sha256'),('metadata_path','metadata_sha256')]:
            assert digest(row[name])==row[checksum]
        assert read(row['metadata_path'])['split']=='train'
        fork=load(row['fork_path']);ref=load(row['reference_path']);base=load(row['restart_source_path'])
        ti=np.flatnonzero(base['time_s']==row['anchor_s'])[0];indices=np.searchsorted(ref['time_s'],fork['time_s'])
        np.testing.assert_array_equal(ref['time_s'][indices],fork['time_s'])
        assert fork['time_s'][0]==row['anchor_s']
        assert fork['time_s'][-1]==min(row['anchor_s']+90,ref['time_s'][-1])
        for name in maxima:
            np.testing.assert_allclose(fork[name][0],base[name][ti],atol=1e-12,rtol=0)
            error=float(np.max(abs(fork[name]-ref[name][indices])))
            assert abs(error-row['maximum_absolute_errors'][name])<1e-18
            maxima[name]=max(maxima[name],error);scalars+=fork[name].size
        info=registries[row['mesh']][row['key']]
        assert digest(info['projected_path'])==info['projected_sha256']
        projected=load(info['projected_path'])
        indices=np.searchsorted(projected['time_s'],fork['time_s'])
        error=float(np.max(abs(fork['linear_depth_m']-projected['linear'][indices])))
        assert abs(error-row['fixed_point_max_depth_difference_m'])<1e-12
        depth_maximum=max(depth_maximum,error)
    summary=read(run/'summary.json')
    assert summary['compared_state_scalars']==scalars
    assert all(abs(maxima[name]-summary['maximum_absolute_errors'][name])<1e-18 for name in maxima)
    assert all(error<=design['maximum_absolute_field_difference'] for error in maxima.values())
    assert depth_maximum<=design['maximum_absolute_fixed_point_depth_difference_m']
    assert summary['all_passed']
    write_json(run/'independent_audit.json',dict(passed=True,comparisons=len(rows),compared_state_scalars=scalars,
        maximum_absolute_errors=maxima,maximum_fixed_point_depth_difference_m=depth_maximum,
        audit_source_sha256=digest(__file__),scope='full saved postfork states and fixed-point depths, not model efficacy'))
    pointer=read(ROOT/'results/latest_hydraulic_fork_equivalence.json');assert Path(pointer['run'])==run
    pointer.update(audit_pending=False,independent_audit_passed=True)
    write_json(ROOT/'results/latest_hydraulic_fork_equivalence.json',pointer)
    print('AUDIT_PASSED',len(rows),scalars,maxima,depth_maximum)


if __name__=='__main__':main()
