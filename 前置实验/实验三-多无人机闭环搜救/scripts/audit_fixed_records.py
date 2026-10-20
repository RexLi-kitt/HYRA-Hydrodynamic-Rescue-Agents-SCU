"""Read back every fixed-record validation row and recompute its aggregates."""
from pathlib import Path
import argparse
import math
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
from hyra_exp3.artifacts import sha256,write_json
from hyra_exp3.protocol import load_protocol
from hyra_exp3.execution_plan import register
from hyra_exp3.models import METHODS

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();directory=args.run.resolve()
    summary=load_protocol(directory/'fixed_validation_summary.json')
    sources=load_protocol(directory/'arguments.json')
    training=load_protocol(Path(sources['training_run'])/'training_summary.json')
    checkpoints={(r['method'],r['seed']):r['selected'] for r in training['models']}
    families={f['family_id'] for f in register(ROOT)['allocation'] if f['split']=='validation'}
    panels={}
    for p in (Path(sources['data_run'])/'jobs').glob('*side50/case.json'):
        case=load_protocol(p)
        if case['family']['family_id'] in families:
            panels.update({(case['family']['family_id'],p['layout']):p for p in case['panels']})
    expected={(f,l,d,m,s) for f in families for l in (0,1) for d in (0.,10.) for m in METHODS for s in (101,202,303)}
    seen=set();proofs=[];frame_count=0
    assert summary['status']=='completed' and not summary['test_accessed'] and len(summary['records'])==720
    for row in summary['records']:
        key=row['family_id'],row['layout'],row['delay_s'],row['method'],row['training_seed']
        assert key in expected and key not in seen,'Duplicate/unregistered fixed-record identity'
        seen.add(key)
        path=directory/key[0]/f'layout{key[1]}_delay{key[2]:g}'/f'{key[3]}_seed{key[4]}.json'
        raw=load_protocol(path)
        assert {k:v for k,v in raw.items() if k!='frames'}==row,'Raw/summary fixed-record mismatch'
        assert raw['split']=='validation' and raw['labels_used_only_after_forward']
        assert raw['source_panel']==panels[key[:2]],'Methods did not share the same replay input'
        cp=checkpoints[key[3:]];actual=raw['checkpoint']
        assert actual['checkpoint_path']==cp['path'] and actual['checkpoint_sha256']==cp['sha256']
        assert sha256(cp['path'])==cp['sha256'],'Selected checkpoint changed'
        frames=raw['frames'];assert [f['time_s'] for f in frames]==list(range(0,1201,10))
        for frame in frames:
            now=frame['time_s']
            assert frame['points']==2500*sum(now+5*i<=1200 for i in range(13))
            windows=2500*sum(now+offset+12<=min(1200,now+60) for offset in (0,10,20,30,40))
            assert frame['windows']==windows
            assert 0<=frame['false_safe']+frame['false_unsafe']<=windows
            assert all(math.isfinite(frame[k]) and frame[k]>=0 for k in ('squared_error_sum_m2','absolute_error_sum_m','false_safe','false_unsafe'))
        points=sum(f['points'] for f in frames);windows=sum(f['windows'] for f in frames)
        assert (raw['points'],raw['windows'])==(points,windows)
        calculated=dict(mse_m2=sum(f['squared_error_sum_m2'] for f in frames)/points,
            mae_m=sum(f['absolute_error_sum_m'] for f in frames)/points,
            false_safe_rate=sum(f['false_safe'] for f in frames)/windows,
            false_unsafe_rate=sum(f['false_unsafe'] for f in frames)/windows,
            window_error_rate=sum(f['false_safe']+f['false_unsafe'] for f in frames)/windows)
        assert all(math.isclose(raw[k],v,rel_tol=0,abs_tol=1e-14) for k,v in calculated.items()),'Fixed-record aggregate arithmetic differs'
        proofs.append(dict(path=str(path),sha256=sha256(path)));frame_count+=len(frames)
    assert seen==expected and len(expected)==720
    result=dict(status='passed',records=720,frames=frame_count,validation_families=12,test_accessed=False,
        complete_method_seed_layout_delay_matrix=True,shared_input_identity_and_checkpoints_verified=True,
        per_frame_point_and_complete_12s_window_denominators_recomputed=True,all_error_aggregates_recomputed=True,
        per_point_prediction_forward_repeated=False,summary_sha256=sha256(directory/'fixed_validation_summary.json'),raw_records=proofs)
    write_json(directory/'fixed_record_integrity_audit.json',result)
    print({k:v for k,v in result.items() if k!='raw_records'})
