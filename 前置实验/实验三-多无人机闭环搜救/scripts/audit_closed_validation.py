"""Independently verify all saved validation episodes and the fixed B* rule."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import math
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
import numpy as np
from hyra_exp3.artifacts import new_run,finish_run,sha256,write_json
from hyra_exp3.protocol import load_protocol
from hyra_exp3.execution_plan import register
from hyra_exp3.models import METHODS
from hyra_exp3.diagnostics import load_field
from hyra_exp3.evaluation import config_for
from hyra_exp3.formal_audit import audit_saved_episode


def audit_family(case,rows,plan,checkpoints):
    for kind in ('truth','nominal'):
        manifest=case[kind]['manifest']
        assert manifest['kind']==kind and manifest['independent_initial_condition']
        assert not manifest.get('formal_test',False)
        assert sha256(manifest['field_path'])==manifest['field_sha256']
    truth=load_field(case['truth']['manifest']['field_path'])
    assert np.array_equal(truth['times_s'],np.arange(1201.))
    totals={}
    for row in rows:
        config=config_for(plan,row['family_id'],50,row['layout'])
        episode=row['episode'];assert sha256(episode['path'])==episode['sha256']
        result=load_protocol(episode['path']);record=result['execution_record']
        assert record['trained_model'] and not record['formal_test'] and record['model_parameter_count']==37070
        assert record['method']==row['method'] and record['provenance']==checkpoints[row['method'],row['training_seed']]
        assert row['summary']==result['summary'] and row['outcome_analysis']==result['outcome_analysis']
        candidates=np.argwhere(np.ones((50,50),bool))
        rng=np.random.default_rng(np.random.SeedSequence([config.seed,1]))
        targets=candidates[rng.choice(len(candidates),30,replace=False)].tolist()
        assert targets==[t['cell'] for t in result['targets']]
        for key,value in audit_saved_episode(episode['path'],truth,config).items():totals[key]=totals.get(key,0)+value
    return totals


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3,choices=range(1,5));args=parser.parse_args()
    directory=args.run.resolve();sources=load_protocol(directory/'arguments.json')
    target=new_run(ROOT,'closed_loop_validation_independent_audit',dict(run=str(directory),workers=args.workers))
    print(f'Closed-loop validation independent audit directory: {target}',flush=True)
    try:
        summary=load_protocol(directory/'validation_summary.json');rows=summary['rows'];plan=register(ROOT)
        families={f['family_id'] for f in plan['allocation'] if f['split']=='validation'}
        expected={(f,l,m,s) for f in families for l in (0,1) for m in METHODS for s in (101,202,303)}
        identities=[(r['family_id'],r['layout'],r['method'],r['training_seed']) for r in rows]
        assert summary['status']=='completed' and summary['episodes']==360 and not summary['test_accessed']
        assert len(identities)==len(set(identities))==360 and set(identities)==expected
        assert all(r['side']==50 and r['delay_s']==0 for r in rows)
        training=load_protocol(Path(sources['training_run'])/'training_summary.json');checkpoints={}
        for row in training['models']:
            cp=row['selected'];assert sha256(cp['path'])==cp['sha256']
            checkpoints[row['method'],row['seed']]=dict(method=row['method'],seed=row['seed'],
                checkpoint_path=cp['path'],checkpoint_sha256=cp['sha256'],selected_step=cp['step'])
        cases=[load_protocol(p) for p in (Path(sources['data_run'])/'jobs').glob('*side50/case.json')]
        cases=[c for c in cases if c['family']['family_id'] in families];assert len(cases)==12
        totals={}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(audit_family,c,[r for r in rows if r['family_id']==c['family']['family_id']],plan,checkpoints) for c in cases]
            for i,future in enumerate(as_completed(futures),1):
                for key,value in future.result().items():totals[key]=totals.get(key,0)+value
                print(f'Closed-loop validation read-back audit {i}/12 families',flush=True)
        fixed_path=Path(sources['fixed_run'])/'fixed_validation_summary.json';fixed=load_protocol(fixed_path)
        selected=load_protocol(directory/'B_star_selection.json');assert selected==summary['B_star']
        scores={}
        for method in ('H','E'):
            services=[r['summary']['success_rate'] for r in rows if r['method']==method]
            predictions=[r for r in fixed['records'] if r['method']==method and r['delay_s']==0]
            assert len(services)==len(predictions)==72
            scores[method]=dict(success_rate=math.fsum(services)/72,
                window_error_rate=math.fsum(r['window_error_rate'] for r in predictions)/72,
                prediction_mse_m2=math.fsum(r['mse_m2'] for r in predictions)/72)
            assert all(math.isclose(selected['scores'][method][k],v,rel_tol=0,abs_tol=1e-14) for k,v in scores[method].items())
        ranks={m:(-s['success_rate'],s['window_error_rate'],s['prediction_mse_m2'],m!='H') for m,s in scores.items()}
        assert selected['B_star']==min(ranks,key=ranks.get) and not selected['P_or_test_results_used']
        assert selected['selected_once'] and selected['selection_split']=='validation_only'
        result=dict(status='passed',**totals,validation_families=12,full_five_method_seed_layout_matrix=True,
            test_accessed=False,B_star=selected['B_star'],selection_rule_recomputed_for_audit_only=True,
            all_saved_episodes_independently_reaudited=True,summary_sha256=sha256(directory/'validation_summary.json'),
            fixed_summary_sha256=sha256(fixed_path),selection_sha256=sha256(directory/'B_star_selection.json'))
        write_json(directory/'validation_integrity_audit.json',result);write_json(target/'audit_result.json',result)
        finish_run(ROOT,target);print(result,flush=True)
    except BaseException as error:
        (target/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        finish_run(ROOT,target,status='failed',error=repr(error));raise
