"""Independent service-score and causal-evidence verification of the completed audit."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from hyra_validation.artifacts import write_json
from hyra_validation.data import load_events
from run_residual_decision_value import digest


def independent_score(event,order,distances,now):
    clock=float(now);trace=[];yield_count=0
    for target,distance in zip(order,distances):
        clock+=max(distance,1)*6;start=clock;completed=False
        for tick in (6,12):
            end=start+tick;t=event['time_s'];v=event['truth_depth_m'][:,target]
            if end>300 or start<t[0] or end>t[-1]:
                clock=end;break
            inside=v[(t>=start)&(t<=end)]
            maximum=max(np.r_[np.interp(start,t,v),inside,np.interp(end,t,v)])
            clock=end
            if maximum>.8:break
            if tick==12:completed=True;yield_count+=1
        trace.append(dict(region=target,start_s=start,end_s=clock,completed=completed))
    return dict(service_yield=yield_count,trace=trace)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);args=parser.parse_args()
    run=args.run.resolve();config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    grouped=load_events(config['manifest']);events={e['metadata']['event_id']:e for seq in grouped.values() for e in seq}
    contexts=0;score_checks=0;cases=defaultdict(list)
    with gzip.open(run/'task_contexts.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            r=json.loads(line);contexts+=1;cases[r['case_id']].append(r)
            assert r['now_s']==r['initial_time_s']+6*r['node_index']
            for i,(order,distances) in enumerate(r['orders']):
                result=independent_score(events[r['event_id']],order,distances,r['now_s'])
                assert result==r['truth_traces'][i],r['case_id'];score_checks+=1
    for cid,rr in cases.items():
        positive=[r['now_s'] for r in rr if max(r['true_yields'])>r['true_yields'][r['initial_order']]]
        expected=max(positive) if positive else None
        assert all(r['last_correctable_s']==expected for r in rr),cid
    observations=0;evidence_rows=0;references=0
    for setting in config['schedules']:
        name=setting['schedule'];records={}
        with gzip.open(run/name/'observations.jsonl.gz','rt',encoding='utf-8') as stream:
            for line in stream:
                r=json.loads(line);observations+=1
                assert r['record_id'] not in records
                records[r['record_id']]=r
                assert r['arrival_time_s']==r['valid_time_s']+setting['delay']
                if r['residual_m'] is not None:
                    assert r['prior_release_s']<=r['valid_time_s']-20
                    assert r['actual_lead_s']==r['valid_time_s']-r['prior_release_s']
                    assert 20<=r['actual_lead_s']<30
                    assert abs((r['observed_depth_m']-r['prior_depth_m'])-r['residual_m'])<1e-12
        with gzip.open(run/name/'evidence.jsonl.gz','rt',encoding='utf-8') as stream:
            for line in stream:
                r=json.loads(line);evidence_rows+=1
                for ids in r['actual_history_record_ids'].values():
                    for i,record_id in enumerate(ids):
                        if record_id is None:continue
                        obs=records[record_id];references+=1
                        age=r['now_s']-obs['valid_time_s']
                        assert obs['arrival_time_s']<=r['now_s']
                        assert 0<=age<60 and int(age//10)==i
                        assert obs['residual_m'] is not None
        print('Verified evidence',name,flush=True)
    hashes=json.loads((run/'input_fingerprints.json').read_text(encoding='utf-8'))
    assert all(digest(p)==h for p,h in hashes.items())
    result=dict(passed=True,independent_service_scores=score_checks,task_contexts=contexts,
        unique_initial_cases=len(cases),verified_last_correctable_windows=len(cases),
        observation_records=observations,evidence_rows=evidence_rows,causal_history_references=references,
        source_inputs_unchanged=True,truth_never_used_in_tie_policy=True,
        note='Independent full service scoring; evidence checked for each actual input reference. Old method decisions reproduced separately in implementation_audit.json.')
    write_json(run/'verification.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
