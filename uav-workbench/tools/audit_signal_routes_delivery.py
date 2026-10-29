"""Independently verify final supplementary tables and reproducibility files.

This audit refuses to read confirmation performance until the complete study
and its independent replay/statistics audit have already passed.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import math
import signal_routes
import numpy as np
from signal_routes import METHODS
from signal_routes.common import DEFAULT,ROOT,read,write,digest,sources


def require(value,message):
    if not value: raise AssertionError(message)


def field_matches(text,value):
    if value is None: return text==''
    if isinstance(value,bool): return text==str(value)
    if isinstance(value,(float,int)):
        return math.isfinite(float(text)) and math.isclose(float(text),value,rel_tol=0,abs_tol=1e-12)
    return text==str(value)


def audit(study):
    study=Path(study)
    require(read(study/'status.json')['status']=='completed','Full study is incomplete; confirmation remains unread')
    full=read(study/'completion-audit.json')
    require(full['passed'] and full['report_sha256']==digest(study/'report.md'),'Independent completion audit is absent/stale')
    require(full['results_sha256']==digest(study/'confirmation/results.json'),'Confirmation results changed after independent audit')
    require(full['sources']==sources(),'Frozen source files changed')
    result=read(study/'confirmation/results.json')['rows']
    with (study/'per-episode-expanded.csv').open(encoding='utf-8-sig',newline='') as stream:
        exported=list(csv.DictReader(stream))
    require(len(result)==len(exported)==1944,'Expanded CSV has the wrong episode count')
    duration=read(study/'task-selection.json')['duration_s']; computed=[]; model_runs=0
    failure_keys=('failure_undiscovered','failure_discovered_interrupted','failure_discovered_other_unfinished')
    for raw,actual in zip(result,exported):
        expected={k:raw[k] for k in ('method','training_seed','scenario_seed','parent','mechanism')}
        expected.update(raw['metrics']); is_model=raw['method'] in METHODS
        expected['event_times_supported']=is_model
        for event in ('discovery','rescue'):
            time=raw['metrics'].get(f'first_{event}_s')
            expected[f'first_{event}_restricted_s']=(duration if time is None else min(duration,time)) if is_model else None
            expected[f'first_{event}_censored']=(time is None) if is_model else None
        if is_model:
            events=read(raw['replay'])['events']; sets={}
            for name in ('discovered','rescued','service_interrupted'):
                sets[name]={tuple(e['cell']) for e in events if e['type']==name}
            discovered,rescued,interrupted=(sets[k] for k in ('discovered','rescued','service_interrupted'))
            require(rescued<=discovered and len(discovered)<=5,'Invalid discovery/rescue event sets')
            require(len(rescued)==raw['metrics']['rescued'],'Rescue event count differs from metrics')
            unfinished=discovered-rescued
            expected.update(failure_undiscovered=5-len(discovered),
                failure_discovered_interrupted=len(unfinished&interrupted),
                failure_discovered_other_unfinished=len(unfinished-interrupted),
                rescued_after_interruption=len(rescued&interrupted),all_rescued_censored=len(rescued)<5,
                all_rescued_s=max(e['time_s'] for e in events if e['type']=='rescued') if len(rescued)==5 else None)
            require(sum(expected[k] for k in failure_keys)==5-len(rescued),'Failure categories do not partition unfinished rescues')
            model_runs+=1
        else:
            for key in (*failure_keys,'rescued_after_interruption','all_rescued_s','all_rescued_censored'):
                expected[key]=None
        for key,value in expected.items():
            require(key in actual and field_matches(actual[key],value),f'Expanded CSV mismatch: {raw["method"]} {key}')
        computed.append(expected)
    require(model_runs==1728,'Not every model episode was checked')
    supplement=read(study/'supplemental-results.json'); parents=sorted({r['parent'] for r in computed})
    require(len(parents)==8,'Wrong independent parent count')
    methods=set(METHODS)|{'frontier','sweep_refresh','water_aware'}
    require({r['method'] for r in supplement['parent_equal_weighted']}==methods,'Supplement omitted a method')
    for entry in supplement['parent_equal_weighted']:
        for key,recorded in entry.items():
            if key=='method': continue
            groups=[]
            for parent in parents:
                values=[r[key] for r in computed if r['method']==entry['method'] and r['parent']==parent and r.get(key) is not None]
                if values: groups.append(sum(values)/len(values))
            expected=sum(groups)/len(groups) if groups else None
            require((recorded is None and expected is None) or (recorded is not None and expected is not None and
                    math.isclose(recorded,expected,rel_tol=0,abs_tol=1e-10)),f'Parent summary mismatch: {entry["method"]} {key}')
    indices=np.random.default_rng(20261027).integers(0,8,size=(10000,8))
    expected_ids={(condition,m,b) for condition in ('seen_single','unseen_double')
                  for i,m in enumerate(('R1','R2','R3'),1) for b in ('Bpost',f'C{i}')}
    contrasts=supplement['mechanism_paired_contrasts']
    require(len(contrasts)==12 and {(r['condition'],r['method'],r['baseline']) for r in contrasts}==expected_ids,
            'Missing mechanism subgroup/matched control')
    for row in contrasts:
        values={}
        for method in (row['method'],row['baseline']):
            values[method]=np.asarray([np.mean([r['rescue_fraction'] for r in computed
                if r['method']==method and r['parent']==parent and
                (r['mechanism']=='double')==(row['condition']=='unseen_double')]) for parent in parents])
        difference=values[row['method']]-values[row['baseline']]
        np.testing.assert_allclose(row['paired_parent_deltas'],difference,rtol=0,atol=1e-14)
        np.testing.assert_allclose(row['delta'],difference.mean(),rtol=0,atol=1e-14)
        np.testing.assert_allclose(row['ci95'],np.percentile(difference[indices].mean(axis=1),[2.5,97.5]),rtol=0,atol=1e-14)
    direction=read(study/'independent-direction-diagnostics.json')
    require(not direction['confirmation_data_read'] and not direction['policies_or_parameter_files_modified'],
            'Direction diagnostic violated its development-only scope')
    require(direction['frozen_training_sources']==sources(),'Stale directional source evidence')
    require(direction['development_records_sha256']==digest(study/'development-records.npz'),'Development diagnostic inputs changed')
    for name,expected in direction['parameter_sha256'].items():
        require(digest(study/name)==expected,'Directional diagnostic parameter file changed')
    require(len(direction['rows'])==684 and direction['parent_count']==3,'Incomplete direction diagnostic')
    mean_direction=read(study/'mean-direction-diagnostic.json')
    require(mean_direction['passed'] and not mean_direction['confirmation_performance_read'],
            'Mean-route direction diagnostic scope invalid')
    require(mean_direction['contract']['sources']==sources() and
            mean_direction['contract']['parameters_sha256']==digest(study/'parameters.json') and
            mean_direction['contract']['development_sha256']==digest(study/'development-records.npz'),
            'Stale mean-route direction diagnostic')
    require(len(mean_direction['rows'])==36 and all(r['nonzero_fitted_mean'] and
            r['mean_max_difference']==r['variance_max_difference']==0 for r in mean_direction['rows']),
            'Incomplete mean-route reversal evidence')
    reproduction=read(study/'reproduction-manifest.json')
    require(reproduction['frozen_sources']==sources(),'Reproduction source manifest changed')
    for relative,expected in reproduction['frozen_sources'].items():
        require(digest(study/'frozen-sources'/relative)==expected,'Frozen source snapshot differs')
    for name,expected in reproduction['external_tools'].items():
        require(digest(study/'external-sources'/name)==expected,'External tool snapshot differs')
        require(digest(ROOT/'tools'/name)==expected,'Current external tool differs from its recorded snapshot')
    for key,name in (('dataset_manifest_sha256','data/dataset.json'),('parameters_sha256','parameters.json'),
                     ('registration_sha256','registration.json'),('task_sha256','task-selection.json')):
        require(reproduction[key]==digest(study/name),'Reproduction input digest differs')
    figures=('paired-effects.png','paired-effects.svg','training-curves.png','training-curves.svg')
    for name in figures:
        path=study/'figures'/name
        require(path.is_file() and path.stat().st_size>1000,f'Missing scientific figure: {name}')
    evidence=dict(passed=True,audit_tool_sha256=digest(__file__),expanded_rows=1944,model_failure_and_time_rows=1728,
        parent_aggregated_methods=11,mechanism_paired_contrasts=12,directional_scoring_rows=684,mean_direction_reversals=36,
        source_snapshot_files=len(reproduction['frozen_sources']),figure_files=list(figures),
        report_sha256=digest(study/'report.md'),supplemental_sha256=digest(study/'supplemental-results.json'),
        expanded_csv_sha256=digest(study/'per-episode-expanded.csv'),
        independent_completion_audit_sha256=digest(study/'completion-audit.json'))
    write(study/'delivery-audit.json',evidence)
    return evidence


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--study',type=Path,default=DEFAULT)
    print(audit(parser.parse_args().study))
