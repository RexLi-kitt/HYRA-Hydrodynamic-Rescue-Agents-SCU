"""One-unit sensitivity around the80-person burst candidate, not score search."""
from pathlib import Path
from dataclasses import replace,asdict
import sys,json,time,copy,os,argparse,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'flood-drop-variable-v1'))
from screen import read,write,sha,fingerprint,run_case,summarize
from variable_environment import Config

def run(a):
    out=a.output;out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    try:
        assert read(a.previous/'status.json')['status']=='completed'
        assert read(a.previous.parent/'completion-audit.json')['status']=='passed'
        parent=read(a.previous/'protocol.json');s=read(a.previous/'scenarios.json')
        rows=[json.loads(line) for line in (a.previous/'results.jsonl').read_text().splitlines()]
        index={(r['case'],r['method']):r for r in rows}
        center_ids=s['links']['n80_w180_three_bursts']['candidate'];whole_ids=s['links']['n120_w180_three_bursts']['candidate']
        specs=[dict(id=f'{key}_{value}',key=key,value=value) for key,values in (('people',(79,81)),('window',(179,181))) for value in values]
        protocol=dict(version='burst-one-unit-neighborhood-v1',date='2026-11-02',center='N80,T1200,W180,three_bursts60',
            reason='Middle of three observed target-band load groups; selected before neighbors. Check robustness, do not replace center with a worse-scoring neighbor.',
            specs=specs,methods=['original','dispatch'],cases_per_profile=12,executed_rollouts=96,reused_center_rollouts=24,
            primary='People completion; retain all-person and ceil95% scene metrics. No neural runs or final test.',
            selection='Center remains the next-study candidate regardless of which neighbor favors learning; no learned results used.',
            pairing='Take identical exposure IDs from full120-person pool; only N or individual deadline changes by1.',
            source_hashes={str(f):sha(f) for f in HERE.glob('*.py')},
            legacy_source_hashes={**parent['source_hashes'],**parent['legacy_source_hashes']},water_hashes=parent['water_hashes'],
            parent_results_sha256=sha(a.previous/'results.jsonl'),parent_protocol_sha256=sha(a.previous/'protocol.json'),
            stop='STOP_AFTER_CASE or600s; no automatic additional sweep/training')
        write(out/'protocol.json',protocol)
        cases={};links={};center_rows=[copy.deepcopy(index[c,m]) for c in center_ids for m in protocol['methods']]
        write(out/'center-reused.json',dict(rows=center_rows,source=str(a.previous/'results.jsonl'),source_sha256=protocol['parent_results_sha256']))
        for spec in specs:
            ids=[]
            for center_id,whole_id in zip(center_ids,whole_ids):
                center=s['cases'][center_id];whole=s['cases'][whole_id]
                cfg=replace(Config(**center['scenario']['config']),**{spec['key']:spec['value']})
                scenario=copy.deepcopy(whole['scenario']);select=set(sorted(e['exposure_id'] for e in scenario['events'])[:cfg.people])
                scenario['events']=[e for e in scenario['events'] if e['exposure_id'] in select];scenario['config']=asdict(cfg)
                for e in scenario['events']:e['deadline']=e['fall']+cfg.window
                common={e['exposure_id']:e for e in center['scenario']['events']}
                for e in scenario['events']:
                    if e['exposure_id'] in common:assert all(e[k]==common[e['exposure_id']][k] for k in ('fall','xy','field','trigger'))
                cid=fingerprint(dict(parent=center['field']['parent'],scenario=scenario))
                cases[cid]=dict(id=cid,field=center['field'],scenario=scenario);ids.append(cid)
            links[spec['id']]=ids
        assert len(cases)==48
        write(out/'scenarios.json',dict(cases=cases,links=links));input_hash=sha(out/'scenarios.json');protocol_hash=sha(out/'protocol.json')
        write(out/'input-hashes.json',dict(scenarios=input_hash,protocol=protocol_hash))
        for path,h in {**protocol['source_hashes'],**protocol['legacy_source_hashes'],**protocol['water_hashes']}.items():assert sha(path)==h
        results=[]
        for cid,case in cases.items():
            for method in protocol['methods']:
                if (out/'STOP_AFTER_CASE').exists():raise InterruptedError('Stop requested')
                if time.monotonic()-start>600:raise TimeoutError('Neighborhood cap')
                r=run_case(case,method);results.append(r)
                with (out/'results.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,allow_nan=False)+'\n')
                write(out/'status.json',dict(status='running',pid=os.getpid(),completed=len(results),total=96,elapsed_s=time.monotonic()-start))
        lookup={(r['case'],r['method']):r for r in results};summaries={'center':{m:summarize([r for r in center_rows if r['method']==m]) for m in protocol['methods']}}
        for spec in specs:summaries[spec['id']]={m:summarize([lookup[c,m] for c in links[spec['id']]]) for m in protocol['methods']}
        for x in summaries.values():x['stronger']=max(protocol['methods'],key=lambda m:(x[m]['people_supplied'],x[m]['all_successes'],x[m]['at_least95_successes']))
        for path,h in {**protocol['source_hashes'],**protocol['legacy_source_hashes'],**protocol['water_hashes']}.items():assert sha(path)==h
        assert sha(out/'scenarios.json')==input_hash and sha(out/'protocol.json')==protocol_hash
        write(out/'summary.json',dict(profiles=summaries,neural_runs=0,formal_test=False,source_water_unchanged=True))
        write(out/'status.json',dict(status='completed',pid=os.getpid(),completed=96,total=96,elapsed_s=time.monotonic()-start))
        print(json.dumps(read(out/'status.json')),flush=True)
    except BaseException as ex:
        write(out/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:p.error('--execute required')
    run(a)
