from pathlib import Path
import json, hashlib, math
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'uav-frozen-rigor-20261104/run01'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

assert read(RUN/'status.json')['state']=='completed'
rows=[json.loads(x) for x in (RUN/'results.jsonl').read_text(encoding='utf-8').splitlines()]
scenarios=read(RUN/'scenarios.json');summary=read(RUN/'summary.json')
assert sha(RUN/'scenarios.json')==read(RUN/'input-lock.json')['scenarios_sha256']
methods=['coverage100','flow_coverage','statistic','linear','gru-frozen']
assert len(rows)==260
assert {(r['case'],r['method']) for r in rows}=={(c['id'],m) for c in scenarios for m in methods}
for c in scenarios:
 rr=[r for r in rows if r['case']==c['id']]
 assert len(rr)==5
 expected=[(i,e['fall'],e['deadline']) for i,e in enumerate(c['scenario']['events'])]
 for r in rr:
  assert [(p['id'],p['fall'],p['deadline']) for p in r['persons']]==expected
  n=sum(p['status']==2 for p in r['persons'])
  assert n==r['supplied'] and r['p95_scene']==(n>=math.ceil(.95*r['people'])) and r['all_scene']==(n==r['people'])
  assert r['refloats']==sum(p['refloats'] for p in r['persons'])
  assert r['hidden_checks']==2 and r['min_separation']>=5-1e-8
  assert all(p['status'] in [2,3,4] for p in r['persons'])
  for p in r['persons']:
   if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
for family,mm in summary['results'].items():
 for m,a in mm.items():
  rr=[r for r in rows if r['family']==family and r['method']==m]
  actual=dict(scenes=len(rr),people=sum(r['people'] for r in rr),supplied=sum(r['supplied'] for r in rr),p95_scenes=sum(r['p95_scene'] for r in rr),all_scenes=sum(r['all_scene'] for r in rr),refloats=sum(r['refloats'] for r in rr))
  assert actual==a,(family,m)
keys=sorted(r['case'] for r in rows if r['family']=='original' and r['method']=='gru-frozen')
idx=np.random.default_rng(2026112902).integers(0,40,(20000,40))
g={r['case']:r for r in rows if r['family']=='original' and r['method']=='gru-frozen'}
pair_out={}
for m in methods[:-1]:
 b={r['case']:r for r in rows if r['family']=='original' and r['method']==m}
 p=np.array([100*(g[k]['supplied']-b[k]['supplied'])/40 for k in keys]);q=np.array([100*(int(g[k]['p95_scene'])-int(b[k]['p95_scene'])) for k in keys])
 a=summary['paired_primary'][m]
 np.testing.assert_allclose(a['people_95'],np.quantile(p[idx].mean(1),[.025,.975]))
 np.testing.assert_allclose(a['scene_95'],np.quantile(q[idx].mean(1),[.025,.975]))
 assert a['people_pp']==p.mean() and a['scene_pp']==q.mean()
 pair_out[m]=dict(people_wins=int((p>0).sum()),people_ties=int((p==0).sum()),people_losses=int((p<0).sum()),p95_gru_only=sum(g[k]['p95_scene'] and not b[k]['p95_scene'] for k in keys),p95_baseline_only=sum(b[k]['p95_scene'] and not g[k]['p95_scene'] for k in keys))
prior=read(ROOT/'uav-fine-memory-20261104/run01/scenarios.json')
prior_hashes={sha(c['field']['path']) for group in prior.values() for c in group}
new_hashes={sha(c['field']['path']) for c in scenarios if c['family']!='manufactured_rewet'}
assert len(new_hashes)==48 and not prior_hashes&new_hashes
failures={}
for family in summary['results']:
 failures[family]={}
 for m in methods:
  pp=[p for r in rows if r['family']==family and r['method']==m for p in r['persons']]
  failures[family][m]=dict(failed=sum(p['status'] in [3,4] for p in pp),expired=sum(p['status']==3 for p in pp),outside=sum(p['status']==4 for p in pp),failed_never_seen=sum(p['status'] in [3,4] and p['first_seen']<0 for p in pp),failed_not_seen_by_deadline=sum(p['status'] in [3,4] and (p['first_seen']<0 or p['first_seen']>p['deadline']) for p in pp),people_refloated=sum(p['refloats']>0 for p in pp),supplied_after_any_refloat=sum(p['status']==2 and p['refloats']>0 for p in pp))
audit=dict(rows=260,person_records=10400,scenario_hash=True,same_events_for_all_methods=True,all_summary_cells_recomputed=True,bootstrap_intervals_recomputed=True,old_dataset_field_hashes=len(prior_hashes),new_independent_swe_field_hashes=48,no_old_field_hash_overlap=True,paired_counts=pair_out,failure_and_refloat=failures,limitations='Hashes and mutation probes are implementation checks, not proof of realism or exhaustive absence of information leaks. Bootstrap intervals are per comparison, not simultaneous multiple-comparison bounds. Frozen GRU is one selected training seed/checkpoint; training-seed variability remains unmeasured.')
(RUN/'secondary-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False))
