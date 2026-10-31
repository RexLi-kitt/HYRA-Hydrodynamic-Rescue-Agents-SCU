"""User-requested read-only multi-aspect check of complete saved experiments."""
from pathlib import Path
import csv,hashlib,importlib.util,itertools,json,math
import numpy as np
out=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');r=out/'round76/qualitative_request';r.mkdir(exist_ok=False)
repo=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def mod(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
cases={'baseline_0p7m':out/'round60/runs/f4-n0.042-layered-v1','candidate_0p35m_not_adopted':out/'round62/runs/f2-n0.042-layered-v1'}
obs_path=out/'round65/original-dg2-inputs/Location_of_velocity_sampling_points.csv'
protocol=dict(request='测试几个方面数据然后告诉我吻合度',
 purpose='New descriptive classification requested by user, reusing all saved complete data; not a blinded independent validation and not a claim of new solver accuracy.',
 evidence='Okushiri and Merewether laboratory measurements; not field-event safety certification.',
 cases={k:str(v) for k,v in cases.items()},okushiri_case='round47/exact_full; fixed0..22.5s, three stations,451 samples; prior69 rules unchanged',
 trend='Recompute prior69 primary1s block-range net trend, retain final.5s block and all unknowns. Record same/opposite/unknown.',
 crest_order='All3 pair orders of sampled global maxima with prior69 uncertainty brackets; not arrival time.',
 direction='All40 points retained; inherit benchmark0.05m/s observation/model floor for direction. Compute wrapped vector angle at saved model maximum-speed time. Report <=45deg coarse direction match, >45deg mismatch, unavailable unknown. Also show >90deg opposite-hemisphere errors separately.45deg is a declared descriptive orientation bin, not an instrument uncertainty or field acceptance standard.',
 relative_speed='All780 unordered pairs of40 points: compare sign of difference in observed and simulated maximum speeds; difference <=1e-12 treated tied/unknown, never discarded. Pair counts share the40 points and are NOT780 independent experiments. No fabricated speed uncertainty or safety category.',
 wet_extent='No TP/TN/IoU inferred from positive-velocity samples. Carlisle observations without matched complete simulation cannot be scored. Zero speed is not dry.',
 reporting='Primary percentage = matched/all registered comparisons. Also report matched/resolved and resolved/all explicitly; unknown not passed. Never average heterogeneous aspects into overall reality fit.',
 frozen_inputs={str(p):sha(p) for p in [obs_path,out/'round69/protocol.json',out/'round39/output_ch5-7-9.txt',out/'round47/exact_full/timeseries.npz',*[v/'result.json' for v in cases.values()],*[v/'series.json' for v in cases.values()]]})
(r/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')

def counts(labels):
 c={x:labels.count(x) for x in ['consistent','inconsistent','unknown']};n=len(labels);resolved=n-c['unknown']
 return dict(**c,total=n,matched_fraction_all=c['consistent']/n if n else None,matched_fraction_resolved=c['consistent']/resolved if resolved else None,resolved_fraction=resolved/n if n else None)
qual=mod('qual',repo/'flood-platform/benchmarks/compare_okushiri_qualitative.py')
saved=json.loads((out/'round69/protocol.json').read_text(encoding='utf8'))
for f in saved['inputs']:assert sha(f['path'])==f['sha256']
raw=np.loadtxt(out/'round39/output_ch5-7-9.txt',skiprows=1);ref=raw[(raw[:,0]>=0)&(raw[:,0]<=22.5)]
with np.load(out/'round47/exact_full/timeseries.npz') as d:t,o,p=[d[k].copy() for k in ['time_s','observed_m','predicted_m']]
qual.validate_series(t,p,o,ref);rules=saved['phase_rules']
rows=qual.classify_windows(t,o,p,rules['primary_window_s'],rules['resolution_m'],rules['edge_block_fraction'])
peaks={};pairrows=[]
for j,n in enumerate(qual.STATIONS):peaks[n]=dict(observed=qual.peak_interval(t,o[:,j],rules['resolution_m'],saved['sample_step_s']),model=qual.peak_interval(t,p[:,j],rules['resolution_m'],saved['sample_step_s']))
for a,b in saved['peak_order']['all_pairs']:
 x=qual.order(peaks[a]['observed']['sample_support_s'],peaks[b]['observed']['sample_support_s']);y=qual.order(peaks[a]['model']['sample_support_s'],peaks[b]['model']['sample_support_s'])
 pairrows.append(dict(pair=[a,b],observed=x,model=y,comparison=qual.compare(x,y)))
result=dict(protocol_sha256=sha(r/'protocol.json'),okushiri=dict(trend=counts([x['comparison'] for x in rows]),crest_order=counts([x['comparison'] for x in pairrows]),all_trend_rows=rows,all_order_rows=pairrows,peaks=peaks))

# Original CSV stays intact. Check records against archived source after reading headers.
with obs_path.open(encoding='utf-8-sig',newline='') as f:source=list(csv.DictReader(f))
result['observation_csv_columns']=list(source[0]);result['observation_csv_rows']=len(source)
verify=mod('verify',out/'round62/verify_saved_run.py');result['merewether']={}
for label,folder in cases.items():
 check=verify.verify_saved_run(folder);data=json.loads((folder/'result.json').read_text());config=json.loads((folder/'config.json').read_text())
 assert sha(obs_path)==config['source']['files'][obs_path.name]
 points=data['points'];direction=[];speedpairs=[]
 for q in points:
  available=q['direction_error_deg'] is not None
  delta=abs(((math.degrees(math.atan2(q['model_v_at_max_m_s'],q['model_u_at_max_m_s']))-q['angle_xy_deg']+180)%360)-180) if available else None
  direction.append(dict(id=q['id'],observed_angle=q['angle_xy_deg'],model_u=q['model_u_at_max_m_s'],model_v=q['model_v_at_max_m_s'],absolute_direction_difference_deg=delta,
    comparison='unknown' if delta is None else 'consistent' if delta<=45 else 'inconsistent',opposite_hemisphere=delta is not None and delta>90,
    observation_speed=q['speed_m_s'],model_speed=q['model_max_speed_m_s'],maximum_sample_time_s=q['time_of_model_max_s']))
 for a,b in itertools.combinations(points,2):
  od=a['speed_m_s']-b['speed_m_s'];md=a['model_max_speed_m_s']-b['model_max_speed_m_s']
  c='unknown' if abs(od)<=1e-12 or abs(md)<=1e-12 else 'consistent' if od*md>0 else 'inconsistent'
  speedpairs.append(dict(pair=[a['id'],b['id']],observed_difference=od,model_difference=md,comparison=c))
 core={}
 for name,expected in config['code_sha256'].items():
  found=list((repo/'terrain-workbench/terrain-lab').rglob(name));matches=[str(p) for p in found if sha(p)==expected]
  core[name]=dict(archived_sha256=expected,current_matching_paths=matches)
 result['merewether'][label]=dict(direction=counts([x['comparison'] for x in direction]),opposite_hemisphere_ids=[x['id'] for x in direction if x['opposite_hemisphere']],relative_speed=counts([x['comparison'] for x in speedpairs]),all_direction_rows=direction,all_relative_speed_pairs=speedpairs,saved_run_audit=check,current_core_hash_matches=core,quasi_steady=data['quasi_steady'],observation_origin=config['source'],scope='40 point maximum-velocity laboratory comparison over original800s; no dry-area/roads/population validation')
result['unscorable']={'field_spatial_inundation':'Carlisle complete model/epoch/outlet datum not closed; do not claim coverage accuracy.','arrival_and_recession_time':'No matched station arrival/departure observations established here; crest is not arrival.','water_depth_rank':'Merewether lacks observed water-depth values; Okushiri stage is not local depth.','road_or_launch_site':'No matched road passability/launch suitability observations.','people_search_priority':'No independent population/discovery observations.'}
result['reference_backend_round76']='Source-time integration and startup positivity checks passed internally, but no observation-scored complete event yet; these percentages must not be transferred to LF8.'
(r/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'okushiri':{k:result['okushiri'][k] for k in ['trend','crest_order']},'merewether':{k:{n:v[n] for n in ['direction','relative_speed','opposite_hemisphere_ids','quasi_steady']} for k,v in result['merewether'].items()},'columns':result['observation_csv_columns'],'rows':len(source)},ensure_ascii=False,indent=2),flush=True)
