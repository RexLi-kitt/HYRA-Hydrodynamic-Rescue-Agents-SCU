from pathlib import Path
import hashlib,json
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round85';B=O/'round83/carlisle_2400';C=R/'carlisle_2400_cfl_half'
s=json.loads((R/'status.json').read_text());b=json.loads((O/'round83/status.json').read_text())
assert s['state']!='running'
if not s.get('internal_gate_pass'): raise SystemExit('Numerical gate failed; do not score incomplete fields.')
protocol=json.loads((R/'protocol.json').read_text());thresholds=protocol['internal_sensitivity_only']['depth_thresholds_m']
frames=[];masks={};baseline_fields=[];candidate_fields=[]
for idx,t in enumerate(protocol['predeclared_times_s']):
    name=f'check-{idx:04d}.wd';a=np.loadtxt(B/'results'/name,skiprows=6);z=np.loadtxt(C/'results'/name,skiprows=6)
    assert a.shape==z.shape==(306,476)
    baseline_fields.append(a);candidate_fields.append(z);d=z-a
    row={'time_s':t,'baseline_sha256':hashlib.sha256((B/'results'/name).read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256((C/'results'/name).read_bytes()).hexdigest(),
         'max_abs_depth_difference_m':float(np.abs(d).max()),'domain_mean_abs_depth_difference_m':float(np.abs(d).mean()),'absolute_volume_difference_m3':float(abs(d.sum()*100)),'wet_mask_comparisons':[]}
    for threshold in thresholds:
        x=a>threshold;y=z>threshold
        both=int((x&y).sum());bx=int((x&~y).sum());cy=int((~x&y).sum());union=both+bx+cy
        row['wet_mask_comparisons'].append({'threshold_m':threshold,'both_wet':both,'baseline_only_wet':bx,'candidate_only_wet':cy,'neither':int((~x&~y).sum()),'union':union,'software_mask_IoU':both/union if union else None,'changed_area_m2':100*(bx+cy)})
        masks[f't{t}_h{threshold}']=(x.astype(np.uint8)+2*y.astype(np.uint8))
    frames.append(row)
first=[]
for threshold in thresholds:
    aa=np.array(baseline_fields)>threshold;zz=np.array(candidate_fields)>threshold
    ax=np.where(aa.any(axis=0),aa.argmax(axis=0),-1);zx=np.where(zz.any(axis=0),zz.argmax(axis=0),-1)
    common=(ax>=0)&(zx>=0);diff=(zx[common]-ax[common])*300
    first.append({'threshold_m':threshold,'both_ever_wet':int(common.sum()),'baseline_only_ever_wet':int(((ax>=0)&(zx<0)).sum()),'candidate_only_ever_wet':int(((ax<0)&(zx>=0)).sum()),'same_first_saved_frame':int((diff==0).sum()),'candidate_earlier_saved_frame':int((diff<0).sum()),'candidate_later_saved_frame':int((diff>0).sum()),'max_abs_saved_frame_time_difference_s':int(np.abs(diff).max()) if len(diff) else None,'not_subframe_arrival_time':True})
    masks[f'first_baseline_h{threshold}']=ax;masks[f'first_candidate_h{threshold}']=zx
np.savez_compressed(R/'full-mask-comparisons.npz',**masks)
steps0=np.loadtxt(B/'accepted_timesteps.txt',ndmin=2);steps1=np.loadtxt(C/'accepted_timesteps.txt',ndmin=2)
lines=[x.strip() for x in (C/'carlisle.bdy').read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')]
blocks={};i=0
while i<len(lines):
    name=lines[i];n,u=lines[i+1].split();n=int(n);assert u=='seconds'
    blocks[name]=np.array([[float(v) for v in x.split()] for x in lines[i+2:i+2+n]]);i+=2+n
expected=0.
for name in ['upstream1','upstream2','upstream3']:
    a=blocks[name];ts=np.r_[0,a[(a[:,1]>0)&(a[:,1]<2400),1],2400];qs=np.interp(ts,a[:,1],a[:,0]);expected+=10*float(((qs[:-1]+qs[1:])*.5*np.diff(ts)).sum())
source_error=s['source_m3']-expected
rec={'evidence_type':'same-model time-discretization sensitivity, not independent field observations','field_passes_added':0,'baseline':{'dtmax':1,'cfl':.33,'accepted_steps':len(steps0),'median_dt':float(np.median(steps0[:,1])),'minimum_dt':float(steps0[:,1].min()),'rejections':b['rejections'],'wall_s':b['wall_s']},'candidate':{'dtmax':1,'cfl':.165,'accepted_steps':len(steps1),'median_dt':float(np.median(steps1[:,1])),'minimum_dt':float(steps1[:,1].min()),'rejections':s['rejections'],'wall_s':s['wall_s']},
 'independent_source_expected_m3':expected,'source_error_m3':source_error,'source_interval_continuity':bool(np.allclose(steps1[1:,0],steps1[:-1].sum(axis=1),atol=1e-10,rtol=0)),
 'frames':frames,'first_saved_wet_frame_comparisons':first,'claims_not_supported':['No calibrated truth threshold for wetness','No two-step convergence claim','No global real-world accuracy percentage','2400s is not the full event']}
(R/'comparison.json').write_text(json.dumps(rec,indent=2))
assert abs(source_error)<1e-6 and rec['source_interval_continuity']
print(json.dumps({'baseline':rec['baseline'],'candidate':rec['candidate'],'last_frame':frames[-1],'first_saved_wet_frame_comparisons':first,'independent_source_error':source_error}),flush=True)
