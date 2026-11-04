"""Score only a completed original window against pre-registered positive evidence.
Outside the observed polygon is unknown, not a true negative or a false positive.
"""
from pathlib import Path
import json,hashlib,struct
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round91';C=R/'full_event';res=C/'results'
st=json.loads((R/'status.json').read_text());p=json.loads((R/'protocol.json').read_text())
assert st['state']=='completed' and st['exit_code']==0 and not st['stop_reason'],'No score for failed/incomplete simulation'
sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
assert sha(O/'round90/observation-footprint-mask.npz')==p['observation_mask_sha256']
assert sha(O/'round90/observation-protocol.json')==p['observation_protocol_sha256']
def read(path):
 with path.open('rb') as f:
  nc,nr=struct.unpack('<ii',f.read(8));hd=struct.unpack('<dddd',f.read(32));a=np.fromfile(f,dtype='<f8')
 assert (nc,nr)==(951,611) and hd[:3]==(338500.,554700.,5.) and a.size==nc*nr
 return a.reshape(nr,nc)
m=np.loadtxt(res/'check.mass',skiprows=1);times=m[:,0];dt=np.diff(np.r_[0,times]);starts=np.r_[0,times[:-1]]
frames=[]
for f in sorted(res.glob('*.wdb')):
 h=read(f);frames.append({'name':f.name,'sha256':sha(f),'finite_nonnegative':bool(np.isfinite(h).all() and h.min()>=0),'volume_m3':float(h.sum()*25)})
mx=read(res/'check.maxb');massgate=bool(np.all(np.isfinite(m)) and np.all(abs(m[:,10])<=np.maximum(1e-4,1e-8*abs(m[:,5]))))
checks={'end_within_one_original_cap':245700<=float(times[-1])<=245702,'all_steps_recorded':bool(np.array_equal(m[:,3],np.arange(1,len(m)+1))),'saved_fields_finite_nonnegative':bool(frames) and all(f['finite_nonnegative'] for f in frames),'max_finite_nonnegative':bool(np.isfinite(mx).all() and mx.min()>=0),'per_step_mass_guard':massgate}
diag={'checks':checks,'actual_end_s':float(times[-1]),'steps':len(m),'frames':frames,'total_signed_mass_residual_m3':float(m[:,10].sum()),'sum_abs_mass_residual_m3':float(abs(m[:,10]).sum()),'wall_s':st['wall_s']}
(R/'numerical-verification.json').write_text(json.dumps(diag,indent=2));assert all(checks.values()),'Internal checks failed; no field score'
mask=np.load(O/'round90/observation-footprint-mask.npz')['mask'];known=mask==1;rows=[]
for th in [.01,.1,.3]:
 wet=mx>th;hit=int((wet&known).sum());miss=int((~wet&known).sum())
 rows.append({'threshold_m':th,'main_indicator':th==.1,'observed_fully_covered_cells':int(known.sum()),'model_above_indicator':hit,'model_below_or_equal_indicator':miss,'positive_coverage_fraction':hit/int(known.sum()),'edge_unknown_predicted_above':int((wet&(mask==2)).sum()),'outside_unknown_predicted_above':int((wet&(mask==0)).sum()),'overall_accuracy':None,'false_positives':None,'true_negatives':None,'scope':'positive maximum-outline coverage only; not total accuracy'})
marks=json.loads((O/'round90/all-marks-fixed-domain.json').read_text());dem=np.loadtxt(C/'Carlisle_5m.asc',skiprows=6)
for a in marks:
 col=int((a['E']-338500)//5);row=610-int((a['N']-554700)//5)
 if a['inside_fixed_grid']:
  a.update(grid_row=row,grid_col=col,sim_max_h_m=float(mx[row,col]),dem_cell_mean_m=float(dem[row,col]),model_zero_max_depth=bool(mx[row,col]==0),above_main_analysis_indicator=bool(mx[row,col]>.1))
 a['verification']='reported water-mark location diagnostic; no stage-derived depth substitution; original point/coordinate uncertainty retained'
(R/'all263-mark-comparison.json').write_text(json.dumps(marks,indent=2))
out={'evidence':'2005 reported maximum polygons vs complete-window conditional software simulation','backend':'LF8 fast acceleration; not production HYRA or DG2','rows':rows,'zero_max_depth_in_reported_fully_covered_cells':int((known&(mx==0)).sum()),'unknown_edge_cells':int((mask==2).sum()),'unknown_outside_cells':int((mask==0).sum()),'all263_marks_retained':len(marks)==263,'all_target_qualitative_requirements_pass':False,'why_not_overall_pass':'unknown dry coverage, boundary precision, initial-state realism, time-specific rise/fall, directions and route evidence remain; any main-indicator misses also retained','production_adopted':False,'original_version_replication':False}
(R/'maximum-outline-comparison.json').write_text(json.dumps(out,indent=2));np.savez_compressed(R/'full-classification.npz',observation_class=mask,maximum_depth=mx,above001=mx>.01,above010=mx>.1,above030=mx>.3)
print(json.dumps(out))
