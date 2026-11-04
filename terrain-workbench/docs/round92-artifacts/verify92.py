from pathlib import Path
import numpy as np,json,hashlib
from scipy import ndimage as ndi
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round92'
b=np.load(O/'round91/full-classification.npz');a=np.load(R/'all-miss-diagnostics.npz');z=np.loadtxt(O/'round91/full_event/Carlisle_5m.asc',skiprows=6)
wet=b['maximum_depth']>.1;miss=(b['observation_class']==1)&~wet;s=ndi.generate_binary_structure(2,1)
cs=json.loads((R/'all-miss-components.json').read_text());paths=json.loads((R/'all32-minimax-paths.json').read_text());marks=json.loads((R/'all263-diagnostic.json').read_text());byid={p['fid']:p for p in marks}
checks={'all24884_misses_identical':bool(np.array_equal(miss,a['miss'])),'all263_unique_ids_retained':len(byid)==263,'component_counts_match_independent_bincount':all(int(np.bincount(a['component4'].ravel())[p['id']])==p['cells'] for p in cs),'all_component_cells_accounted':sum(p['cells'] for p in cs)==24884}
pathchecks=[]
for p in paths:
 pts=np.array(p['raster_path_row_col_bed']);rr=pts[:,0].astype(int);cc=pts[:,1].astype(int);crest=p['minimum_raster_crest_masl'];target=byid[p['fid']];r,c=target['grid_row'],target['grid_col']
 allowed=z<=crest
 reachable=ndi.binary_propagation(wet&allowed,structure=s,mask=allowed)
 below=z<crest
 reachable_below=ndi.binary_propagation(wet&below,structure=s,mask=below)
 pathchecks.append({'fid':p['fid'],'cardinal_path':bool(np.all(np.abs(np.diff(rr))+np.abs(np.diff(cc))==1)),
 'starts_at_original_point':bool(rr[0]==r and cc[0]==c),'ends_in_model_wet':bool(wet[rr[-1],cc[-1]]),
 'bed_and_crest_exact':bool(np.array_equal(pts[:,2],z[rr,cc]) and z[rr,cc].max()==crest),
 'independent_threshold_connectivity_proves_minimum':bool(reachable[r,c] and not reachable_below[r,c])})
checks['all32_paths_independently_valid']=len(pathchecks)==32 and all(all(v for k,v in p.items() if k!='fid') for p in pathchecks)
protocol=json.loads((R/'protocol.json').read_text(encoding='utf-8'))
checks['all_inputs_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in protocol['input_sha256'].items())
wetmarks=[p for p in marks if p['above_main_analysis_indicator']]
resid=[p['dem_cell_mean_m']+p['sim_max_h_m']-p['Z_masl'] for p in wetmarks]
output={'checks':checks,'paths':pathchecks,'wet137_survey_stage_diagnostic':{'count':len(resid),'below_observed':sum(d<0 for d in resid),'median_model_minus_observed_m':float(np.median(resid)),'range_m':[float(min(resid)),float(max(resid))],'meaning':'Internal cause diagnostic at137 model-wet marks only; not a new primary score or overall bias estimate.'},'no_hydrodynamic_rerun':True,'old_mass_and_stability_checks':'Inherited unchanged hashed round91 run; no repeat required.'}
(R/'independent-verification.json').write_text(json.dumps(output,indent=2));assert all(checks.values())
print(json.dumps({k:v for k,v in output.items() if k!='paths'}))
