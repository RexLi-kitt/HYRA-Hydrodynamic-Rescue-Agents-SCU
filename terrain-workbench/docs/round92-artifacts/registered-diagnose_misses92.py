"""Frozen-result diagnosis only. Never edits solver inputs or observed labels."""
from pathlib import Path
import json, hashlib, struct, sys
from datetime import datetime, timezone
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
R=O/'round92'; B=O/'round91'; C=B/'full_event'
R.mkdir(exist_ok=True)
def write(n,x):
    (R/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs=[B/'full-classification.npz',C/'Carlisle_5m.asc',B/'all263-mark-comparison-with-source.json',C/'carlisle.bci',B/'protocol.json']+sorted((C/'results').glob('*.wdb'))
protocol={
 'created_utc':datetime.now(timezone.utc).isoformat(),
 'purpose':'Diagnose every pre-existing miss; no new acceptance rule, simulation, calibration or input edit.',
 'baseline':'round91 complete 245700s, original 5m geometry and positive-outline mask',
 'rules':[
  'Main miss = original observation_class==1 and maximum_depth<=0.1; retain all 24884.',
  'Label all 4-connected miss components, retaining every size; 8-connectivity is diagnostic sensitivity only.',
  'Report exact Euclidean distance to any simulated maximum_depth>0.1 cell; such cells need not be observed wet.',
  'Topological holes = cells enclosed by simulated maximum-wet set using 4-neighbor background. Enclosed is not evidence of a building.',
  'Record terrain minus nearest maximum-wet cell terrain continuously, without declaring a causal threshold.',
  'For each missed cell with an immediate wet neighbor in a saved frame, test whether that SAME FRAME neighbor z+h exceeds target z+0.1. This is pressure/head opportunity only, not proof of flow or a conservation error.',
  'Retain all263 marks. Cross terrain incompatibility Z<cell bed, insufficient observed head Z<=bed+0.1, and compatible head Z>bed+0.1.',
  'At every missed mark, test 4-connected bed+0.1<observed peak Z paths within a fixed 100m square radius to any simulated maximum-wet cell. Static necessary-geometry diagnostic, NOT a hydraulic prediction; water maxima are asynchronous.',
  'No terrain lowering, point moving, observation cropping or threshold optimization. All grouping is post-result diagnosis, not independent validation.'
 ],
 'success':'Account for all misses and identify reproducible causes or bounded hypotheses for a single-factor follow-up.',
 'rejection':'No physical intervention is adopted from topology alone. Missing independent geometry remains a data gap.',
 'cost':'Static arrays/12 saved frames, expected <2 minutes; no heavy solver.',
 'input_sha256':{str(p):sha(p) for p in inputs},
 'script_sha256':sha(__file__)
}
if not (R/'protocol.json').exists(): write('protocol.json',protocol)
else:
    old=json.loads((R/'protocol.json').read_text(encoding='utf-8'))
    assert old['input_sha256']==protocol['input_sha256'], 'Inputs changed'
if '--register-only' in sys.argv:
    print('Protocol registered; no arrays scored.');sys.exit(0)

data=np.load(B/'full-classification.npz'); obs=data['observation_class']; mx=data['maximum_depth']
z=np.loadtxt(C/'Carlisle_5m.asc',skiprows=6); wet=mx>.1; miss=(obs==1)&~wet
assert miss.sum()==24884 and (obs==1).sum()==217464
s4=ndi.generate_binary_structure(2,1);s8=ndi.generate_binary_structure(2,2)
labels,n=ndi.label(miss,s4);labels8,n8=ndi.label(miss,s8)
dist,indices=ndi.distance_transform_edt(~wet,sampling=5,return_indices=True)
nearest_z=z[indices[0],indices[1]]
enclosed=ndi.binary_fill_holes(wet,structure=s4)&~wet
obsedge=ndi.binary_dilation(obs!=1,structure=s4)
def stats(a):
    a=np.asarray(a);return {'n':int(a.size),'min':float(a.min()),'median':float(np.median(a)),'p90':float(np.quantile(a,.9)),'max':float(a.max())} if a.size else {'n':0}
def readbin(p):
    with p.open('rb') as f:
        nc,nr=struct.unpack('<ii',f.read(8));hd=struct.unpack('<dddd',f.read(32));a=np.fromfile(f,dtype='<f8')
    assert (nr,nc)==z.shape and hd[:3]==(338500.,554700.,5.)
    return a.reshape(z.shape)
opportunity=np.zeros(z.shape,bool); margin=np.full(z.shape,-np.inf); counts=np.zeros(z.shape,np.uint8);framerecord=[]
for f in sorted((C/'results').glob('*.wdb')):
    h=readbin(f); eta=np.where(h>.1,z+h,-np.inf)
    # Explicit cardinal footprint to avoid including the central cell.
    cross=s4.copy();cross[1,1]=False
    adjacent=ndi.maximum_filter(eta,footprint=cross,mode='constant',cval=-np.inf)
    excess=adjacent-z-.1;flag=miss&(excess>0)
    opportunity|=flag;counts+=flag; margin=np.maximum(margin,excess)
    framerecord.append({'file':f.name,'missed_cells_with_adjacent_same_time_head_above_indicator':int(flag.sum())})

components=[]
for i,sl in enumerate(ndi.find_objects(labels),1):
    local=labels[sl]==i;rr,cc=np.where(local);rr+=sl[0].start;cc+=sl[1].start
    components.append({'id':i,'cells':len(rr),'area_m2':len(rr)*25,
      'bbox_E_N':[338500+5*int(cc.min()),554700+5*(610-int(rr.max())),338500+5*(int(cc.max())+1),554700+5*(611-int(rr.min()))],
      'centroid_E_N':[float(338502.5+5*cc.mean()),float(557752.5-5*rr.mean())],
      'touches_observation_unknown':bool(obsedge[rr,cc].any()),
      'touches_grid_edge':bool(((rr==0)|(cc==0)|(rr==610)|(cc==950)).any()),
      'enclosed_by_simulated_max_wet_cells':int(enclosed[rr,cc].sum()),
      'zero_recorded_max_cells':int((mx[rr,cc]==0).sum()),
      'nearest_simulated_wet_distance_m':stats(dist[rr,cc]),
      'bed_minus_nearest_wet_bed_m':stats(z[rr,cc]-nearest_z[rr,cc]),
      'same_time_adjacent_head_opportunity_cells':int(opportunity[rr,cc].sum())})
components.sort(key=lambda a:(-a['cells'],a['id']));write('all-miss-components.json',components)
marks=json.loads((B/'all263-mark-comparison-with-source.json').read_text(encoding='utf-8'))
for a in marks:
    r,c=a['grid_row'],a['grid_col'];head=a['Z_masl']-z[r,c]
    a.update(miss_component_id=int(labels[r,c]),distance_to_simulated_wet_m=float(dist[r,c]),
      observed_peak_minus_cell_bed_m=float(head),enclosed_by_simulated_max_wet=bool(enclosed[r,c]),
      same_time_adjacent_head_opportunity=bool(opportunity[r,c]),
      head_compatibility='peak_below_grid_terrain' if head<0 else ('observed_head_not_above_0.1_indicator' if head<=.1 else 'observed_head_above_0.1_indicator'))
    if not a['above_main_analysis_indicator']:
      sl=(slice(max(0,r-20),min(611,r+21)),slice(max(0,c-20),min(951,c+21)))
      passable=z[sl]+.1<a['Z_masl'];ids,k=ndi.label(passable,s4);target=int(ids[r-sl[0].start,c-sl[1].start])
      reachable=(ids==target) if target else np.zeros(ids.shape,bool)
      a['static_100m_path_to_any_simulated_max_wet_at_observed_peak']=bool((reachable&wet[sl]).any())
      a['static_path_limitation']='Asynchronous peak and rectangular 100m neighborhood; no flow direction, available volume or travel-time guarantee.'
write('all263-diagnostic.json',marks)
failed=[a for a in marks if not a['above_main_analysis_indicator']]
summary={
 'baseline_positive_coverage_unchanged':192580/217464,'all_misses_retained':int(miss.sum()),'components4':n,'components8':n8,
 'component_sizes_top20':[a['cells'] for a in components[:20]],
 'top10_fraction_of_misses':sum(a['cells'] for a in components[:10])/int(miss.sum()),
 'enclosed_miss_cells':int((miss&enclosed).sum()),
 'distance_to_wet_m':stats(dist[miss]),
 'distance_bins':{str(d):int((miss&(dist<=d)).sum()) for d in [5,10,25,50,100]},
 'bed_above_nearest_wet_bed_m':stats((z-nearest_z)[miss]),
 'same_time_adjacent_head_opportunity_cells':int(opportunity.sum()),
 'same_time_opportunity_margin_m':stats(margin[opportunity]),'per_frame':framerecord,
 'all263_retained':len(marks),'failed_marks':len(failed),
 'failed_marks_head_categories':{v:sum(a['head_compatibility']==v for a in failed) for v in ['peak_below_grid_terrain','observed_head_not_above_0.1_indicator','observed_head_above_0.1_indicator']},
 'failed_marks_static_path_exists':sum(a['static_100m_path_to_any_simulated_max_wet_at_observed_peak'] for a in failed),
 'new_simulations':0,'production_modified':False,'accuracy_improved':False,
 'limits':['Topology does not identify buildings or prove defective terrain.','Same-frame adjacent head only diagnoses potential local impediments; local-inertia hydraulic flux and barriers still need examination.','Original known-wet polygon may cover features without explicit holes; no independent building exclusion is available.','No new dry-region, road, direction or timing verification.']}
assert sum(a['cells'] for a in components)==24884
write('summary.json',summary)
np.savez_compressed(R/'all-miss-diagnostics.npz',component4=labels,component8=labels8,miss=miss,enclosed=enclosed,nearest_wet_distance_m=dist,same_time_head_opportunity=opportunity,same_time_head_frame_count=counts)
fig,axs=plt.subplots(1,2,figsize=(16,7),layout='constrained')
extent=[338500,343255,554700,557755]
axs[0].imshow(np.where(wet,1,np.nan),extent=extent,origin='upper',cmap='Blues',vmin=0,vmax=2)
im=axs[0].imshow(np.where(miss,dist,np.nan),extent=extent,origin='upper',cmap='plasma',vmin=0)
fig.colorbar(im,ax=axs[0],label='Miss distance to any simulated max-wet cell (m)',shrink=.65)
for a in components[:10]:
    x,y=a['centroid_E_N'];axs[0].text(x,y,str(a['id']),fontsize=8,bbox={'facecolor':'white','alpha':.8,'pad':1})
axs[0].set_title('All 24,884 missed known-wet cells; top10 component IDs')
axs[1].imshow(z,extent=extent,origin='upper',cmap='terrain',vmin=10,vmax=35)
for key,color,label in [('peak_below_grid_terrain','red','78: peak below grid terrain'),('observed_head_not_above_0.1_indicator','orange','Observed head <=0.1m'),('observed_head_above_0.1_indicator','cyan','Observed head >0.1m')]:
    pts=[a for a in failed if a['head_compatibility']==key]
    if pts:axs[1].scatter([a['E'] for a in pts],[a['N'] for a in pts],c=color,s=20,edgecolors='black',linewidths=.4,label=label)
axs[1].legend(loc='lower center',bbox_to_anchor=(.5,-.24),fontsize=8)
axs[1].set_title('All126 failed survey locations on original terrain')
for ax in axs:
    ax.set_xlabel('Easting EPSG:27700 (m)');ax.set_ylabel('Northing (m)');ax.ticklabel_format(style='plain',useOffset=False);ax.set_aspect('equal')
fig.suptitle('Round92: diagnosis only; no points removed, no model or acceptance changes')
fig.savefig(R/'miss-diagnosis.png',dpi=160);plt.close(fig)
write('artifact-hashes.json',{p.name:sha(p) for p in R.iterdir() if p.is_file() and p.name!='artifact-hashes.json'})
print(json.dumps(summary,ensure_ascii=False))
