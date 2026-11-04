"""Lower-bound terrain-path diagnostic for ALL32 compatible but missed marks."""
from pathlib import Path
import json,heapq,struct,hashlib
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round92';C=O/'round91/full_event'
z=np.loadtxt(C/'Carlisle_5m.asc',skiprows=6);d=np.load(O/'round91/full-classification.npz');wet=d['maximum_depth']>.1;mx=d['maximum_depth'];nr,nc=z.shape
marks=json.loads((R/'all263-diagnostic.json').read_text(encoding='utf8'))
targets=[a for a in marks if not a['above_main_analysis_indicator'] and a['head_compatibility']=='observed_head_above_0.1_indicator'];assert len(targets)==32
protocol={'scope':'All32 missed marks whose observed peak exceeds cell bed+0.1m; diagnostic subgroup, does not replace all263 or full outline scores.',
 'rules':'Global 4-neighbor minimax bed path to ANY cell with model max depth>0.1. Dijkstra key=(highest bed along path, steps). No diagonal jumps, no radius truncation, no DEM edits. Endpoint head uses that cell cumulative maximum only; never treats neighboring asynchronous maxima as a real flow field.',
 'interpretation':'Barrier above observed peak implies no positive-depth constant-head connection along even easiest raster route at that peak, under fixed DEM. Barrier below peak is necessary, not sufficient for flooding; path selection uses model output and is diagnosis only.',
 'limitations':'No bridge/culvert underpass, time/direction/available-volume guarantee or independent water-level uncertainty. Global result can select a physically unrelated pool and is only a lower bound.',
 'next_gate':'Need independent historical crest/underpass/blockage/flow evidence before hydro candidate; no fitting to path or survey peaks.'}
(R/'path-protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
results=[]
for a in targets:
 r,c=a['grid_row'],a['grid_col'];start=r*nc+c
 best={start:(float(z[r,c]),0)};parent={};q=[(float(z[r,c]),0,start)]
 while q:
  cost,steps,v=heapq.heappop(q)
  if best[v]!=(cost,steps):continue
  y,x=divmod(v,nc)
  if wet[y,x]:end=v;break
  for yy,xx in [(y-1,x),(y+1,x),(y,x-1),(y,x+1)]:
   if 0<=yy<nr and 0<=xx<nc:
    u=yy*nc+xx;k=(max(cost,float(z[yy,xx])),steps+1)
    if k<best.get(u,(float('inf'),10**9)):
     best[u]=k;parent[u]=v;heapq.heappush(q,(*k,u))
 path=[end]
 while path[-1]!=start:path.append(parent[path[-1]])
 path.reverse();rr=np.array(path)//nc;cc=np.array(path)%nc;er,ec=divmod(end,nc)
 peak=a['Z_masl'];crest=float(z[rr,cc].max());endstage=float(z[er,ec]+mx[er,ec]);imax=int(np.argmax(z[rr,cc]));br,bc=int(rr[imax]),int(cc[imax])
 results.append({'fid':a['fid'],'E':a['E'],'N':a['N'],'observed_peak_masl':peak,'target_bed_masl':float(z[r,c]),
   'minimum_raster_crest_masl':crest,'crest_minus_observed_peak_m':crest-peak,
   'path_length_m':(len(path)-1)*5,'endpoint_max_stage_masl':endstage,
   'endpoint_max_stage_minus_crest_m':endstage-crest,'endpoint_max_stage_minus_observed_peak_m':endstage-peak,
   'endpoint_E_N':[338502.5+5*ec,557752.5-5*er],
   'bottleneck_E_N':[338502.5+5*bc,557752.5-5*br],
   'barrier_above_observed_peak':crest>=peak,
   'raster_path_row_col_bed':[[int(y),int(x),float(z[y,x])] for y,x in zip(rr,cc)]})
(R/'all32-minimax-paths.json').write_text(json.dumps(results,indent=2),encoding='utf8')
summary={'n':len(results),'minimum_crest_at_or_above_observed_peak':sum(p['barrier_above_observed_peak'] for p in results),
 'endpoint_peak_below_observed_peak':sum(p['endpoint_max_stage_minus_observed_peak_m']<0 for p in results),
 'endpoint_peak_below_minimum_crest':sum(p['endpoint_max_stage_minus_crest_m']<0 for p in results),
 'rows':[{k:v for k,v in p.items() if k!='raster_path_row_col_bed'} for p in results],
 'physical_model_modified':False,'real_accuracy_improvement':False}
(R/'path-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary,indent=2))
