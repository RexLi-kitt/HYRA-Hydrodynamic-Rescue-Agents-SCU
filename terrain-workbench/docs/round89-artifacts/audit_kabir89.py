from pathlib import Path
import sys,json,hashlib,collections
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round89';P=R/'original'
sys.path.insert(0,str(O/'round73'))
from audit_carlisle_public_inputs import read_bdy
b=read_bdy(P/'carlisle_run1.bdy'); old=read_bdy(O/'round73/original/Carlisle/carlisle.bdy')
csv=np.genfromtxt(P/'Upstream_Flows_Run1.csv',delimiter=',',names=True)
with (P/'Carlisle_5m.asc').open() as f: h={k.lower():float(v) for k,v in [f.readline().split() for _ in range(6)]}
z=np.loadtxt(P/'Carlisle_5m.asc',skiprows=6); valid=z!=h['nodata_value']
counts=collections.Counter(); pts=[]; boundary=[]
for line in (P/'carlisle.bci').read_text().splitlines():
    tok=line.split()
    if not tok:continue
    if tok[0]!='P':boundary.append(line);continue
    _,x,y,typ,name=tok;x=float(x);y=float(y);counts[name]+=1
    col=int(np.floor((x-h['xllcorner'])/h['cellsize']));row=int(h['nrows'])-1-int(np.floor((y-h['yllcorner'])/h['cellsize']))
    inside=0<=row<z.shape[0] and 0<=col<z.shape[1]
    pts.append(dict(name=name,x=x,y=y,row=row,col=col,inside=inside,valid=bool(valid[row,col]) if inside else False))
rows=[];arrays={}
for i,name in enumerate(['upstream1','upstream2','upstream3'],1):
    t=b[name]['time_s']; q=b[name]['q_as_stored']*counts[name]*h['cellsize'];cq=csv['Upstream'+str(i)];ct=csv['Time']
    assert np.array_equal(t,ct)
    d=q-cq;oq=old[name]['q_as_stored']*10
    item={'name':name,'point_count':counts[name],'rows':len(t),'dt_s':np.unique(np.diff(t)).tolist(),'end_s':float(t[-1]),'bdy_peak_m3_s':float(q.max()),'bdy_peak_s':float(t[q.argmax()]),'csv_peak_m3_s':float(cq.max()),'csv_peak_s':float(t[cq.argmax()]),'bdy_vs_csv_equal_within_1e-5_m3_s':int((abs(d)<=1e-5).sum()),'bdy_vs_csv_different_rows':int((abs(d)>1e-5).sum()),'max_difference_m3_s':float(abs(d).max()),'bdy_initial_m3_s':float(q[0]),'csv_initial_m3_s':float(cq[0]),'csv_vs_rong10m_different_rows':int((abs(cq-oq)>1e-5).sum()),'difference_rows':[{'index':int(j),'t_s':float(t[j]),'bdy_total_m3_s':float(q[j]),'csv_m3_s':float(cq[j])} for j in np.flatnonzero(abs(d)>1e-5)]}
    rows.append(item);arrays[name+'_bdy']=q;arrays[name+'_csv']=cq;arrays[name+'_time']=t
result={'evidence_type':'original model inputs, not new real observations','source_revision':'eadfc848a6d37cb3476da2fe6ec7087262d9ef54','grid':h,'dem_checks':{'shape_correct':z.shape==(int(h['nrows']),int(h['ncols'])),'valid_finite':bool(np.isfinite(z[valid]).all()),'nodata_cells':int((~valid).sum()),'min_m':float(z[valid].min()),'max_m':float(z[valid].max()),'all_23_sources_inside_valid':all(p['inside'] and p['valid'] for p in pts),'unique_source_cells':len(set((p['row'],p['col']) for p in pts))},'source_points':pts,'boundary':boundary,'series':rows,'unit_assumption':'standard floodplain QVAR unit-width values times 5m times source count; archived acceleration+sgc_enable combination still requires source-code path verification','source_conflict':any(x['bdy_vs_csv_different_rows'] for x in rows),'publication':'https://arxiv.org/pdf/2006.11555v2','publication_start_claim':'Figure4 identifies 2005 hydrographs t0 as 2005-01-07 00:00; cannot assign this to conflicting bdy or other archive without full lineage','input_adopted':False,'simulation_started':False,'field_passes_added':0}
(R/'input-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
np.savez_compressed(R/'unaltered-series-comparison.npz',**arrays)
print(json.dumps({k:v for k,v in result.items() if k not in ['source_points','series']},ensure_ascii=False))
for x in rows:print(json.dumps({k:v for k,v in x.items() if k!='difference_rows'}))
