from pathlib import Path
import json,heapq,struct,hashlib
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round95';C=O/'round91/full_event'
d=np.load(R/'southern-static-reachability.npz');z=d['terrain'];r0,r1,c0,c1=map(int,d['crop_row_col'])
f=C/'results/check-0005.wdb'
with f.open('rb') as s:
 nc,nr=struct.unpack('<ii',s.read(8));s.read(32);h=np.fromfile(s,dtype='<f8').reshape(nr,nc)[r0:r1,c0:c1]
eta=np.where(h>.1,z+h,-np.inf);best=eta.copy();nr,nc=z.shape;parent={};origin={}
q=[]
for y,x in zip(*np.where(np.isfinite(eta))):
 v=int(y)*nc+int(x);origin[v]=v;heapq.heappush(q,(-float(eta[y,x]),v))
while q:
 neg,v=heapq.heappop(q);head=-neg;y,x=divmod(v,nc)
 if head!=best[y,x]:continue
 for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
  if 0<=yy<nr and 0<=xx<nc and head>z[yy,xx]+.1 and head>best[yy,xx]:
   u=yy*nc+xx;best[yy,xx]=head;origin[u]=origin[v];parent[u]=v;heapq.heappush(q,(-head,u))
marks=json.loads((O/'round92/all263-diagnostic.json').read_text(encoding='utf-8'));res=[]
for m in marks:
 if m['miss_component_id']!=2031:continue
 y=m['grid_row']-r0;x=m['grid_col']-c0;v=y*nc+x
 if not np.isfinite(best[y,x]):continue
 path=[v]
 while path[-1] in parent:path.append(parent[path[-1]])
 path.reverse();rows=[]
 for u in path:
  yy,xx=divmod(u,nc);rows.append({'E':338502.5+5*(xx+c0),'N':557752.5-5*(yy+r0),'bed':float(z[yy,xx]),'depth':float(h[yy,xx]),'stage':float(z[yy,xx]+h[yy,xx]) if h[yy,xx]>.001 else None})
 first=next((i for i,r in enumerate(rows) if r['depth']<=.1),None)
 # The uppermost seed stage is retained; witness is a static accessible route, not actual flow.
 res.append({'fid':m['fid'],'file':f.name,'seed':rows[0],'distance_m':5*(len(rows)-1),'maximum_path_bed':max(r['bed'] for r in rows),'minimum_static_depth_margin_above_0_1':float(best[y,x])-max(r['bed'] for r in rows)-.1,'first_not_above_indicator':rows[first] if first is not None else None,'previous_actual_wet_cell':rows[first-1] if first is not None and first>0 else None,'rows':rows})
(R/'route-witnesses.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
print(json.dumps([{k:v for k,v in a.items() if k!='rows'} for a in res],indent=2))
