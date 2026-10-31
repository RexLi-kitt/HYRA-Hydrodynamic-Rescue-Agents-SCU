from pathlib import Path
import json,struct
from collections import deque
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round95'
d=np.load(R/'southern-static-reachability.npz');z=d['terrain'];r0,r1,c0,c1=map(int,d['crop_row_col']);nr,nc=z.shape
with (O/'round91/full_event/results/check-0005.wdb').open('rb') as f:
 cols,rows=struct.unpack('<ii',f.read(8));f.read(32);h=np.fromfile(f,dtype='<f8').reshape(rows,cols)[r0:r1,c0:c1]
marks={a['fid']:a for a in json.loads((O/'round92/all263-diagnostic.json').read_text(encoding='utf-8'))};out=[]
for record in json.loads((R/'route-witnesses.json').read_text(encoding='utf-8')):
 level=record['seed']['stage'];m=marks[record['fid']];target=(m['grid_row']-r0,m['grid_col']-c0)
 passable=z+.1<level;starts=list(zip(*np.where((h>.1)&(z+h>=level))));queue=deque(starts);prev={s:None for s in starts}
 while queue:
  y,x=queue.popleft()
  if (y,x)==target:break
  for yy,xx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)):
   if 0<=yy<nr and 0<=xx<nc and passable[yy,xx] and (yy,xx) not in prev:prev[(yy,xx)]=(y,x);queue.append((yy,xx))
 assert target in prev
 path=[];cur=target
 while cur is not None:path.append(cur);cur=prev[cur]
 path.reverse();points=[{'E':338502.5+5*(x+c0),'N':557752.5-5*(y+r0),'bed':float(z[y,x]),'depth':float(h[y,x]),'stage':float(z[y,x]+h[y,x])} for y,x in path]
 first=next(i for i,a in enumerate(points) if a['depth']<=.1)
 crest=max(a['bed'] for a in points);out.append({'fid':m['fid'],'shortest_at_available_constant_head_m':5*(len(path)-1),'head':level,'crest':crest,'margin_above_indicator':level-crest-.1,'seed':points[0],'first_below_indicator':points[first],'previous_wet_cell':points[first-1],'rows':points})
(R/'shortest-static-witnesses.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps([{k:v for k,v in a.items() if k!='rows'} for a in out],indent=2))
