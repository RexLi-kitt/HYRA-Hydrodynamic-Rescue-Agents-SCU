from pathlib import Path
import sys,json,hashlib,urllib.request
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round89';P=R/'original'
sys.path.insert(0,str(O/'round73'))
from audit_carlisle_public_inputs import read_bdy
tree=json.loads((R/'repository-tree.json').read_text(encoding='utf-8-sig')); manifest=json.loads((R/'download-manifest.json').read_text())
names={f'Data/carlisle_run{i}.bdy' for i in range(2,10)}|{f'Data/Upstream_Flows_Run{i}.csv' for i in range(2,10)}
for x in tree['tree']:
    if x['path'] not in names: continue
    existing=P/Path(x['path']).name
    if existing.exists():
        saved=existing.read_bytes()
        assert hashlib.sha1(b'blob '+str(len(saved)).encode()+b'\0'+saved).hexdigest()==x['sha']
        continue
    url='https://raw.githubusercontent.com/SRKabir/Rapid_FloodModelling_CNN/'+tree['sha']+'/'+x['path']
    with urllib.request.urlopen(url,timeout=60) as response:data=response.read()
    blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    assert len(data)==x['size'] and blob==x['sha']
    (P/Path(x['path']).name).write_bytes(data)
    manifest.append({'path':x['path'],'url':url,'bytes':len(data),'git_blob':blob,'sha256':hashlib.sha256(data).hexdigest()})
    (R/'download-manifest.json').write_text(json.dumps(manifest,indent=2))
cs={i:np.genfromtxt(P/f'Upstream_Flows_Run{i}.csv',delimiter=',',names=True) for i in range(1,10)}
rows=[]
for i in range(1,10):
    b=read_bdy(P/f'carlisle_run{i}.bdy')
    for j,c in cs.items():
        rr=[]
        for k,mult in enumerate([55,30,30],1):
            v=b[f'upstream{k}'];t=v['time_s'];q=v['q_as_stored']*mult
            eq=np.array_equal(t,c['Time']);delta=q-c[f'Upstream{k}'] if eq else None
            rr.append({'source':k,'time_equal':eq,'matching_rows':int((abs(delta)<=1e-5).sum()) if eq else 0,'total_rows':len(t),'max_abs_delta_m3_s':float(abs(delta).max()) if eq else None})
        rows.append({'bdy_run':i,'csv_run':j,'all_three_match':all(x['time_equal'] and x['matching_rows']==x['total_rows'] for x in rr),'sources':rr})
(R/'all-run-fingerprints.json').write_text(json.dumps({'tolerance_m3_s':1e-5,'time_shift_applied':False,'rescale_other_than_documented_point_width':False,'rows':rows},indent=2))
print(json.dumps({'matches':[{'bdy':x['bdy_run'],'csv':x['csv_run']} for x in rows if x['all_three_match']],'files':len(manifest)}))
