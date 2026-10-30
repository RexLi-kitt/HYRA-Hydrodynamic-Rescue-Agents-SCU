"""Read only CP_1 archive; preserve nulls and source sensor time origin."""
import argparse,io,json,zipfile,hashlib
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def audit(folder):
 archive=folder/'CP_1.zip'
 assert hashlib.md5(archive.read_bytes()).hexdigest()=='dc031343e2a3f57220d09bba73b8c891'
 with zipfile.ZipFile(archive) as z:
  data=loadmat(io.BytesIO(z.read('CP_1/0.discharge/Qdata_20190424.mat')),simplify_cells=True)
  note=z.read('CP_1/2.DEMmodel/readme.txt').decode()
 summary={'source':'https://zenodo.org/records/17828278','evidence':'large-scale experiment, not real flood event','dem_note':note,'time_origin':'sensor recording start; do not shift to overtopping','solver_ready':False,'series':{}}
 payload={}
 for key in ('WL20190424','Qout_OR_20190424','storage20190424','storage_smooth20190424'):
  a=np.asarray(data[key]);t=a[:,0];v=a[:,1];assert np.isfinite(t).all() and np.all(np.diff(t)>0)
  summary['series'][key]={'count':len(t),'first_s':float(t[0]),'last_s':float(t[-1]),'intervals_s':np.unique(np.diff(t)).tolist(),'missing':int((~np.isfinite(v)).sum()),'min_finite':float(v[np.isfinite(v)].min()),'max_finite':float(v[np.isfinite(v)].max())}
  payload[key]=[{'time_s':float(x),'value':float(y) if np.isfinite(y) else None} for x,y in a]
 summary['events']={key:float(data[key]) for key in ('overtopping20190424','endtime20190424','peak_Q_time20190424','peak_Q20190424','inflow20190424')}
 summary['units']={'WL':'m relative to initial sensor reference, per README','Q':'cms as written in README, physical interpretation to verify before coupling','storage':'m3 per README'}
 (folder/'series.json').write_text(json.dumps(payload,indent=2,allow_nan=False),encoding='utf-8')
 (folder/'audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8');return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();print(json.dumps(audit(a.folder),indent=2))
