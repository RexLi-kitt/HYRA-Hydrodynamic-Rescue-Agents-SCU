"""Check dependence of CP-1 outflow on smoothed reservoir storage."""
import argparse,json
from pathlib import Path
import numpy as np

def audit(source,output):
 data=json.loads((source/'series.json').read_text());meta=json.loads((source/'audit.json').read_text())
 q=np.array([np.nan if r['value'] is None else r['value'] for r in data['Qout_OR_20190424']]);t=np.array([r['time_s'] for r in data['Qout_OR_20190424']]);qin=meta['events']['inflow20190424'];result={}
 for key in ('storage20190424','storage_smooth20190424'):
  v=np.array([r['value'] for r in data[key]]);vt=np.array([r['time_s'] for r in data[key]])
  assert np.array_equal(t,vt) and np.all(np.diff(t)>0)
  err=qin-np.diff(v)/np.diff(t)-q[1:]
  result[key]={'rmse_m3_s':float(np.sqrt(np.mean(err**2))),'max_abs_m3_s':float(abs(err).max()),'samples':len(err)}
 result['negative_Q_count']=int((q<0).sum());result['negative_Q_during_breach_count']=int(((q<0)&(t>=527)&(t<=909)).sum())
 output.write_text(json.dumps(result,indent=2),encoding='utf-8');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path);a=p.parse_args();print(audit(a.source,a.output))
