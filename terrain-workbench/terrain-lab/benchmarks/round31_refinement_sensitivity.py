"""Single-factor refinement extent and timestep study; synthetic evidence only."""
import argparse,json,time
from pathlib import Path
from round30_fine_terrain import run

CASES={'original':(3,9,4,10),'buffer':(1,11,2,12),'interior':(1,11,1,15),'all':(0,12,0,16)}
def main(output):
 rows=[]
 for dt in (.02,.01):
  for name,bounds in [*CASES.items(),('uniform',None)]:
   started=time.perf_counter()
   row=run('uniform' if name=='uniform' else 'native',bounds=bounds or CASES['all'],max_dt=dt)
   row.update(extent=name,max_dt=dt,spatial_order=2,elapsed_s=time.perf_counter()-started)
   rows.append(row)
   output.write_text(json.dumps(rows,indent=2),encoding='utf-8')
   print(name,dt,row['downstream_volume_m3'],flush=True)
 for name in ('all','uniform'):
  started=time.perf_counter()
  row=run('uniform' if name=='uniform' else 'native',bounds=CASES['all'],max_dt=.01,spatial_order=1)
  row.update(extent=name,max_dt=.01,spatial_order=1,elapsed_s=time.perf_counter()-started)
  rows.append(row)
  output.write_text(json.dumps(rows,indent=2),encoding='utf-8')
  print('first_order',name,row['downstream_volume_m3'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();main(a.output)
