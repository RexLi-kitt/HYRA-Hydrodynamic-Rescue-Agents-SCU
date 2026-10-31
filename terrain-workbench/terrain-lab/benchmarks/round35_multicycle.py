import argparse,json
from pathlib import Path
from round33_two_dimensional import bowl
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
 for cfl in (.4,.2):
  for kind in ('native','baseline','candidate'):
   r=bowl(kind,.1,.02,a.output,cycles=3,cfl=cfl);rows.append(r)
   (a.output/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
   print(kind,cfl,r['records'][-1],flush=True)
