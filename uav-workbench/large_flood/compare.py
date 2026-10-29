"""Run the preregistered development grid; output summaries, never formal results."""
import argparse
import json
import time
from pathlib import Path
from .input import FloodInput
from .simulate import run

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    start=time.perf_counter(); f=FloodInput(a.input); f.frame(0); read=time.perf_counter()-start
    rows=[]
    for n in (4,8,12):
        for speed in (5.,8.,12.):
            for width in (25.,50.,100.):
                out=run(a.input,n,speed,width)
                rows.append(dict(agents=n,speed_m_s=speed,footprint_m=width,**out['summary']))
                print(f'{len(rows)}/27 agents={n} speed={speed:g} width={width:g}',flush=True)
    result=dict(schema='hyra-large-flood-development-comparison-v1',source_sha256=f.sha256,
                source_bytes=a.input.stat().st_size,initial_read_and_validation_s=read,
                scope='single synthetic uncalibrated development event; not independent confirmation',rows=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as out: json.dump(result,out,ensure_ascii=False,indent=2,allow_nan=False)
if __name__=='__main__': main()
