"""Bounded synthetic capacity probe; these are not Eden simulation results."""
from pathlib import Path
import argparse
import ctypes
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

LAB=Path(__file__).resolve().parents[1]
WORK=LAB/'.runtime/lisflood-comparison'
sys.path.insert(0,str(WORK/'python-packages'))
sys.path.insert(0,str(LAB))

def memory():
    class P(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    p=P();p.cb=ctypes.sizeof(p)
    ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    get=ctypes.windll.psapi.GetProcessMemoryInfo
    get.argtypes=[ctypes.c_void_p,ctypes.POINTER(P),ctypes.c_ulong]
    if not get(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(p),p.cb):
        raise ctypes.WinError()
    return dict(peak_working_set_bytes=p.PeakWorkingSetSize,private_commit_bytes=p.PagefileUsage,peak_private_commit_bytes=p.PeakPagefileUsage)

def worker(n):
    import numpy as np
    from flood_solver import FloodSolver
    # Trigger JIT outside measured integration; same production implementation.
    warm=FloodSolver(np.zeros((16,16)),np.full((16,16),.02),40,max_timestep=30)
    warm.step(.01)
    start=time.perf_counter()
    bed=np.broadcast_to(np.linspace(1,0,n),(n,n)).copy()
    depth=np.full((n,n),.02);depth[:,n//2:n//2+max(2,n//100)]=2.
    dx=20 if n==2500 else 40
    s=FloodSolver(bed,depth,dx,manning=.075,max_timestep=30,max_steps=1000000)
    s.mx[:]=.2*s.h
    allocation=time.perf_counter()-start
    rates=[];dts=[]
    for _ in range(3):
        dt=s.timestep(limit=30,rain=5/1000/3600)
        t=time.perf_counter();s.step(dt,rain=5/1000/3600);rates.append(time.perf_counter()-t);dts.append(dt)
    result=dict(n=n,cells=n*n,dx_m=dx,backend='numba' if s.acceleration else 'numpy',allocation_wall_s=allocation,
        wall_per_step_s=rates,median_wall_per_step_s=statistics.median(rates),dt_s=dts,**memory(),
        finite=bool(np.isfinite(s.h).all()),nonnegative=bool((s.h>=0).all()),
        source_sha256=hashlib.sha256((LAB/'flood_solver.py').read_bytes()).hexdigest(),
        scope='synthetic thin water plus channel, 3 RK2 steps, excludes JIT; no Eden input or full-event claim')
    del s,depth,bed
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int);a=p.parse_args()
    if a.worker:
        r=worker(a.worker);out=WORK/'capacity';out.mkdir(parents=True,exist_ok=True)
        (out/f'{a.worker}.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r),flush=True);return
    out=WORK/'capacity';out.mkdir(parents=True,exist_ok=True)
    for n in [128,256,512,768,1250]:
        r=subprocess.run([sys.executable,__file__,'--worker',str(n)],capture_output=True,timeout=180)
        print(r.stdout.decode('utf-8',errors='replace'),flush=True)
        if r.returncode:
            print(r.stderr.decode('utf-8',errors='replace')[-1500:],flush=True);break

if __name__=='__main__':main()
