"""64 rule-policy episodes to measure environment cost before any learner run."""
import argparse,ctypes,json,time
from pathlib import Path
import numpy as np
from .env import Env,rule_action
from .train import aggregate

def peak_rss():
    class Info(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    info=Info();info.cb=ctypes.sizeof(info)
    ctypes.windll.kernel32.GetCurrentProcess.restype=ctypes.c_void_p
    getinfo=ctypes.windll.psapi.GetProcessMemoryInfo
    getinfo.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
    if not getinfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(info),info.cb):
        raise OSError('Unable to measure memory; preflight is incomplete')
    return int(info.PeakWorkingSetSize)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    manifest=json.loads((a.data/'dataset.json').read_text());cases=[r for r in manifest['cases'] if r['split']=='train']
    arrays=[np.load(r['path'],mmap_mode='r') for r in cases];rows=[];start=time.perf_counter()
    for i in range(64):
        e=Env(arrays[i%len(arrays)],31000+i);done=False
        while not done:_,_,done,_=e.step(rule_action(e))
        rows.append(dict(episode=i,**e.metrics()))
    elapsed=time.perf_counter()-start
    result=dict(episodes=rows,summary=aggregate(rows),elapsed_s=elapsed,joint_steps=sum(r['ST'] for r in rows),
                peak_rss_bytes=peak_rss(),scope='rule-policy cost and task check; no trained model')
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    if result['peak_rss_bytes']>8*1024**3:raise RuntimeError('Memory preflight exceeded 8 GiB')
    if elapsed>1200:raise RuntimeError('64-episode environment cost exceeded pilot allowance')
