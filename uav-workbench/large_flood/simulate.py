"""Deterministic stripe scan and time-stamped, causal water observations."""
import argparse
import base64
import json
import math
import time
from pathlib import Path
import numpy as np
from .input import FloodInput

def peak_memory_bytes():
    if __import__('os').name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),
                        ('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),
                        ('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
        value=Counters(); value.cb=ctypes.sizeof(value)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        get_info=ctypes.WinDLL('psapi',use_last_error=True).GetProcessMemoryInfo
        get_info.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        ok=get_info(kernel.GetCurrentProcess(),ctypes.byref(value),value.cb)
        return int(value.PeakWorkingSetSize) if ok else None
    import resource
    import sys
    size=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(size if sys.platform == 'darwin' else size*1024)


def waypoints(width, height, count, footprint):
    """Separate x-stripes; each route is an ordered continuous lawnmower."""
    routes=[]
    for i in range(count):
        left=width*i/count; right=width*(i+1)/count
        lanes=max(1, math.ceil((right-left)/footprint))
        xs=[left+(j+.5)*(right-left)/lanes for j in range(lanes)]
        points=[]
        for j,x in enumerate(xs):
            points.extend([(x,0. if j%2==0 else height), (x,height if j%2==0 else 0.)])
        routes.append(points)
    return routes


def segments_close(a,b,c,d, threshold):
    # Minimum separation of simultaneously moving points during this substep.
    relative=np.asarray(a)-np.asarray(c); velocity=(np.asarray(b)-np.asarray(a))-(np.asarray(d)-np.asarray(c))
    denom=float(velocity@velocity)
    t=0. if denom==0 else float(np.clip(-(relative@velocity)/denom,0,1))
    return float(np.linalg.norm(relative+t*velocity)) < threshold-1e-9

def advance_waypoints(position, route, cursor, distance):
    p=np.asarray(position, dtype=float).copy(); j=cursor; remaining=distance
    while remaining > 1e-9 and j<len(route):
        target=np.asarray(route[j],dtype=float); delta=target-p; length=float(np.linalg.norm(delta))
        if length <= remaining+1e-10:
            p=target.copy();remaining-=length;j+=1
        else:
            p+=delta*(remaining/length);remaining=0.
    return p,j,distance-remaining


def run(source, agents=8, speed=8., footprint=25., duration=1800., step=5., shape=(240,240), dx=5.):
    if agents not in (4,8,12) or speed not in (5.,8.,12.) or footprint not in (25.,50.,100.):
        raise ValueError('机数、速度或感知宽度不在登记选项')
    src=FloodInput(source,shape=shape,dx=dx,duration=duration)
    if not math.isclose(duration/step,round(duration/step)): raise ValueError('时间必须是决策周期的整数倍')
    start=time.perf_counter(); height,width=np.array(shape)*dx
    routes=waypoints(width,height,agents,footprint)
    positions=np.array([r[0] for r in routes],dtype=float); cursors=[1]*agents
    last=np.full(shape,-1.,dtype=float); observed=np.zeros(shape,dtype=bool); labels=np.zeros(shape,dtype=bool)
    counts=np.zeros(shape,dtype=np.uint32); previous_bounds=[None]*agents
    trajectories=[]; observations=[]; metrics=[]; distance=np.zeros(agents); collisions=0

    def observe(t):
        index, frame_time, depth=src.frame(t)
        for i,(x,z) in enumerate(positions):
            # Cell-centre inclusion in the square footprint, clipped to map.
            c0=max(0, math.ceil((x-footprint/2)/dx-.5)); c1=min(shape[1], math.floor((x+footprint/2)/dx-.5)+1)
            r0=max(0, math.ceil((z-footprint/2)/dx-.5)); r1=min(shape[0], math.floor((z+footprint/2)/dx-.5)+1)
            sl=np.s_[r0:r1,c0:c1]; observed[sl]=True; counts[sl]+=1; labels[sl]=depth[sl]>=.05; last[sl]=frame_time
            if previous_bounds[i] is not None:
                pr0,pr1,pc0,pc1=previous_bounds[i]
                ar0,ar1,ac0,ac1=max(r0,pr0),min(r1,pr1),max(c0,pc0),min(c1,pc1)
                if ar0<ar1 and ac0<ac1: counts[ar0:ar1,ac0:ac1]-=1
            previous_bounds[i]=(r0,r1,c0,c1)
            bits=base64.b64encode(np.packbits(labels[sl].ravel(), bitorder='big').tobytes()).decode('ascii')
            depth_bytes=base64.b64encode(np.asarray(depth[sl],dtype='<f4').tobytes()).decode('ascii')
            observations.append(dict(t=round(t,6),agent=i,x=round(float(x),4),z=round(float(z),4),frame=index,frame_time=frame_time, bounds=[r0,r1,c0,c1],wet_bits=bits,depth_f32_le=depth_bytes))

    observe(0.)
    trajectories.append(dict(t=0.,positions=positions.tolist()))
    for k in range(int(round(duration/step))):
        t0=k*step
        substeps=math.ceil(speed*step/dx); dt=step/substeps
        for sub in range(substeps):
            proposals=positions.copy(); next_cursors=cursors.copy(); travelled=np.zeros(agents)
            for i in range(agents):
                proposals[i], next_cursors[i], travelled[i]=advance_waypoints(positions[i], routes[i], cursors[i], speed*dt)
            rejected=set()
            for i in range(agents):
                for j in range(i):
                    if segments_close(positions[i],proposals[i],positions[j],proposals[j],dx): rejected.update((i,j))
            # Conservatively stop the entire simultaneous movement if a conflict occurs.
            if rejected:
                collisions+=1; proposals=positions.copy(); next_cursors=cursors.copy(); travelled[:]=0
            distance+=travelled; positions=proposals; cursors=next_cursors
            observe(t0+(sub+1)*dt)
        t=(k+1)*step; trajectories.append(dict(t=t,positions=positions.tolist()))
        _,_,truth=src.frame(t); correct=(labels==(truth>=.05)) & observed
        metrics.append(dict(t=t,coverage=float(observed.mean()),correct_coverage=float(correct.mean()),
                            mean_age_s=float((t-last[observed]).mean()) if observed.any() else None))
    elapsed=time.perf_counter()-start
    summary=dict(coverage=float(observed.mean()),current_correct_coverage_mean=float(np.mean([m['correct_coverage'] for m in metrics])),
                 mean_observed_age_s=float(np.mean([m['mean_age_s'] for m in metrics])),
                 repeated_cell_fraction=float((counts>1).mean()), total_distance_m=float(distance.sum()),
                 distance_per_agent_m=distance.tolist(), collision_stops=collisions, elapsed_s=elapsed,
                 peak_process_memory_bytes=peak_memory_bytes())
    return dict(schema='hyra-large-flood-rule-v1', config=dict(shape=shape,dx_m=dx,agents=agents,speed_m_s=speed,
        footprint_m=footprint,duration_s=duration,step_s=step,threshold_m=.05),source_sha256=src.sha256,
        summary=summary,trajectories=trajectories, observations=observations,metrics=metrics)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--agents',type=int,default=8,choices=[4,8,12]);p.add_argument('--speed',type=float,default=8,choices=[5,8,12]);p.add_argument('--footprint',type=float,default=25,choices=[25,50,100])
    a=p.parse_args(); result=run(a.input,a.agents,a.speed,a.footprint)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,allow_nan=False)
    print(json.dumps(result['summary'],ensure_ascii=False))

if __name__=='__main__': main()
