"""Local hydraulic pulses and causal lead-information features (offline study)."""
import numpy as np
from .data import current_forecast,observation


XY=np.array([(5+10*x,5+10*y) for y in range(10) for x in range(10)])
ROWS=(1,2,3,6,7,8)
SOURCES=[10*y for y in ROWS]
TARGETS=[10*y+x for x in (3,5) for y in ROWS]
MONITORS=SOURCES+[10*y+x for x in (3,5,7) for y in ROWS]


def pulse(t,start=90,amplitude=.1,plateau=20):
    z=t-start
    if z<=0 or z>=20+plateau:return 0.
    if z<10:return amplitude*.5*(1-np.cos(np.pi*z/10))
    if z<=10+plateau:return amplitude
    return amplitude*.5*(1+np.cos(np.pi*(z-10-plateau)/10))


def hydraulic_domain(params,nmesh=10,factor=.9,initial=None,start=0,
                     pulse_amplitude=0,pulse_start=90,pulse_center=25,plateau=20,switched=False):
    import anuga
    d=anuga.rectangular_cross_domain(nmesh,nmesh,len1=100.,len2=100.)
    d.set_store(False);d.set_flow_algorithm("DE0")
    d.set_quantity("elevation",lambda x,y:-params["slope"]*x+params["ridge"]*np.exp(-((y-50)/12)**2))
    d.set_quantity("friction",params.get("friction",.03))
    d.set_quantity("stage",expression="elevation + 0.15")
    if initial is not None:
        for k,v in initial.items():d.set_quantity(k,v,location="centroids")
    d.set_starttime(float(start))
    # Retag only the selected physical boundary edges, never modify interior fields.
    local=[]
    if pulse_amplitude:
        for edge,tag in list(d.boundary.items()):
            if tag=="left":
                _,y=d.get_edge_midpoint_coordinate(*edge)
                if pulse_center-10<=y<pulse_center+10:
                    d.boundary[edge]="pulse";local.append(edge)
        assert local,"No edges assigned to local pulse"
        # ANUGA caches tag->edge indices at construction. Retagging the dict
        # alone leaves vectorized boundary updates using the old left segment.
        tagged={tag:[] for tag in set(d.boundary.values())}
        for j,(cell,edge) in enumerate(zip(d.boundary_cells,d.boundary_edges)):
            tagged[d.boundary[int(cell),int(edge)]].append(j)
        d.tag_boundary_cells=tagged
        d.mesh.tag_boundary_cells=tagged
    def level(t):
        before=max(min(t,90)-params["onset"],0)
        after=max(t-max(90,params["onset"]),0)
        rise=factor*max(t-params["onset"],0) if not switched else factor*before+.65*after
        return .15+params["rate"]*rise
    wall=anuga.Reflective_boundary(d)
    boundaries=dict(left=anuga.Time_boundary(d,lambda t:[level(t),0,0]),right=wall,top=wall,bottom=wall)
    if pulse_amplitude:
        boundaries["pulse"]=anuga.Time_boundary(d,lambda t:[level(t)+pulse(t,pulse_start,pulse_amplitude,plateau),0,0])
    d.set_boundary(boundaries)
    return d


def available_history(records,region,now,k=6,max_age=60):
    sequence=records.get(region,[]) if isinstance(records,dict) else records
    candidates=[r for r in sequence if r["region_id"]==region and r["arrival_time_s"]<=now
        and 0<=now-r["valid_time_s"]<=max_age and r.get("residual_m") is not None]
    unique={r["valid_time_s"]:r for r in sorted(candidates,key=lambda r:r["arrival_time_s"],reverse=True)}
    return sorted(unique.values(),key=lambda r:r["valid_time_s"],reverse=True)[:k]


def future_label(event,region,now,tau):
    # Lock the version actually published at decision time, not a future issue.
    prediction=published(event,now,tau)[region]
    return float(np.interp(now+tau,event["time_s"],event["truth_depth_m"][:,region])-prediction)


def published(event,now,tau):
    cache=event.setdefault("_published_cache",{})
    if (now,tau) not in cache:cache[now,tau]=current_forecast(event,now,tau)
    return cache[now,tau]


def features(event,records,region,now,tau,graph,method):
    k=6;own=available_history(records,region,now,k)
    raw=np.zeros(4*k)
    for q,r in enumerate(own):raw[q*4:q*4+4]=[r["observed_depth_m"],r["prior_depth_m"],(now-r["valid_time_s"])/60,1]
    common=[]
    for i in MONITORS:
        h=available_history(records,i,now,k)
        if h:common.append(h[0]["residual_m"])
    level=float(np.mean(common)) if common and method!="I0_self" else 0.
    trend=(own[0]["residual_m"]-own[-1]["residual_m"])/(own[0]["valid_time_s"]-own[-1]["valid_time_s"] or 1) if len(own)>1 else 0.
    global_past=[]
    for i in MONITORS:
        h=available_history(records,i,now-10,k)
        if h:global_past.append(h[0]["residual_m"])
    global_trend=(level-np.mean(global_past))/10 if global_past and method!="I0_self" else 0.
    upstream=0.;support=0.;age=0.
    if method not in ("I0_self","I1_common"):
        for link in graph[str(region)]:
            source=link["source"]+7 if method=="I5_reverse_direction" else link["source"]
            h=available_history(records,source,now,k)
            desired=max(link["travel_s"]-tau,0)
            if method=="I4_wrong_lag":desired=(desired+20)%40
            for r in h:
                a=now-r["valid_time_s"]
                weight=link["weight"]*(1 if method=="I2_unaligned" else np.exp(-.5*((a-desired)/10)**2))/k
                upstream+=weight*r["residual_m"];support+=weight;age+=weight*a/60
    base=[published(event,now,0)[region],published(event,now,tau)[region],now/300,
        event["xy_m"][region,0]/100,event["xy_m"][region,1]/100,trend,level,global_trend]
    return np.r_[raw,base,upstream,support,age]


def sensor_records(event,noise,delay,moving):
    groups=[SOURCES]+[[10*y+x for y in ROWS] for x in (3,5,7)]
    family=event["metadata"]["family_index"]
    rng=np.random.default_rng(17000+family)
    records=[]
    for t in np.arange(20,301,10):
        ids=groups[(int(t)//10)%4] if moving else MONITORS
        for region in ids:
            value=np.interp(t,event["time_s"],event["truth_depth_m"][:,region])+rng.normal(0,noise)
            records.append(observation(event,region,t,t+delay,float(value),20))
    return {i:[r for r in records if r["region_id"]==i] for i in MONITORS}
