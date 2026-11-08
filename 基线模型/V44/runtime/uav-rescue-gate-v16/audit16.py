from pathlib import Path
import sys,os
OUT=Path(__file__).resolve().parent
V15=OUT.parent/"uav-fleet-paths-v15"
sys.path.insert(0,str(V15))
from tune15 import material,make_agent,read,save,sha,batch,runtime,np,s
def cases():
    return [(str(s.ROOT/"uav-priority-confirm-v10"),97151001+i,"ordinary",i) for i in range(12)]+[(str(s.ROOT/"uav-priority-confirm-v10"),97161001+i,f,i) for i in range(6) for f in s.FAMILIES[1:]]
def run(c):
    path=OUT/"search-audit"/f"{c[1]}-{c[2]}.json"
    if path.exists():return
    f,p=material(c,0);cls,_=runtime(1,"planning")
    cfg=read(V15/"tuning.json")["final"][read(V15/"development.json")["strongest"]];rows={}
    for mode in ("normal","search_only"):
        a=make_agent("",cfg);env=cls(f,p,(64,720,2400))
        if mode=="search_only":a.assign=lambda o:(np.full(4,-1,int),o.pos.copy())
        totals=np.zeros(6,int);first=np.full(64,-1)
        while env.t<2400:
            o=env.observation();first[(first<0)&o.visible]=o.t
            act,claims=a.act(o);targets=claims.copy()
            assigned=(a.track>=0) if mode=="normal" else np.zeros(4,bool)
            assigned |= claims>=0
            targets[(targets<0)&assigned]=a.track[(targets<0)&assigned]
            gap=np.full(4,np.inf)
            ii=np.flatnonzero(targets>=0)
            gap[ii]=np.linalg.norm(env.pos[ii]-p["points"][env.t,targets[ii]],axis=1)
            expired=np.zeros(4,bool);expired[ii]=(~p["available"][env.t,targets[ii]])|(env.t>=env.deadline[targets[ii]])
            hold=assigned&(act==8)&(claims>=0)
            totals+=[(~assigned).sum(),(assigned&~hold).sum(),hold.sum(),expired.sum(),(hold&(gap>1+1e-9)).sum(),(hold&(gap<=1+1e-9)).sum()]
            env.step(act,claims)
        first[(first<0)&env.visible]=env.t
        seen=first>=0;done=env.supplied>=0
        end=np.full(64,2400)
        for j in range(64):
            ex=np.flatnonzero(~p["available"][:,j]);end[j]=min(env.deadline[j],int(ex[0]) if len(ex) else 2400)
        rows[mode]=dict(count=64,found=int(seen.sum()),delivered=int(done.sum()),time=totals.tolist(),
            missed=int((~seen).sum()),unknown_window=int(((~seen)&(env.deadline<2400)&(end==env.deadline)).sum()),
            unknown_exit=int(((~seen)&(end<env.deadline)).sum()),unknown_mission=int(((~seen)&(end==2400)).sum()),
            available_end=end.tolist(),first_seen=first.tolist(),delivery_s=env.supplied.tolist())
        assert totals[:3].sum()==4*480
    save(path,dict(seed=c[1],family=c[2],results=rows))
def main():
    (OUT/"search-audit").mkdir(exist_ok=True);batch(run,cases(),6)
    rr=[read(OUT/"search-audit"/f"{c[1]}-{c[2]}.json") for c in cases()]
    info={}
    for f in s.FAMILIES:
        info[f]={}
        for m in ("normal","search_only"):
            rows=[r["results"][m] for r in rr if r["family"]==f]
            info[f][m]={k:sum(r[k] for r in rows) for k in ("count","found","delivered","missed","unknown_exit","unknown_window","unknown_mission")}
            info[f][m]["time"]=np.sum([r["time"] for r in rows],0).tolist()
    save(OUT/"search-audit.json",info);print("AUDIT",info)
if __name__=="__main__":main()
