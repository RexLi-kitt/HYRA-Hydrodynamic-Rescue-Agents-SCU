import json
from time import perf_counter
import numpy as np
from .planner import *
from flood_dqn.env import START
def run(parent,layout,radius=60,refloat=True,arm='B0',predictor=None,alpha=0.,collect=False,intervene=None):
    birth,xy,status,sha,strong=scenario(parent,layout,refloat);f=fields(parent)
    m=Memory();p=START.copy();found=np.full(40,-1);saved=found.copy();began=found.copy();cnt=np.zeros(40,int);owner=np.full(40,-1)
    frames=[];labels=[];choices=[];conflicts=0;elapsed=0
    for t in range(0,T,5):
        alive=(birth<=t)&(t<birth+LIMIT)&(status[t]!=4)&(saved<0)
        visible=np.flatnonzero(alive&(np.linalg.norm(xy[t,:,None]-p,axis=2)<=radius).any(1))
        found[visible[found[visible]<0]]=t
        m.observe(t,p,radius,f,visible,xy[t,visible])
        eligible=alive&(found<0)
        zone=np.floor(xy[t,eligible]/100).astype(int).clip(0,9)
        y=np.bincount(zone[:,0]+10*zone[:,1],minlength=100)
        tm=perf_counter()
        age,wet=m.features(t,collect or arm in ('LG','TG'))
        dm=None
        if arm=='oracle':dm=y
        elif predictor is not None:dm=predictor(m.sequence())
        if collect:frames.append(m.history[-1]);labels.append(y)
        aa=alpha
        if intervene is not None and t!=intervene:aa=0
        assign,end,z=plan(m,p,t,age,wet,dm,aa);path,blocked=collide(p,end);conflicts+=blocked;elapsed+=perf_counter()-tm
        choices.append(z.tolist())
        for k,newp in enumerate(path,1):
            jsec=t+k;continuing=set()
            for i,j in enumerate(assign):
                if j<0 or saved[j]>=0 or jsec>birth[j]+LIMIT or status[jsec,j]==4:continue
                if max(np.linalg.norm(p[i]-xy[jsec-1,j]),np.linalg.norm(newp[i]-xy[jsec,j]))<=10+1e-8:
                    cnt[j]=(cnt[j] if owner[j]==i else 0)+1;owner[j]=i;continuing.add(j)
                    if began[j]<0:began[j]=jsec-1
                    if cnt[j]>=10:saved[j]=jsec;m.done.add(j)
            drop=[j for j in np.flatnonzero(cnt) if j not in continuing]
            if drop:cnt[drop]=0;owner[drop]=-1
            p=newp
    persons=[]
    for j,b in enumerate(birth):
        end=int(saved[j] if saved[j]>=0 else b+LIMIT);s=status[int(b):end+1,j]
        outcome='delivered' if saved[j]>=0 else ('left' if 4 in s else 'unseen_timeout' if found[j]<0 else 'seen_timeout')
        persons.append(dict(id=j,birth=int(b),found=int(found[j]),start=int(began[j]),saved=int(saved[j]),reason=outcome,stranded=int(((s[1:]==2)&(s[:-1]!=2)).sum()),snagged=int((s==3).any()),refloat=int(((s[1:]==1)&(s[:-1]==2)).sum())))
    count=int((saved>=0).sum())
    row=dict(parent=parent,layout=layout,radius=radius,arm=arm,saved=count,found=int((found>=0).sum()),strong=strong,all40=int(count==40),near=int(count>=38),conflict_stops=conflicts,ms=elapsed*1000/(T/5),scenario_hash=sha)
    return dict(row=row,persons=persons,choices=choices),frames,labels
