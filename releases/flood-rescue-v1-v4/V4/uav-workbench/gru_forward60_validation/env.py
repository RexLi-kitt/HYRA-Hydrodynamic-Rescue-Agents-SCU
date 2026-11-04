"""One-second public observation, common planner and delivery environment."""
from types import SimpleNamespace
import numpy as np
from dynamic_drop_rebuild.planner import CENTERS, GRID, ZONE, collide, plan
from flood_dqn.env import START

N = 1500
AX = np.arange(100)
OFF = np.arange(-7,8)

def zone_ids(pos):
    cells = np.floor(pos / 100).astype(int).clip(0,9)
    return cells[:,0]+10*cells[:,1]

def footprint(p):
    c=np.floor(p/10).astype(int)
    xs=(c[:,0,None,None]+OFF[None,None,:]).clip(0,99)
    zs=(c[:,1,None,None]+OFF[None,:,None]).clip(0,99)
    valid=(xs*10+5-p[:,0,None,None])**2+(zs*10+5-p[:,1,None,None])**2<=3600
    return np.unique((xs+100*zs)[valid])

def transport(mass,flow):
    """Frozen measured velocity; bilinear Eulerian forward splat with open edges."""
    if not flow.any():return mass.copy(),0.
    dest=np.stack((AX%10,AX//10),axis=1)+flow/100
    lo=np.floor(dest).astype(int);frac=dest-lo
    idx=[];weights=[]
    for a,b in ((0,0),(1,0),(0,1),(1,1)):
        x,z=lo[:,0]+a,lo[:,1]+b
        idx.append(np.where((x>=0)&(x<10)&(z>=0)&(z<10),x+10*z,100))
        weights.append((frac[:,0] if a else 1-frac[:,0])*(frac[:,1] if b else 1-frac[:,1]))
    idx=np.asarray(idx).ravel();weights=np.asarray(weights);result=mass
    for _ in range(60): result=np.bincount(idx,weights=(weights*result).ravel(),minlength=101)[:100]
    return result,float(mass.sum()-result.sum())

def movement(m,p,t,x,demand):
    assign,end,z=plan(m,p,t,x[:,3],x[:,6],demand,.5)
    d=end-p;end=p+d*np.minimum(1,10/np.maximum(np.linalg.norm(d,axis=1),1e-12))[:,None]
    path,n=collide(p,end)
    return assign,path[0],z,n

def run(data,arm,predict=None,collect=False):
    f,birth,xy,state,meta=data
    m=SimpleNamespace(last=np.full(40,-1),loc=np.zeros((40,2)),owner=np.full(4,-1),done=set())
    p=START.copy();found=np.full(40,-1);saved=found.copy();start=found.copy()
    owner=found.copy();count=np.zeros(40,int);foundzone=np.zeros(40,int);savedzone=np.zeros(40,int)
    stamp=np.full(10000,-10000);cells=np.zeros((3,10000));hist=np.zeros((91,100,14),'f')
    X=np.zeros((N,100,14),'f') if collect else None
    Y=np.zeros((N,100),'u1') if collect else None
    conflicts=0;lost=0.
    for t in range(N):
        alive=(birth<=t)&(t<birth+180)&(state[t]!=4)&(saved<0)
        visible=np.flatnonzero(alive&(((xy[t,:,None]-p)**2).sum(2)<=3600).any(1))
        new=visible[found[visible]<0];found[new]=t;foundzone[new]=zone_ids(xy[t,new])
        m.loc[visible]=xy[t,visible];m.last[visible]=t
        missing=(m.last>=0)&(m.last<t)&(((m.loc[:,None]-p)**2).sum(2)<=3600).any(1)
        m.last[missing]=-2
        ids=footprint(p);stamp[ids]=t
        k,rem=divmod(t,5)
        current=f[k].reshape(3,-1)[:,ids]
        if rem:current=current+(rem/5)*(f[k+1].reshape(3,-1)[:,ids]-current)
        cells[:,ids]=current
        ages=t-stamp
        def total(v):return np.bincount(ZONE,weights=v,minlength=100)
        ever=stamp>=0;recent=ages<=120;nr=total(recent);valid=nr>0
        means=[total(v*recent)/np.maximum(nr,1) for v in cells]
        wet=total((cells[0]>=.01)&recent)/np.maximum(nr,1)
        known=(m.last>=0)&(saved<0)
        x=np.column_stack((CENTERS/1000,np.full(100,t/N),total(np.minimum(ages/180,1))/100,
             total(ever)/100,total(ages<30)/100,wet,*means,valid,
             np.bincount(foundzone[(found>=0)&(t-found<30)],minlength=100),
             np.bincount(zone_ids(m.loc[known]),minlength=100),
             np.bincount(savedzone[saved>=0],minlength=100))).astype('f')
        hist[t%91]=x
        if collect:
            X[t]=x;Y[t]=np.bincount(zone_ids(xy[t,alive&(found<0)]),minlength=100)
        demand=None
        if arm=='P1':demand=(.12+x[:,3])*(1+3*wet)
        elif arm in ('M0','M1'):
            seq=hist[(t+np.array([-90,-60,-30,0]))%91].transpose(1,0,2).copy()
            seq[:,np.array([t<90,t<60,t<30,False])]=0
            demand=np.asarray(predict(seq),dtype=float).reshape(100)
        if arm in ('P1','M1'):
            ahead,leak=transport(demand,np.column_stack(means[1:]));lost+=leak
            demand=.5*(demand+ahead)
        assign,newp,z,n=movement(m,p,t,x,demand);conflicts+=n
        keep=np.zeros(40,bool)
        for i,j in enumerate(assign):
            if j<0 or saved[j]>=0 or t+1>birth[j]+180 or state[t+1,j]==4:continue
            if np.linalg.norm(p[i]-xy[t,j])<=10+1e-8 and np.linalg.norm(newp[i]-xy[t+1,j])<=10+1e-8:
                keep[j]=True;count[j]=(count[j] if owner[j]==i else 0)+1;owner[j]=i
                if start[j]<0:start[j]=t
                if count[j]==10:
                    saved[j]=t+1;savedzone[j]=zone_ids(xy[t+1,j:j+1])[0];m.done.add(j)
        count[~keep]=0;owner[~keep]=-1;p=newp
    persons=[]
    for i in range(40):
        why='delivered' if saved[i]>=0 else 'left' if (state[:birth[i]+181,i]==4).any() else 'unseen_timeout' if found[i]<0 else 'seen_timeout'
        persons.append(dict(id=i,birth=int(birth[i]),found=int(found[i]),start=int(start[i]),saved=int(saved[i]),reason=why))
    n=int((saved>=0).sum())
    return dict(saved=n,found=int((found>=0).sum()),near=int(n>=38),all40=int(n==40),conflict_stops=conflicts,transport_out=lost),persons,X,Y

def checks():
    from unittest.mock import patch
    import sys
    module=sys.modules[__name__]
    q=np.array([[65.,5.]]*4)
    assert 0 in footprint(q) and 1 in footprint(q) and 13 not in footprint(q)
    mass=np.zeros(100);mass[44]=1;speed=np.zeros((100,2))
    still,leak=transport(mass,speed);assert np.array_equal(still,mass) and leak==0
    speed[:]=[1,0];moved,leak=transport(mass,speed)
    assert moved[45]>0 and np.isclose(moved.sum()+leak,1) and leak>=-1e-12
    p=np.array([[100.,100.],[106.,94.],[900.,5.],[995.,995.]])
    path,blocked=collide(p,p+[[10,0],[0,10],[0,0],[0,0]])
    assert blocked==2 and np.array_equal(path[0],p)
    f=np.broadcast_to(np.zeros((1,3,1,1),'f'),(301,3,100,100))
    b=np.full(40,1490);b[0]=0
    xy=np.broadcast_to([500.,500.],(1501,40,2)).copy();xy[:,0]=START[0]
    status=np.ones((1501,40),'u1')
    def trial(hidden,late):
        frames=[];intents=[];dm=[]
        xy[:,1]=hidden
        def dispatch(mem,pos,t,features,demand):
            a=np.full(4,-1)
            if t>=170+late:a[0]=0
            if t==0:frames.append(features.copy());intents.append(plan(mem,pos,t,features[:,3],features[:,6],demand,.5)[1]);dm.append(demand.copy())
            return a,pos,np.full(4,-1),0
        with patch.object(module,'movement',dispatch),patch.object(module,'N',182):
            _,people,_,_=run((f,b,xy,status,{}),'M0',lambda seq:seq[:,-1,3])
        return people[0]['saved'],(frames[0],intents[0],dm[0])
    saved,obs=trial([300,300],0);assert saved==180
    saved2,obs2=trial([700,700],1);assert saved2==-1 and all(np.array_equal(a,b) for a,b in zip(obs,obs2))
    return dict(sensor_boundary=True,collision=True,mass_balance=True,deadline_equality=True,hidden_isolation=True)
