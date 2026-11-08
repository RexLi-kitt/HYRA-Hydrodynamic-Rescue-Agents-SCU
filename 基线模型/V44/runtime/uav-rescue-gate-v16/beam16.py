from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit16 import *
from paths15 import FleetPaths,FIXED,DIRECTIONS,Cohorts,candidate_footprints,endpoints
from service14 import Switcher
from types import SimpleNamespace
from lattice import prepare_plan

def fit_lookup():
    centre=np.array([500.,500.]);table=np.empty((2,41,41),dtype=np.float32)
    for moving,v in enumerate((np.zeros(2),np.array([.4,0.]))):
        for iy in range(41):
            for ix in range(41):
                delta=np.array([ix-20.,iy-20.]);gap=np.linalg.norm(delta)
                if gap<1+1e-9:steps=0
                elif gap>20.8:steps=10
                else:
                    route=prepare_plan(centre,delta/gap,centre+delta,v,True,7)
                    steps=len(route) if route else 10
                table[moving,iy,ix]=steps
    np.save(OUT/"approach-steps.npy",table)
def expected_ends(p):
    # Full, normalized movement only. Boundary-truncated alternatives retained if all full moves excluded.
    return endpoints(p)
PROFILES={
    # lookahead, beam width, easy-service emphasis, approaching-deadline tilt, crossflow gain, commit
    "coverage":(9,18,0.,0.,0.,3),"service":(9,18,1.,.6,0.,3),
    "cross":(9,18,.7,1.2,.45,4),"downstream":(10,16,.7,2.,.15,4),
    "upstream":(10,16,.6,-1.3,.2,4),"broad":(14,22,.65,.9,.15,6)}
class NoveltyBeam(Switcher):
    def __init__(self,name):
        super().__init__("persist")
        self.fresh_search=True
        self.name=name
        self.horizon,self.width,self.easy,self.tilt,self.cross,self.commit=PROFILES[name]
        self.belief=Cohorts(12);self.belief.personal_window_seconds=720
        self.lookup_steps=np.load(OUT/"approach-steps.npy")
        self.plans_search={};self.beams=0;self.movement=0
    def act(self,o):
        self.belief.observe(o)
        return super().act(o)
    def fields(self,o):
        b=self.belief;stationary=np.zeros((100,100));mobile=stationary.copy()
        for j,fall in enumerate(b.fall):
            if fall+b.personal_window_seconds<o.t+10:continue
            if not np.isfinite(fall):
                stationary+=b.unseen[j];continue
            dx,dy=np.rint(b.shift[j]/10).astype(int)
            if abs(dx)>=100 or abs(dy)>=100:continue
            x0,x1=max(0,dx),min(100,100+dx);y0,y1=max(0,dy),min(100,100+dy)
            mobile[y0:y1,x0:x1]+=b.unseen[j,y0-dy:y1-dy,x0-dx:x1-dx]*(1+.35*np.exp(-(fall+b.personal_window_seconds-o.t)/180))
        axis=b.axis
        yy,xx=np.mgrid[:100,:100]
        along=((xx-49.5)*axis[0]+(yy-49.5)*axis[1])/50
        bias=np.exp(self.tilt*along*max(0,1-o.t/1450))
        # Trapping or prolonged stranding makes purely advected occupancy over-confident.
        residual=.035*np.minimum(1,np.maximum(0.,o.t-o.last)/400)
        return ((stationary/b.count+residual)*bias).ravel(),((mobile/b.count)*bias).ravel()
    def candidates(self,p,h,field,seen,moving,offset,axis):
        origins=p-offset
        ids,mask=candidate_footprints(origins,h)
        weight=field[ids]*mask*(~seen[np.arange(len(p))[:,None,None],ids])
        if self.easy:
            dx=10*(ids%100)+5-origins[...,0,None]
            dy=10*(ids//100)+5-origins[...,1,None]
            if moving:
                dx,dy=dx*axis[0]+dy*axis[1],-dx*axis[1]+dy*axis[0]
            ix=np.rint(dx+20).astype(int).clip(0,40);iy=np.rint(dy+20).astype(int).clip(0,40)
            factor=2.8/(2+self.lookup_steps[int(moving),iy,ix])
            weight*=1+self.easy*(factor-1)
        return weight.sum(-1),ids,mask
    def path(self,o,i,fields,reserved):
        horizon=self.horizon;axis=self.belief.axis;velocity=self.belief.field[1:]
        positions=o.pos[i,None].copy();values=np.zeros(1);tracks=np.empty((1,0),int)
        seen=[np.zeros((1,10000),bool),np.zeros((1,10000),bool)]
        for step in range(horizon):
            ends,length=expected_ends(positions)
            # Model the displacement of targets after the current observation.
            g0,id0,m0=self.candidates(ends,DIRECTIONS,fields[0],seen[0],False,np.zeros(2),axis)
            g1,id1,m1=self.candidates(ends,DIRECTIONS,fields[1],seen[1],True,(step+1)*5*velocity,axis)
            gain=g0+g1
            scale=max(.1,float(gain.max()))
            gain+=self.cross*scale*(1-np.abs(DIRECTIONS@axis))
            for reserved_path in reserved:
                # Reserve nearby search territory as well as the other aircraft's precise path.
                centre=reserved_path[min(step,len(reserved_path)-1)]
                dist=np.linalg.norm(ends-centre,axis=-1)
                gain-=.9*scale*np.maximum(0,1-dist/150)
            scores=values[:,None]+(.97**step)*gain
            scores[length<39.9]=-1.e10
            sorted_indices=np.argsort(-scores.ravel(),kind="stable")
            parent,actions=np.unravel_index(sorted_indices,scores.shape)
            good=scores[parent,actions]>-1.e9
            parent,actions=parent[good],actions[good]
            if not len(parent):break
            # Retain competing ways into the same area but avoid flooding the beam with loops.
            ends_sorted=ends[parent,actions]
            keys=np.c_[np.rint(ends_sorted/8).astype(int),actions]
            _,unique=np.unique(keys,axis=0,return_index=True)
            keep=np.sort(unique)[:self.width];parent=parent[keep];actions=actions[keep]
            for b,(ids,mask) in enumerate(((id0,m0),(id1,m1))):
                seen[b]=seen[b][parent].copy()
                jj,cc=np.nonzero(mask[parent,actions])
                seen[b][jj,ids[parent[jj],actions[jj],cc]]=True
            positions=ends[parent,actions];values=scores[parent,actions]
            tracks=np.column_stack((tracks[parent],actions))
        if not tracks.shape[1]:return [int(np.argmax(expected_ends(o.pos[i])[1]))]
        return tracks[np.argmax(values)].tolist()
    def pressure_search(self,o,ends,length,assigned,targets):
        active=np.flatnonzero(assigned>=0);free=np.flatnonzero(assigned<0)
        for i in active:self.plans_search.pop(i,None)
        actions=np.zeros(4,int);reserved=[[targets[i]] for i in active]
        missing=[]
        for i in np.roll(free,o.t//60%max(1,len(free))):
            entry=self.plans_search.get(i)
            if entry and entry[0]==o.t and np.linalg.norm(entry[1]-o.pos[i])<.02 and entry[2]:
                reserved.append(o.pos[i]+(DIRECTIONS[entry[2]]*50).cumsum(0))
            else:missing.append(i)
        if missing:
            fields=self.fields(o)
            for i in missing:
                seq=self.path(o,i,fields,reserved)
                reserved.append(o.pos[i]+(DIRECTIONS[seq]*50).cumsum(0))
                self.plans_search[i]=(o.t,o.pos[i].copy(),seq[:self.commit])
                self.beams+=1
        for i in free:
            seq=self.plans_search[i][2];actions[i]=seq[0]
            self.plans_search[i]=(o.t+5,ends[i,seq[0]],seq[1:])
            self.movement+=1
        return actions
