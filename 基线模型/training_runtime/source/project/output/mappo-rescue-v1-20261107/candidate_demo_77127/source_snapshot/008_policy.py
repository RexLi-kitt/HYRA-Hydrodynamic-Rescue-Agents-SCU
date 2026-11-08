"""Fixed rule baseline. Decisions use only the public actor observation and mask."""
import math
from collections import deque
import numpy as np

DIRECTIONS=np.array([[0,-1],[0,1],[-1,0],[1,0],[-1,-1],[-1,1],[1,-1],[1,1]],float)
DIRECTIONS /= np.linalg.norm(DIRECTIONS,axis=1)[:,None]

def short_end(pos, direction, people):
    limit=50.
    for k in range(2):
        if abs(direction[k])>1e-12:
            limit=min(limit,((1000. if direction[k]>0 else 0.)-pos[k])/direction[k])
    for point in people:
        delta=pos-point
        if delta@delta <= 100.+1e-8: continue
        b=delta@direction; disc=b*b-delta@delta+100.
        if disc>=0:
            hit=-b-math.sqrt(max(0.,disc))
            if -1e-8<=hit<=limit: limit=max(0.,hit)
    return np.clip(pos+direction*max(0.,limit),0.,1000.)

def move_towards(pos,goal,available,people):
    # Two-step lookahead handles the eight fixed headings and prevents a single-step
    # local minimum. Same known-person short-approach geometry as the environment.
    ranked=[]
    for a in range(8):
        if not available[a]:continue
        end=short_end(pos,DIRECTIONS[a],people)
        d1=float(np.linalg.norm(end-goal))
        if np.linalg.norm(end-pos)<1e-5:continue
        d2=d1
        for direction in DIRECTIONS:
            e2=short_end(end,direction,people)
            d2=min(d2,float(np.linalg.norm(e2-goal)))
        ranked.append((min(d1,d2),d1,a))
    return min(ranked)[2] if ranked else 8

class Policy:
    def __init__(self):
        self.routes=[]
        for i in range(4):
            left=0 if i%2==0 else 500; bottom=0 if i<2 else 500
            # Two interleaved passes shift the sampling tracks by 25 m. The
            # physical sensor is directional and sampled, so this is not a
            # promise that all cells will be seen.
            first=[bottom+25+50*j for j in range(10)]
            second=[bottom+50*j for j in range(11)]
            if i>=2:first.reverse();second.reverse()
            rows=first+second
            near=left if i%2==0 else left+500; far=left+500 if i%2==0 else left
            points=[]
            for n,z in enumerate(rows):
                points.append(np.array([near if n%2==0 else far,z],float))
                points.append(np.array([far if n%2==0 else near,z],float))
            self.routes.append(points)
        self.route_index=[0]*4;self.claims=[None]*4;self.cooldown={};self.previous_hover=[None]*4
        self.scan_complete=[False]*4;self.seen=np.zeros((100,100),bool)
        self.recent=[[] for _ in range(4)]
        self.pending=[[] for _ in range(4)]
    def ingest_public_local(self,obs,positions):
        # Each actor's local 11x11 explored bitmap is part of its 740-D input.
        # A public 10x10 aggregate equal to one proves every cell in that
        # block is explored, even when decoded position straddles a grid line.
        for br,bc in np.argwhere(obs[0,:100].reshape(10,10)>=1.):
            self.seen[br*10:br*10+10,bc*10:bc*10+10]=True
        for i in range(4):
            local=obs[i,601:722].reshape(11,11)
            def possible(v):
                n=round(v/10)
                if abs(v-n*10)<=.0002:return sorted(set(np.clip([n-1,n],0,99)))
                return [int(np.clip(math.floor(v/10),0,99))]
            projections=[]
            for col in possible(positions[i,0]):
                for row in possible(positions[i,1]):
                    known=np.zeros((100,100),bool)
                    for dz in range(-5,6):
                        r=row+dz
                        if not 0<=r<100:continue
                        l=max(0,col-5);h=min(100,col+6)
                        known[r,l:h]=local[dz+5,l-col+5:h-col+5]>.5
                    projections.append(known)
            self.seen |= np.logical_and.reduce(projections)
    def cleanup(self,i,pos,available,people):
        # Choose among legal movement samples using the same 25m forward
        # half-disc; no map/target data except the arrived coverage bitmap.
        lo_x=0 if i%2==0 else 50;lo_z=0 if i<2 else 50
        uncovered=np.argwhere(~self.seen[lo_z:lo_z+50,lo_x:lo_x+50])
        if not len(uncovered):
            uncovered=np.argwhere(~self.seen);lo_x=lo_z=0
        if not len(uncovered):return 8
        grid=uncovered+np.array([lo_z,lo_x]);xy=grid[:,::-1]*10+5.
        distances=np.linalg.norm(xy-pos,axis=1);goal=xy[np.argmin(distances)]
        if self.pending[i]:
            expected,a=self.pending[i].pop(0)
            if np.linalg.norm(pos-expected)<.01 and available[a]:return a
            self.pending[i]=[]
        ranked=[];recent=self.recent[i]
        def gain_at(end,a):
            x,z=end;c0=max(0,math.ceil((x-25)/10-.5));c1=min(100,math.floor((x+25)/10-.5)+1)
            r0=max(0,math.ceil((z-25)/10-.5));r1=min(100,math.floor((z+25)/10-.5)+1)
            cc=np.arange(c0,c1)*10+5-x;rr=np.arange(r0,r1)*10+5-z
            visible=(rr[:,None]**2+cc[None,:]**2<=625+1e-8)&(cc[None,:]*DIRECTIONS[a,0]+rr[:,None]*DIRECTIONS[a,1]>=-1e-8)
            return int((visible&~self.seen[r0:r1,c0:c1]).sum())
        for a in range(8):
            if not available[a]:continue
            end=short_end(pos,DIRECTIONS[a],people)
            if np.linalg.norm(end-pos)<1e-5:continue
            x,z=end;gain=gain_at(end,a)
            # Avoid wasting travel far outside the assigned quadrant when
            # a legal action stays within it; other quadrants remain observable.
            outside=max(0.,lo_x*10-x,x-(lo_x+50)*10)+max(0.,lo_z*10-z,z-(lo_z+50)*10)
            repeated=any(np.linalg.norm(end-v)<1e-3 for v in recent[-8:])
            ranked.append((-gain,round(outside,4),repeated,float(np.linalg.norm(end-goal)),a))
        if not ranked:return 8
        if -min(ranked)[0]==0:
            # A nearest cell can be behind the sensor: plan several discrete
            # movements to a future forward-facing sample, not just closer.
            queue=deque([(pos.copy(),[])]);visited={tuple(np.round(pos,3))};nodes=0
            while queue and nodes<5000:
                here,path=queue.popleft();nodes+=1
                order=sorted(range(8),key=lambda a:(float(np.linalg.norm(short_end(here,DIRECTIONS[a],people)-goal)),a))
                for a in order:
                    if not path and not available[a]:continue
                    end=short_end(here,DIRECTIONS[a],people)
                    if np.linalg.norm(end-here)<1e-5:continue
                    key=tuple(np.round(end,3))
                    if key in visited:continue
                    visited.add(key);newpath=path+[(here.copy(),a)]
                    if gain_at(end,a)>0:
                        self.pending[i]=newpath[1:];return newpath[0][1]
                    if len(newpath)<5:queue.append((end,newpath))
        a=min(ranked)[-1];recent.append(pos.copy());del recent[:-8]
        return a
    def act(self,obs,mask):
        obs=np.asarray(obs);mask=np.asarray(mask);assert obs.shape==(4,740) and mask.shape==(4,9)
        positions=obs[0,400:416].reshape(4,4)[:,:2].astype(float)*1000
        self.ingest_public_local(obs,positions)
        rows=obs[0,417:597].reshape(20,9);people={}
        for row in rows:
            if row[5]>.5:
                key=tuple(float(v) for v in row[:2]);people[key]=row.copy()
        active={k:r for k,r in people.items() if r[2]<.75}
        tick=int(round(float(obs[0,416])*300))
        # A failed hover is a public observation: no own service progress next tick.
        for i,key in enumerate(self.previous_hover):
            if key in active and not (active[key][2]>.25 and round(float(active[key][4])*4)==i+1):
                self.cooldown[key]=tick+15;self.claims[i]=None
        self.previous_hover=[None]*4
        # Preserve claims to prevent oscillation. Unsafe most-recent readings cause
        # release; retry after 15 steps, with another delay on each fresh unsafe reading.
        for i,key in enumerate(self.claims):
            if key not in active:self.claims[i]=None;continue
            r=active[key];owner=round(float(r[4])*4)-1
            if r[2]>.25 and owner!=i:self.claims[i]=None;continue
            if r[8]==0 and (r[6]*.5>.0800001 or r[7]*.3>.3000001):
                self.cooldown[key]=tick+15;self.claims[i]=None
        assigned={k for k in self.claims if k is not None}
        pairs=[]
        for key,r in active.items():
            if key in assigned:continue
            if r[8]==0 and (r[6]*.5>.0800001 or r[7]*.3>.3000001):
                self.cooldown[key]=tick+15;continue
            if self.cooldown.get(key,0)>tick:continue
            owner=round(float(r[4])*4)-1 if r[2]>.25 else -1
            for i in range(4):
                if self.claims[i] is None and (owner<0 or owner==i):
                    pairs.append((float(np.linalg.norm(positions[i]-np.array(key)*1000)),i,key))
        for _,i,key in sorted(pairs):
            if self.claims[i] is None and key not in assigned:self.claims[i]=key;assigned.add(key)
        obstacles=[np.array(k)*1000 for k in active]
        actions=[]
        for i,pos in enumerate(positions):
            key=self.claims[i]
            if key is not None:
                self.pending[i]=[]
                goal=np.array(key)*1000
                if np.linalg.norm(pos-goal)<=10.0002:
                    actions.append(8);self.previous_hover[i]=key;continue
            else:
                route=self.routes[i];idx=self.route_index[i]
                if not self.scan_complete[i]:
                    while np.linalg.norm(pos-route[idx])<=26.:
                        if idx==len(route)-1:
                            self.scan_complete[i]=True;break
                        idx+=1
                    self.route_index[i]=idx
                if self.scan_complete[i]:
                    actions.append(self.cleanup(i,pos,mask[i],obstacles));continue
                goal=route[idx]
            actions.append(move_towards(pos,goal,mask[i],obstacles))
        return np.array(actions,dtype=np.int64)
