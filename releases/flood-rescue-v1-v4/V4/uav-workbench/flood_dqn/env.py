"""Joint clock in seconds, (x,z) in metres, sensor memory shared before decisions."""
import math
import numpy as np

START = np.array([[5,5],[995,5],[5,995],[995,995]], dtype=float)
DIRECTIONS = np.array([[0,-1],[0,1],[-1,0],[1,0],[-1,-1],[-1,1],[1,-1],[1,1],[0,0]], float)
DIRECTIONS[4:8] /= math.sqrt(2)
DZ, DX = np.indices((7,7)).reshape(2,-1) - 3
EPS = 1e-8
LOCAL = 5*49
GLOBAL = 25
VECTOR = 2+2+8+2+1+1+1+1+1+1
DIM = LOCAL+GLOBAL+VECTOR


def layout(seed):
    pool = np.delete(np.arange(10000), [0,99,9900,9999])
    return np.random.default_rng(seed).choice(pool, 20, replace=False)


def stop_distance(p, d, targets):
    limits = [(1000-p[k])/v if v > 0 else -p[k]/v for k,v in enumerate(d) if v]
    edge = min(limits, default=50.)
    length = min(50., edge)
    if len(targets) and d.any():
        delta = targets-p
        ahead = delta@d
        squared = (delta*delta).sum(1)
        disc = 100 - squared + ahead*ahead
        eligible = (squared>100+EPS) & (ahead>0) & (disc>=-EPS)
        if eligible.any():
            entry = ahead[eligible]-np.sqrt(np.maximum(0,disc[eligible]))
            length = min(length, float(entry.min()))
    return length if d.any() else 0., edge < 50-EPS and length >= edge-EPS


def close(p, v, a, q, w, b):
    # Piecewise constant 10 m/s, then stationary for the rest of the step.
    if np.max(np.abs(p-q)) > 105: return False
    cuts = sorted({0., a, b, 50.})
    for lo, hi in zip(cuts, cuts[1:]):
        r = p-q + v*min(lo,a) - w*min(lo,b)
        u = (v if lo<a else 0.) - (w if lo<b else 0.)
        norm = np.sum(u*u)
        t = min(hi-lo,max(0.,-np.sum(r*u)/norm)) if norm else 0.
        if np.sum((r+t*u)**2) < 25-EPS: return True
    return False


class Environment:
    def __init__(self, fields, safe, cells, log=False):
        self.fields, self.safe, self.cells = fields, safe, np.asarray(cells)
        self.targets = np.column_stack((self.cells%100,self.cells//100))*10+5
        self.p = START.copy()
        self.heading = (500-self.p)/(495*math.sqrt(2))
        self.step_no = 0
        self.stamps = np.full(10000,-1,dtype=np.int16)
        self.water = np.zeros((3,10000),dtype=np.float32)
        self.pool = np.zeros(25, dtype=np.float32)
        self.found = np.zeros(20,bool)
        self.rescued = np.zeros(20,bool)
        self.owner = np.full(20,-1)
        self.first = self.last = None
        self.pairs = self.covered = 0
        self.log = [] if log else None
        self.observe()
        if log: self.log.append(self.event([], [], [], [], []))

    def observe(self):
        ij = np.floor(self.p/10).astype(int)
        x, z = ij[:,0,None]+DX, ij[:,1,None]+DZ
        self.inside = (x>=0)&(x<100)&(z>=0)&(z<100)
        self.ids = np.clip(z*100+x,0,9999)
        dx, dz = x*10+5-self.p[:,0,None], z*10+5-self.p[:,1,None]
        hit = self.inside & (dx*dx+dz*dz<=625+EPS) & (dx*self.heading[:,0,None]+dz*self.heading[:,1,None]>=-EPS)
        ids = np.unique(self.ids[hit])
        new = ids[self.stamps[ids]<0]
        self.covered += len(new)
        np.add.at(self.pool, (new//100//20)*5+(new%100//20), 1.)
        self.stamps[ids] = self.step_no
        self.water[:,ids] = self.fields[self.step_no].reshape(3,-1)[:,ids]
        found = (self.stamps[self.cells]>=0) & ~self.found
        self.found |= found
        if self.log is not None:
            self.sample = [self.ids[i,hit[i]].tolist() for i in range(4)]
        return len(new), np.flatnonzero(found)

    def observation(self):
        stamps = self.stamps[self.ids]
        known = self.inside & (stamps>=0)
        local = np.empty((4,5,49),np.float32)
        local[:,:3] = self.water[:,self.ids].transpose(1,0,2)*known[:,None,:]
        local[:,3] = known
        local[:,4] = np.where(known,(self.step_no-stamps)/300,0.)
        delta = self.targets[None,:,:]-self.p[:,None,:]
        dist = (delta**2).sum(2)
        pending = self.found & ~self.rescued
        dist[:,~pending] = np.inf
        nearest = dist.argmin(1)
        relative = delta[np.arange(4),nearest] if pending.any() else np.zeros((4,2))
        own = self.owner[nearest] == np.arange(4)
        vector = np.column_stack((self.p/1000, self.heading,
            ((self.p[None,:,:]-self.p[:,None,:])/1000).reshape(4,8),
            relative/1000, np.minimum(1,np.linalg.norm(relative,axis=1)/50),
            np.full(4,pending.any()), own, np.full(4,self.found.sum()/20),
            np.full(4,self.rescued.sum()/20), np.full(4,1-self.step_no/300)))
        return np.concatenate((local.reshape(4,-1),np.broadcast_to(self.pool/400,(4,25)),vector),axis=1).astype(np.float32)

    def event(self, actions, pairs, edges, found, saved):
        return dict(t=self.step_no, actions=list(map(int,actions)), xy=self.p.tolist(),
                    observed=self.sample, discovered=list(map(int,found)), rescued=list(map(int,saved)),
                    conflicts=pairs, boundary=edges, serving=self.owner.tolist(), covered=self.covered)

    def step(self, actions):
        v = DIRECTIONS[actions]
        pending = self.found & ~self.rescued
        length, boundary = zip(*(stop_distance(p,d,self.targets[pending]) for p,d in zip(self.p,v)))
        length = list(length)
        def conflicts():
            return [(i,j) for i in range(4) for j in range(i+1,4)
                    if close(self.p[i],v[i],length[i],self.p[j],v[j],length[j])]
        original = conflicts()
        problems = original
        while problems:
            for i,j in problems: length[i] = length[j] = 0.
            problems = conflicts()
        moved = np.asarray(length)>EPS
        self.p += v*np.asarray(length)[:,None]
        self.heading[moved] = v[moved]
        safe = self.safe[self.step_no].reshape(-1)[self.cells]
        distances = ((self.p[:,None,:]-self.targets)**2).sum(2)
        candidates = np.argwhere((distances<=100+EPS)&(np.asarray(actions)[:,None]==8)&pending&safe)
        next_owner = np.full(20,-1)
        occupied = set()
        saved = []
        for i,j in sorted(candidates,key=lambda ij:(self.owner[ij[1]]!=ij[0],distances[tuple(ij)],*ij)):
            if i in occupied or next_owner[j]!=-1: continue
            occupied.add(i)
            next_owner[j]=i
            if self.owner[j]==i:
                self.rescued[j]=True
                saved.append(j)
        next_owner[self.rescued]=-1
        self.owner=next_owner
        self.step_no+=1
        new,found=self.observe()
        if saved and self.first is None: self.first=self.step_no
        if saved and self.rescued.all(): self.last=self.step_no
        self.pairs+=len(original)
        if self.log is not None: self.log.append(self.event(actions,original,np.flatnonzero(boundary).tolist(),found,saved))
        done = self.covered==10000 and self.rescued.all()
        reward = .02*new+len(found)+10*len(saved)-.5*len(original)-.1*sum(boundary)
        return reward, bool(done or self.step_no==300)

    def metrics(self):
        return dict(ST=self.step_no, ACC=self.pairs/self.step_no,RCR=float(self.rescued.mean()),
                    FRT=self.first, CRT=self.last, coverage=self.covered/10000,
                    timeout=not(self.covered==10000 and self.rescued.all()))
