"""Shared search-rescue environment; only complete feasible service is rewarded."""
import numpy as np
from scipy.linalg import solve
from .data import observation,current_forecast
from .representation import build_representation

MOVES = np.array([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]])


def sar_matrix(n, blocked=None, sources=None):
    xy = np.array([(x,y) for y in range(n) for x in range(n)])
    d = np.max(np.abs(xy[:,None]-xy[None,:]),axis=-1)
    c = ((d==1)).astype(float)
    if blocked is not None:
        c[blocked,:],c[:,blocked] = 0,0
    if sources is not None and len(sources):
        proximity = np.min(np.linalg.norm(xy[:,None]-xy[np.asarray(sources)][None,:],axis=-1),axis=1)
        c *= (1+.5*np.exp(-proximity))[None,:]
    c *= .35/np.maximum(c.sum(axis=1,keepdims=True),1)
    return c


def interval_valid(event, region, begin, end, limit):
    times = event["time_s"]
    if begin < times[0] or end > times[-1] or end < begin:
        return False
    inside = (times>=begin)&(times<=end)
    values = event["truth_depth_m"][:,region]
    depths = np.r_[np.interp(begin,times,values),values[inside],np.interp(end,times,values)]
    return bool(np.max(depths)<=limit)


def pool(image, width=5):
    # n=10, p=5 exact nonoverlapping average pooling.
    c,n,_ = image.shape
    if n%width:
        raise ValueError("Grid must divide pool size")
    factor = n//width
    return image.reshape(c,width,factor,width,factor).mean(axis=(2,4))


class RescueEnv:
    def __init__(self, event, config, structure, method, seed, static=False):
        self.event,self.config,self.structure,self.method,self.static = event,config,structure,method,static
        self.ec = config["environment"]
        self.n,self.n_agents = self.ec["grid_size"],self.ec["n_agents"]
        self.rng = np.random.default_rng(seed)
        size = self.n**2
        self.blocked = np.zeros(size,dtype=bool)
        start = np.array([0,1,self.n,self.n+1])[:self.n_agents]
        available = np.setdiff1d(np.arange(size),start)
        self.blocked[self.rng.choice(available,int(size*self.ec["obstacle_fraction"]),replace=False)] = True
        self.targets = self.rng.choice(np.flatnonzero(~self.blocked),self.ec["n_targets"],replace=False)
        self.status = {int(r):"undiscovered" for r in self.targets}
        self.positions = np.c_[start%self.n,start//self.n]
        self.resources = np.full(self.n_agents,self.ec["resources_per_agent"])
        self.busy = [None]*self.n_agents
        self.explored = np.zeros(size,dtype=bool)
        self.sampled_signal = np.zeros(size)
        self.records,self.log = [],[]
        self.last_sample = -1
        self.step_count,self.now = 0,0.
        self.initial_noise = self.rng.normal(0,.03,size)
        self.terrain_c = sar_matrix(self.n)
        signal_c = sar_matrix(self.n,self.blocked,self.targets)
        raw_signal = solve(np.eye(size)-signal_c,np.ones(size)+self.initial_noise)
        xy = np.c_[np.arange(size)%self.n,np.arange(size)//self.n]
        distance = np.linalg.norm(xy[:,None]-xy[self.targets][None,:],axis=-1).min(axis=1)
        self.true_signal = raw_signal*np.exp(-distance**2/4)
        self.true_signal[self.blocked] = 0
        self.first_service = None
        self.failed_attempts,self.collisions = 0,0
        self._sense()

    def _visible(self, positions):
        local = set()
        half = self.ec["sense_width"]//2
        for x,y in positions:
            for yy in range(max(0,y-half),min(self.n,y+half+1)):
                for xx in range(max(0,x-half),min(self.n,x+half+1)):
                    local.add(yy*self.n+xx)
        return np.array(sorted(local),dtype=int)

    def _sense(self, previous_positions=None):
        ids = self._visible(self.positions)
        self.explored[ids] = True
        self.sampled_signal[ids] = self.true_signal[ids]
        for r in ids:
            if self.status.get(int(r)) == "undiscovered":
                self.status[int(r)] = "discovered"
        # Until movement completes, sample at preceding decision position.
        interval = self.config["observations"]["sample_interval_s"]
        ticks = np.arange((self.last_sample//interval+1)*interval,self.now+1e-6,interval)
        for t in ticks:
            self.last_sample = float(t)
            if self.static or t < self.config["representation"]["lead_time_s"]:
                continue
            sample_ids = self._visible(previous_positions) if previous_positions is not None and t<self.now else ids
            for r in sample_ids:
                value = np.interp(t,self.event["time_s"],self.event["truth_depth_m"][:,r])
                noise = self.rng.normal(0,self.config["observations"]["noise_std_m"])
                self.records.append(observation(self.event,r,t,t+self.config["observations"]["delay_s"],
                    float(value+noise),self.config["representation"]["lead_time_s"]))

    def _map(self):
        size = self.n**2
        # Posterior SAR update uses observed obstacles only, no hidden map.
        c = self.terrain_c.copy()
        c[self.explored,:],c[:,self.explored] = 0,0
        mu = np.zeros(size)
        mu[self.explored] = np.where(self.blocked[self.explored],-1,1)
        terrain = mu+self.initial_noise
        for _ in range(12):
            terrain = mu+c@(terrain-mu)+self.initial_noise
        pos = np.zeros(size)
        for r,s in self.status.items():
            if s != "undiscovered":
                pos[r] = 3 if s=="completed" else 2
        for x,y in self.positions:
            pos[y*self.n+x] = 1
        base = np.stack([self.explored&~self.blocked,self.explored&self.blocked,terrain,self.sampled_signal]).astype(np.float32)
        rep_config = dict(self.config["representation"],method=self.method)
        rep = build_representation(self.records,None,self.now,self.structure,rep_config)
        future = np.zeros(size) if self.static else current_forecast(self.event,self.now,20)
        present = np.zeros(size) if self.static else current_forecast(self.event,self.now,0)
        # Task/service status channels contain only observed tasks.
        service = np.zeros(size)
        for r,s in self.status.items():
            if s != "undiscovered":
                service[r] = {"discovered":.25,"servicing":.5,"completed":1,"expired":-.5}[s]
        extras = np.vstack([rep["features"].T,present,future,service])
        return base.reshape(4,self.n,self.n),extras.reshape(-1,self.n,self.n),pos.reshape(1,self.n,self.n)

    def get_obs(self):
        base,extras,pos = self._map()
        image = np.concatenate([base,extras])
        pooled = pool(image)
        pp = pool(pos)
        half = 2
        pad = np.pad(image,((0,0),(half,half),(half,half)))
        observations = []
        for i,(x,y) in enumerate(self.positions):
            local = pad[:,y:y+5,x:x+5]
            # Remaining supplies and own service time are locally known.
            supplies = np.full((1,5,5),self.resources[i]/self.ec["resources_per_agent"])
            own = np.full((1,5,5),0 if self.busy[i] is None else (self.busy[i]["end"]-self.now)/self.ec["service_s"])
            observations.append(np.concatenate([local,pooled,pp,supplies,own]).astype(np.float32))
        return np.asarray(observations)

    def get_state(self):
        # Training-only state. Never passed to the actor.
        ref = np.where(self.blocked,-1,1).astype(float)
        for r in self.targets:
            ref[r] = 3 if self.status[int(r)]=="completed" else 2
        depth = np.zeros(self.n**2) if self.static else np.array([np.interp(self.now,self.event["time_s"],
            self.event["truth_depth_m"][:,r]) for r in range(self.n**2)])
        return np.r_[ref,depth,self.explored,self.positions.ravel()/self.n,self.resources,self.now/300].astype(np.float32)

    def step(self,actions):
        previous = self.positions.copy()
        proposed = previous.copy()
        invalid = np.zeros(self.n_agents,dtype=bool)
        moving = np.array([b is None for b in self.busy])
        for i,a in enumerate(actions):
            if not moving[i]:
                continue
            dx,dy = MOVES[int(a)]
            x,y = previous[i]+[dx,dy]
            if not (0<=x<self.n and 0<=y<self.n) or self.blocked[y*self.n+x]:
                invalid[i] = True
            elif dx and dy and (self.blocked[previous[i,1]*self.n+x] or self.blocked[y*self.n+previous[i,0]]):
                invalid[i] = True
            else:
                proposed[i] = [x,y]
        # Reject conflicts to a fixed point, including moves into an agent stopped by another collision.
        while True:
            conflict = invalid.copy()
            for i in range(self.n_agents):
                for j in range(i):
                    if np.array_equal(proposed[i],proposed[j]) or (np.array_equal(proposed[i],previous[j]) and np.array_equal(proposed[j],previous[i])):
                        conflict[i] |= moving[i]
                        conflict[j] |= moving[j]
            if np.array_equal(conflict,invalid):
                break
            invalid = conflict
            proposed[invalid] = previous[invalid]
        proposed[invalid] = previous[invalid]
        explored_before = self.explored.sum()
        self.positions = proposed
        reward = -float(invalid.sum())
        self.collisions += int(invalid.sum())
        self.step_count += 1
        self.now = self.step_count*self.ec["step_s"]
        self._sense(previous)
        reward += .1*(1+explored_before/self.n**2)*(self.explored.sum()-explored_before)
        failed_this_step=set()
        for i,b in enumerate(self.busy):
            if b is None:
                continue
            valid = self.static or interval_valid(self.event,b["region"],b["start"],min(self.now,b["end"]),self.ec["depth_limit_m"])
            if not valid:
                self.status[b["region"]] = "discovered"
                self.busy[i] = None
                failed_this_step.add(i)
                self.failed_attempts += 1
                self.log.append(dict(type="failed",time_s=self.now,agent=i,region=b["region"],elapsed_s=self.now-b["start"]))
            elif self.now >= b["end"]:
                self.status[b["region"]] = "completed"
                self.resources[i] -= 1
                self.busy[i] = None
                reward += 5/max(b["entry_distance"],1.)
                if self.first_service is None:
                    self.first_service = self.now
                self.log.append(dict(type="completed",time_s=self.now,agent=i,region=b["region"],start_s=b["start"]))
        for i,(x,y) in enumerate(self.positions):
            r = int(y*self.n+x)
            if i not in failed_this_step and self.busy[i] is None and self.resources[i]>0 and self.status.get(r)=="discovered":
                self.busy[i] = dict(region=r,start=self.now,end=self.now+self.ec["service_s"],
                                    entry_distance=float(np.linalg.norm(previous[i]-self.positions[i])))
                self.status[r] = "servicing"
                self.log.append(dict(type="started",time_s=self.now,agent=i,region=r))
        reward -= min(5.,.1*np.log(max(self.step_count,1))*np.log(self.n**2-self.explored.sum()+1))
        done = self.step_count>=self.ec["episode_limit"] or (self.explored.all() and all(s=="completed" for s in self.status.values()))
        if done and self.step_count>=self.ec["episode_limit"]:
            reward -= 10*(1-sum(s=="completed" for s in self.status.values())/len(self.targets))
        self.log.append(dict(type="decision",time_s=self.now,actions=np.asarray(actions).tolist(),positions=self.positions.tolist(),reward=reward))
        return float(reward),bool(done),{}

    def metrics(self):
        completed = sum(s=="completed" for s in self.status.values())
        missed = 0
        for r,s in self.status.items():
            if s != "completed" and not self.static:
                feasible = any(interval_valid(self.event,r,t,t+self.ec["service_s"],self.ec["depth_limit_m"])
                    for t in np.arange(self.now,301,self.ec["step_s"]))
                missed += not feasible
        return dict(completion_rate=completed/len(self.targets),weighted_service_yield=float(completed),
                    target_denominator=len(self.targets),completed=completed,missed_opportunities=int(missed),
                    first_effective_service_s=self.first_service,failed_attempts=self.failed_attempts,
                    collisions=self.collisions,explored_fraction=float(self.explored.mean()))
