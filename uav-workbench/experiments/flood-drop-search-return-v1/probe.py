"""TRAIN-only full-return search macro-action feasibility probe; no fitted policy."""
from pathlib import Path
import copy, hashlib, json, pickle, sys, time, traceback
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = REPO.parent
BASE = HERE.parent / 'flood-drop-convlstm20-v1'
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE.parent / 'flood-drop-fine-memory-v1'))
from long_flow import FlowFogEnv
from memory import public_features

PREVIOUS = ROOT / 'uav-fine-memory-20261104/run01'
OUT = ROOT / 'uav-search-return-20261104/run01'

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def write(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(p)
def clone(e): return copy.deepcopy(e, {id(e.water): e.water})
def fingerprint(e):
    return hashlib.sha256(pickle.dumps({k:v for k,v in e.__dict__.items() if k != 'water'})).hexdigest()

class SegmentEnv(FlowFogEnv):
    def __init__(self, water, scenario):
        self.segment = None; self.segment_end = None; self.segment_steps = 0
        super().__init__(water, scenario, 'coverage100')

    def arm(self, proposal):
        if proposal is None: return
        self.segment = dict(uav=int(proposal['uav']), xy=np.array(proposal['xy'], float),
                            until=self.t+30, discovered=self.discovered.copy())
        self.segment_end = None

    def choices(self):
        mask, actions, options, features = super().choices()
        s = self.segment
        if s is None: return mask, actions, options, features
        i = s['uav']; reason = None
        if self.t >= s['until']: reason = '30s_limit'
        elif np.any(self.discovered & ~s['discovered']): reason = 'new_discovery'
        elif options[i][actions[i]]['person'] >= 0: reason = 'known_rescue_assignment'
        elif np.linalg.norm(s['xy']-self.pos[i]) <= self.cfg.speed*self.cfg.step: reason = 'goal_reached'
        if reason is not None:
            self.segment_end = dict(time=self.t, reason=reason); self.segment = None
            return mask, actions, options, features
        # Change one currently idle UAV's search target only. All rescue orders,
        # physical movement, collision checks and 10s continuous service stay intact.
        options[i][8] = dict(xy=s['xy'].copy(), person=-1, utility=0., region=-1)
        mask[i,8] = True; actions[i] = 8
        features[i,8] = [1.,0.,*list(s['xy']/1000),*list((s['xy']-self.pos[i])/1000),0.,0.,0.,0.]
        return mask, actions, options, features

    def step(self, actions):
        self.choices()  # Apply expiry/interrupt before recording active duration.
        self.segment_steps += int(self.segment is not None)
        return super().step(actions)

def proposals(e):
    mask, actions, opts, _ = e.choices()
    idle = [i for i in range(e.cfg.uavs) if opts[i][actions[i]]['person'] < 0
            and opts[i][actions[i]]['region'] >= 0]
    result = [None]
    if not idle: return result
    # Deterministic round-robin among idle UAVs, never chosen by label outcomes.
    i = idle[(e.t//480-1) % len(idle)]
    selected = int(actions[i])
    for slot in range(4,8):
        if slot != selected and mask[i,slot]:
            result.append(dict(uav=i, xy=opts[i][slot]['xy'].tolist(), region=opts[i][slot]['region']))
    return result

def context(e, ps):
    _, aa, oo, _ = e.choices()
    shared = np.r_[e.pos.ravel()/1000,
                   np.array([oo[i][aa[i]]['xy'] for i in range(e.cfg.uavs)]).ravel()/1000,
                   [oo[i][aa[i]]['person'] >= 0 for i in range(e.cfg.uavs)],
                   e.t/e.cfg.duration, e.known.mean(),
                   np.sum((e.belief==1)&e.discovered)/e.cfg.people]
    rows = []
    for p in ps:
        if p is None: extra = np.zeros(10)
        else:
            i = p['uav']; goal = np.array(p['xy']); onehot = np.eye(e.cfg.uavs)[i]
            extra = np.r_[1.,onehot,goal/1000,(goal-e.pos[i])/1000,
                          np.linalg.norm(goal-e.pos[i])/e.cfg.speed/e.cfg.duration]
        rows.append(np.r_[shared,extra])
    return np.array(rows, np.float32)

def result(e):
    persons = [dict(id=j,fall=int(e.fall[j]),deadline=int(e.deadline[j]),
        first_seen=float(e.first_seen[j]),supplied_at=float(e.delivered_at[j]),status=int(e.status[j]))
        for j in range(e.cfg.people)]
    assert np.isfinite(e.pos).all() and np.isfinite(e.person).all() and (e.status>=2).all()
    for p in persons:
        if p['status']==2: assert p['fall']+10<=p['supplied_at']<=p['deadline']
    return dict(supplied=int((e.status==2).sum()),people=e.cfg.people,persons=persons,
        refloats=int(e.refloat_counts.sum()),segment_steps=e.segment_steps,
        segment_end=e.segment_end,final_position=e.pos.tolist(),
        all_scene=bool((e.status==2).all()),p95_scene=bool((e.status==2).sum()>=.95*e.cfg.people))

def run():
    OUT.mkdir(parents=True,exist_ok=False)
    # These are the first four newly generated FIT parents, never CAL/DEV/test.
    cases = read(PREVIOUS/'scenarios.json')['fit'][12:16]
    assert len(cases)==4 and len({c['field']['parent'] for c in cases})==4
    inputs = {str(PREVIOUS/'scenarios.json'):sha(PREVIOUS/'scenarios.json')}
    inputs.update({c['field']['path']:sha(c['field']['path']) for c in cases})
    sources = {str(p):sha(p) for d in (HERE,BASE,HERE.parent/'flood-drop-fine-memory-v1',
               HERE.parent/'flood-drop-search-redesign-v1') for p in d.glob('*.py')}
    protocol = dict(date='2026-11-04',stage='phase2 feasibility only, no model fitted',
        parents=[c['field']['parent'] for c in cases],times=[480,960,1440],macro_seconds=30,
        baseline='Complete coverage100 planner unchanged, also used for roll-in and continuation.',
        proposals='Default plus three other-region search goals for one deterministic round-robin idle UAV. No idle UAV: retain unavailable state, do not force a rescue interruption.',
        execution='Fixed target at most30s; interrupt at next1s decision on any new observation of a person, existing rescue assignment, arrival or expiry. Others retain full planner.',
        label='Final supplied people after single search segment then full planner to episode end, minus default. Privileged truth used only in TRAIN offline label evaluation.',
        inputs='64x10x10 public observation memory plus33 public action/context features; no true future/fall/deadline/hidden position.',
        random='Existing counter-based BLAKE2 seed/exposure_id/time refloat draws shared by all branches; pre-generated same fall scenario and read-only same water.',
        budget='12fixed states, at most48 full continuations +1 repeated branch; 4roll-in baselines; first baseline full stepwise fallback comparison. No GPU. One workload, 1800s guard.',
        gate='At least5 positive and5 negative nondefault labels, each sign present in>=2 parents. Passing allows larger TRAIN collection, not a claim of learned improvement.',
        limits='12states insufficient to train or validate a neural policy. Gains at different times share an episode and are NOT additive; oracle best branch not a deployable score. Current policy-state distribution only.',
        unchanged='20m circular sensor,420s window,2100s horizon,40people,4UAV,10s continuous drop,2m/1mps triggers. No weakened planner or changed physics.',
        input_sha256=inputs,source_sha256=sources)
    write(OUT/'protocol.json',protocol);write(OUT/'scenarios.json',cases)
    start=time.monotonic();records=[];audits=[];completed=0
    def guard():
        if (OUT/'STOP').exists(): raise InterruptedError('STOP requested')
        if time.monotonic()-start>1800: raise TimeoutError('1800s budget')
    def status(state,**kw):
        write(OUT/'status.json',dict(state=state,elapsed_s=time.monotonic()-start,
            states=len(records),continuations=completed,**kw))
    def finish(e):
        while e.t<e.cfg.duration and not (e.status>=2).all():
            if e.t%120==0: guard()
            if e.step(e.choices()[1])[2]: break
        return result(e)
    try:
        for ci,c in enumerate(cases):
            guard();status('roll_in',case=c['id'])
            water=np.load(c['field']['path'],mmap_mode='r',allow_pickle=False)
            assert not water.flags.writeable
            e=SegmentEnv(water,c['scenario']);snapshots=[]
            original=FlowFogEnv(water,c['scenario'],'coverage100') if ci==0 else None
            while True:
                if e.t%120==0:guard()
                if e.t in protocol['times']:snapshots.append(clone(e))
                a=e.choices()
                if original is not None:
                    b=original.choices()
                    for k in (0,1,3):np.testing.assert_array_equal(a[k],b[k])
                    for i in range(e.cfg.uavs):np.testing.assert_array_equal(a[2][i][a[1][i]]['xy'],b[2][i][b[1][i]]['xy'])
                    original.step(b[1])
                done=e.step(a[1])[2]
                if original is not None:
                    for name in ('pos','person','status','delivered_at','first_seen','known','last'):
                        np.testing.assert_array_equal(getattr(e,name),getattr(original,name))
                if done:break
            baseline=result(e);write(OUT/f'baseline-{ci}.json',baseline)
            if original is not None:audits.append(dict(test='full_episode_default_exact',steps=e.t,passed=True))
            for state in snapshots:
                ps=proposals(state);x=context(state,ps);f=public_features(state)
                assert x.shape==(len(ps),33) and np.isfinite(x).all()
                other=clone(state);other.person[:]=999;other.status[:]=3
                other.fall[:]+=1;other.deadline[:]+=1;other.water=None
                assert ps==proposals(other)
                np.testing.assert_array_equal(x,context(other,ps));np.testing.assert_array_equal(f,public_features(other))
                audits.append(dict(case=c['id'],time=state.t,test='public_input_hidden_truth_isolation',passed=True))
                if len(ps)>1:
                    q=clone(state);q.arm(ps[1]);j=int(np.flatnonzero(~q.discovered)[0]);q.discovered[j]=True;q.choices()
                    assert q.segment is None and q.segment_end['reason']=='new_discovery'
                    q=clone(state);q.arm(ps[1]);i=ps[1]['uav'];j=0
                    q.discovered[j]=True;q.segment['discovered']=q.discovered.copy();q.belief[j]=1
                    q.seen_at[j]=q.t;q.first_seen[j]=q.t;q.visible[j]=True;q.seen_xy[j]=q.pos[i]
                    q.choices();assert q.segment is None and q.segment_end['reason']=='known_rescue_assignment'
                draws=[[state.refloat_draw(j,now) for now in (0,480,960)] for j in range(state.cfg.people)]
                assert draws==[[other.refloat_draw(j,now) for now in (0,480,960)] for j in range(state.cfg.people)]
                stamp=fingerprint(state);rows=[]
                np.savez_compressed(OUT/f'public-{ci}-{state.t}.npz',memory=f,context=x)
                for pi,p in enumerate(ps):
                    status('branching',case=c['id'],time=state.t,branch=pi)
                    b=clone(state);assert b.water is state.water;b.arm(p);row=finish(b);completed+=1
                    assert fingerprint(state)==stamp
                    if pi==0:
                        assert row['persons']==baseline['persons'] and row['final_position']==baseline['final_position']
                        audits.append(dict(case=c['id'],time=state.t,test='default_matches_unbranched_full_outcome',passed=True))
                    if ci==0 and state.t==480 and pi==1:
                        again=clone(state);again.arm(p);assert finish(again)==row;completed+=1
                        audits.append(dict(test='full_return_repeat_deterministic',passed=True))
                    assert row['segment_steps']<=30
                    rows.append(row)
                    write(OUT/f'branch-{ci}-{state.t}-{pi}.json',dict(proposal=p,**row))
                rec=dict(case=c['id'],parent=c['field']['parent'],time=state.t,proposals=ps,
                    supplied=[r['supplied'] for r in rows],gain=[r['supplied']-rows[0]['supplied'] for r in rows],
                    active_seconds=[r['segment_steps'] for r in rows])
                records.append(rec)
                with (OUT/'labels.jsonl').open('a',encoding='utf-8') as fp:fp.write(json.dumps(rec)+'\n')
                write(OUT/'audits.json',audits)
                print(json.dumps(rec),flush=True)
        for p,h in {**inputs,**sources}.items():assert sha(p)==h,p
        gains=[g for r in records for g in r['gain'][1:]]
        posparents={r['parent'] for r in records if any(g>0 for g in r['gain'][1:])}
        negparents={r['parent'] for r in records if any(g<0 for g in r['gain'][1:])}
        positive=sum(g>0 for g in gains);negative=sum(g<0 for g in gains)
        summary=dict(states=len(records),available_states=sum(len(r['proposals'])>1 for r in records),
            alternatives=len(gains),positive=positive,negative=negative,ties=sum(g==0 for g in gains),
            positive_parents=sorted(posparents),negative_parents=sorted(negparents),
            gate_passed=positive>=5 and negative>=5 and len(posparents)>=2 and len(negparents)>=2,
            full_continuations=completed,elapsed_s=time.monotonic()-start,trained_model=False,
            source_water_unchanged=True,formal_test=False)
        write(OUT/'summary.json',summary);status('completed');print(json.dumps(summary),flush=True)
    except BaseException as ex:
        write(OUT/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()));status('failed',error=repr(ex));raise

if __name__=='__main__':run()
