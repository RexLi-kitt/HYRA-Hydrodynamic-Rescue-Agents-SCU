"""Cross flood direction and dry intervals using identical latent draws within a pair."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "uav-short-window-training-20261105"
sys.path.insert(0, str(PREVIOUS))
from common import np, K, response, history, MissionSupply, read, save, sha, quota, manifest
from campaign import policy, WEIGHT
from concurrent.futures import ProcessPoolExecutor

SEEDS = range(86051001, 86051009)
ANGLES = (0,90,180,270)
ARMS = ("continuous","rewet")
PROTOCOL = HERE/"protocol.json"

def base(seed):
    rng = np.random.default_rng(seed)
    depth, speed = rng.uniform(2.05,2.8), rng.uniform(.35,.45)
    intervals = [(c + int(rng.integers(-60,61))) for c in (450,1030,1600)]
    prng, crng = map(np.random.default_rng, np.random.SeedSequence([91821,seed,0]).spawn(2))
    ids = np.setdiff1d(np.arange(10000), [0,99,9900,9999])
    central = (seed-SEEDS.start)%2==0
    low,high = (200,800) if central else (50,950)
    traps = np.random.default_rng(np.random.SeedSequence([7811,seed])).uniform(low,high,(12,2))
    return dict(seed=seed, depth=depth, speed=speed, dry_start_s=intervals, spatial_class="central" if central else "spread",
                xy_flat=prng.choice(ids,48,replace=False).tolist(), clock=crng.exponential(size=48).tolist(),
                traps=traps.tolist(), refloat_salt=int(np.random.SeedSequence([7812,seed,0]).generate_state(1)[0]))

def fields(scene, angle):
    rad = np.deg2rad(angle)
    wet = np.tile(np.float32([scene["depth"],scene["speed"]*np.cos(rad),scene["speed"]*np.sin(rad)]),(481,1))
    mask = np.zeros(481,bool)
    for start in scene["dry_start_s"]:
        mask[start//5:(start+40)//5+1]=True
    dry = wet.copy()
    dry[mask] = 0
    assert np.array_equal(wet[~mask],dry[~mask]) and (dry[mask]==0).all()
    assert np.array_equal(np.any(wet!=dry,axis=1),mask)
    return wet,dry

def max_dose(field):
    a = field.astype(float)
    dose = 0.
    for sub in range(5):
        h,x,y = (a[:-1]+(sub+.5)/5*(a[1:]-a[:-1])).T
        dose += K*response(h,np.hypot(x,y)).sum()
    return dose

def run_parent(seed):
    scene = base(seed)
    folder = HERE/"pairs"/str(seed)
    folder.mkdir(exist_ok=True,parents=True)
    save(folder/"scene.json",scene)
    weights = sha(WEIGHT)
    lock = read(PREVIOUS/"selection.json")
    assert lock["weight_sha"]==weights and lock["choice"]["best"]=="tracker"
    assert all(sha(PREVIOUS/n)==v for n,v in lock["source_hashes"].items())
    for angle in ANGLES:
        out=folder/f"pair-{angle}.json"
        records=[]
        for arm,frames in zip(ARMS,fields(scene,angle)):
            field_path=folder/f"{angle}-{arm}.npy"
            np.save(field_path,frames)
            assert np.isfinite(frames).all() and (frames[:,0]>=0).all()
            w=np.broadcast_to(frames[:,:,None,None],(481,3,100,100))
            layout={k:scene[k] for k in ("xy_flat","clock")}
            layout["first_fall_by_end"]=(np.array(scene["clock"])<=max_dose(frames)).tolist()
            pop=history(w,layout,seed,0,np.array(scene["traps"]))
            controller=policy("tracker")
            env=MissionSupply(w,pop,(48,1140,2280))
            while env.t<2280:
                env.step(*controller.act(env.observation()))
                if (env.supplied>=0).all() and (env.last>=0).all():break
            metrics=env.metrics()
            assert sum(t>=0 for t in metrics["delivery_s"])==metrics["delivered"]
            records.append(dict(arm=arm,field_sha=sha(field_path),fall_s=pop["fall"].tolist(),**metrics))
        assert np.all(np.asarray(records[0]["fall_s"])<=records[1]["fall_s"])
        save(out,dict(seed=seed,angle=angle,scene_sha=sha(folder/"scene.json"),weights_sha=weights,
                      protocol_sha=sha(PROTOCOL),code_sha=sha(Path(__file__)),arms=records))
    return seed

def analyze():
    results=[]
    for seed in SEEDS:
        for angle in ANGLES:
            r=read(HERE/"pairs"/str(seed)/f"pair-{angle}.json")
            assert r["scene_sha"]==sha(HERE/"pairs"/str(seed)/"scene.json")
            a,b=r["arms"]
            results.append([seed,angle,a["delivered"],b["delivered"]])
    x=np.array(results).reshape(len(SEEDS),len(ANGLES),4)
    totals=x[:,:,2:].sum(axis=(0,1));n=len(SEEDS)*4*48
    delta=(x[:,:,3]-x[:,:,2]).mean(axis=1)*100/48
    rng=np.random.default_rng(20261105)
    estimates=np.zeros(10000)
    for values in (delta[::2],delta[1::2]):
        estimates+=.5*rng.choice(values,(10000,len(values))).mean(axis=1)
    v=dict(parents=len(SEEDS),pairs=len(results),episodes=len(results)*2,people_per_arm=n,
           delivered=totals.tolist(),rate=(totals/n).tolist(),delta_pp=float(delta.mean()),
           delta_ci_pp=np.quantile(estimates,[.025,.975]).tolist(),by_direction=x[:,:,2:].sum(axis=0).tolist(),
           parent_delta_pp=delta.tolist(),rows=results)
    save(HERE/"paired-results.json",v)
    print("Final",totals,n,v["delta_pp"],v["delta_ci_pp"],flush=True)
if __name__=="__main__":
    quota()
    assert not set(SEEDS).intersection(manifest())
    # One campaign, fixed parents, finite work. Leave completed pairs on disk.
    with ProcessPoolExecutor(max_workers=8) as workers:
        for seed in workers.map(run_parent,SEEDS):
            print("complete",seed,flush=True)
    analyze()

