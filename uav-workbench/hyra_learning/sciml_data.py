"""New parent terrains/forcing families and training-only hydraulic labels.

Actor ZIPs keep the 5s actual clock. Auxiliary ten-second targets and a short
accepted internal-step rate come from the SAME simulation, never interpolation.
Confirmation water is generated only after all selected policy hashes freeze.
"""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import uuid

from .runtime import enable_existing_packages
enable_existing_packages()
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TERRAIN = ROOT.parent/'terrain-workbench/terrain-lab'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def seed(name):
    return int(sha256(('hyra-independent-sciml-r1:'+name).encode()).hexdigest()[:8], 16)


def cases(design):
    result = []
    for split, families in design['terrain_families'].items():
        for family in families:
            forcings = design['forcing_families'][split]
            variants = range(3) if split == 'selection' else range(1)
            for forcing in forcings:
                for variant in variants:
                    name = f'{family}-{forcing}-v{variant}'
                    result.append(dict(id=name, split=split, family=family, parent_terrain=family,
                                       forcing_family=forcing, variant=variant, seed=seed(name), parent_seed=seed(family)))
    return result


def terrain(family, parent_seed):
    z, x = np.indices((32, 32), dtype=float)
    rng = np.random.default_rng(parent_seed)
    phase = rng.uniform(-.4, .4)
    if family == 'confluence-runnels':
        bed = np.minimum(.001*(z-8-2*np.sin(x/6+phase))**2+.004*x,
                         .0013*(x-23-2*np.cos(z/7+phase))**2+.003*z)+.008*np.cos((x+z)/3)
    elif family == 'fan-corridor':
        r = np.hypot(x-3, z-17)
        bed = .006*r+.03*np.cos(5*np.arctan2(z-17, x-3)+phase)*np.exp(-r/18)+.001*z
    elif family == 'paired-depressions':
        bed = np.minimum(.0008*((x-9)**2+1.3*(z-11)**2),
                         .0011*((x-23)**2+.6*(z-22)**2))+.012*np.sin((x-2*z)/5+phase)
    elif family == 'sinuous-bench':
        bed = .025*np.floor((z+2*np.sin(x/7+phase))/3)+.0005*(x-17)**2+.005*np.cos(x/3+phase)
    elif family == 'constricted-neck':
        bed = .004*x+.10*np.exp(-((x-16)/4)**2)*(1-np.exp(-((z-16)/3)**2))+.0005*(z-16)**2
        bed += .003*np.sin(z/3+phase)
    elif family == 'branching-gully':
        branch = 3+np.maximum(x-12, 0)*.4
        bed = .004*x+.001*np.minimum((z-16-branch)**2, (z-16+branch)**2)+.006*np.sin(x/3+phase)
    elif family == 'diagonal-islands':
        bed = .002*(x+z)+.11*np.exp(-((x-9)**2+(z-21)**2)/24)+.09*np.exp(-((x-23)**2+(z-8)**2)/18)
        bed += .018*np.sin((x+2*z)/6+phase)*np.sin((2*x-z)/7)
    else:
        raise ValueError('Unregistered parent terrain')
    return bed-bed.min()


def hydrographs(case):
    peak = float(np.random.default_rng(case['seed']).uniform(22, 36))
    kind = case['forcing_family']
    if kind == 'rectangular':
        a, b = [(0,0),(70,peak),(230,0)], [(0,0),(300,.5*peak),(440,0)]
    elif kind == 'ascending-staircase':
        a, b = [(0,0),(60,.25*peak),(120,.5*peak),(180,peak),(260,0)], [(0,0),(330,.5*peak),(400,.8*peak),(460,0)]
    elif kind == 'falling-staircase':
        a, b = [(0,0),(80,peak),(160,.6*peak),(240,.2*peak),(300,0)], [(0,0),(360,.8*peak),(420,.4*peak),(480,0)]
    elif kind == 'offset-doublet':
        shift = (case['variant']-1)*10
        a = [(0,0),(80+shift,.8*peak),(140+shift,0),(240+shift,peak),(310+shift,0)]
        b = [(0,0),(370+shift,.6*peak),(440+shift,0)]
    elif kind == 'late-surge':
        a, b = [(0,0),(220,.25*peak),(290,peak),(340,1.3*peak),(400,0)], [(0,0),(440,.7*peak),(500,0)]
    elif kind == 'gapped-train':
        a, b = [(0,0),(60,peak),(100,0),(180,.8*peak),(220,0),(300,1.1*peak),(350,0)], [(0,0),(400,.8*peak),(430,0),(480,.6*peak),(520,0)]
    elif kind == 'compound-recession':
        a, b = [(0,0),(50,peak),(120,.85*peak),(210,.55*peak),(280,.2*peak),(360,0)], [(0,0),(260,.7*peak),(330,.4*peak),(410,.1*peak),(480,0)]
    else:
        raise ValueError('Unregistered forcing family')
    return {'0': a, '1': b}


def generate(case, case_root):
    os.environ['HYRA_FLOOD_BACKEND'] = 'numpy'
    sys.path.insert(0, str(TERRAIN))
    import flood_run
    from flood_solver import FloodSolver
    from multilayer import MapWorld
    from map_sessions import save_process
    # Incomplete attempts are retained; only a completed, indexed attempt is
    # immutable evidence. Conversation interruption never overwrites history.
    folder = case_root/('attempt-'+uuid.uuid4().hex[:10])
    frame_folder = folder/'generated/event/maps'
    frame_folder.mkdir(parents=True, exist_ok=False)
    bed = terrain(case['family'], case['parent_seed'])
    source = np.full((32,32), -1)
    source[6,4], source[25,27] = 0, 1
    config = flood_run.settings(dict(resolution=32,duration_s=600,rain_mm_h=0,rain_duration_s=0,
                                    infiltration_mm_h=0,boundary='closed',baseflow=False,warmup_s=0),(32,32))
    config['frames'] = 121
    meta = dict(config=config,coarsening_factor=1,origin_blocks=[0,0],
                terrain_source='independent_sciml_synthetic_development',terrain_id=case['parent_terrain'],
                initial_conditions=dict(duration_s=0,converged=False,reason='no_active_baseflow'))
    world = MapWorld(FloodSolver(bed,np.zeros_like(bed),5,boundary='closed',manning=.035),meta,
                     dict(source_id=source,hydrographs=hydrographs(case),drain_capacity=np.full_like(bed,.04)))
    frames, fields, rates, steps, sources = [], [], [], [], []
    def conserved():
        s = world.solver
        return np.stack((s.h,s.mx,s.mz)).copy()
    for i, now in enumerate(range(0,601,5)):
        world.advance_to(float(now))
        s = world.solver
        current = conserved()
        fields.append(current.astype(np.float32))
        u,v = s.velocity()
        name = f'frame-{i:03d}.f32'
        np.stack((s.h,u,v,s.maximum)).astype('<f4').tofile(frame_folder/name)
        frames.append(dict(index=i,url='/generated/event/maps/'+name,**s.summary()))
        if now < 600 and now % 10 == 0:
            s.source_rate = s._scheduled_rate(s.time)
            dt = min(.01,.5*s.timestep(limit=1.))
            sources.append(s.source_rate.astype(np.float32).copy())
            s.step(dt)
            rates.append(((conserved()-current)/dt).astype(np.float32))
            steps.append(dt)
    save_process(folder,'event',world,frames)
    (frame_folder/'map-result.zip').replace(folder/'map-result.zip')
    np.savez_compressed(folder/'auxiliary-labels.npz',U=np.stack(fields),rate=np.stack(rates),dt=np.asarray(steps),
                        source=np.stack(sources),bed=bed,drains=world.base.drain_capacity.astype(np.float32),
                        manning=world.base.n.astype(np.float32),solid=world.base.solid,
                        wall_x=world.base.wall_x,wall_y=world.base.wall_y,times_s=np.arange(0,601,5),anchor_times_s=np.arange(0,600,10))
    scene = dict(**case,path=str((folder/'map-result.zip').resolve()),auxiliary_path=str((folder/'auxiliary-labels.npz').resolve()),
                 kind='independent_sciml_hydraulic_development',source_sha256=digest(folder/'map-result.zip'),
                 auxiliary_sha256=digest(folder/'auxiliary-labels.npz'),bed_sha256=sha256(bed.tobytes()).hexdigest(),
                 mass_residual_max_m3=max(abs(f['balance']['residual_m3']) for f in frames),
                 auxiliary_dt_min_s=min(steps),auxiliary_dt_max_s=max(steps))
    write(case_root/'scene.json', scene)
    return scene


def preparation_sources():
    paths = [Path(__file__),ROOT/'hyra_learning/physics.py']
    paths += [TERRAIN/name for name in ('flood_solver.py','flood_numerics.py','flood_run.py','multilayer.py','map_sessions.py')]
    return {str(p.resolve()):digest(p) for p in paths}


def prepare(root, splits=('fit','selection')):
    design_path = ROOT/'learning_configs/sciml_round1_design_20261025.json'
    if (root/'data-plan.json').exists():
        plan = read(root/'data-plan.json')
        if plan['design_sha256'] != digest(design_path) or plan['generation_sources'] != preparation_sources():
            raise ValueError('Registered generation code or design changed; preserve existing data')
    else:
        if root.exists():
            raise ValueError('New data study needs a new empty output root')
        ledger = ROOT/'research/sciml_rounds.json'
        if ledger.exists():
            raise ValueError('The one independent mechanism round is already registered')
        root.mkdir(parents=True)
        design = read(design_path)
        plan = dict(design=design,cases=cases(design),design_sha256=digest(design_path),generation_sources=preparation_sources())
        write(root/'data-plan.json',plan)
        write(ledger,dict(maximum_rounds=1,rounds=[dict(round=1,output=str(root.resolve()),state='data_preregistered')]))
    if 'holdout' in splits:
        freeze = read(root/'model-freeze.json')
        if len(freeze['selected_models']) != 12:
            raise ValueError('All twelve selected checkpoints must freeze before confirmation water')
        for row in freeze['selected_models']:
            if digest(root/row['path']) != row['sha256']:
                raise ValueError('Selected checkpoint changed')
    entries = read(root/'index.json') if (root/'index.json').exists() else []
    for case in plan['cases']:
        if case['split'] not in splits:
            continue
        old = next((e for e in entries if e['id'] == case['id']),None)
        if old:
            if digest(Path(old['path'])) != old['source_sha256'] or digest(Path(old['auxiliary_path'])) != old['auxiliary_sha256']:
                raise ValueError('Completed scene input changed')
            continue
        case_root = root/'scenes'/case['id']
        scene = read(case_root/'scene.json') if (case_root/'scene.json').exists() else generate(case,case_root)
        entries.append(scene)
        write(root/'index.json',entries)
        print('prepared',case['id'],flush=True)
    return entries


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    parser.add_argument('--splits',nargs='+',choices=('fit','selection','holdout'),default=['fit','selection'])
    args=parser.parse_args();prepare(args.output.resolve(),args.splits)
