"""Fit-only read-only gradient-path audit of selected AND final checkpoints.

The frozen training logger samples shared gradients only through update253.
This independent diagnostic measures the saved late weights without claiming
that a post-hoc probe reconstructs every earlier optimizer update.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from scale100_train import torch, np, F, SWEOperator, FloodCase, ScaleEnv, config, digest, write
from scale100_backend import make_actor, validate_meta
from scale100_formal import validate_plan, check_selected
from scale100_run_audit import require
from scale100_rules import water_aware

ROOT = Path(__file__).resolve().parent
FORMAL = ROOT / 'generated/scale100-formal-20261026'
INDICES = (0, 60, 120, 180)
PREFIXES = ('conv.', 'image.', 'gru.')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def fingerprint(actor):
    value = hashlib.sha256()
    for name, tensor in sorted(actor.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        value.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        value.update(array.tobytes())
    return value.hexdigest()


def norm(grads):
    values = [g.detach().square().sum() for g in grads if g is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


def probe_state(actor, case, view, previous, previous_hidden, step):
    shared = [(name, p) for name, p in actor.named_parameters() if name.startswith(PREFIXES)]
    with torch.enable_grad():
        _, hidden, _ = actor(torch.from_numpy(view.images.copy()), previous, step,
                             previous_hidden.detach(), False)
        bed = torch.tensor(case.bed, dtype=torch.float32)[None].expand(2, -1, -1)
        pre = actor.aux[0](torch.cat((hidden, bed.flatten(1)), dim=-1))
        predicted = actor.predict(hidden, bed)
        before = torch.tensor(case.probes['before'][step], dtype=torch.float32)[None]
        after = torch.tensor(case.probes['after'][step], dtype=torch.float32)[None]
        scale = torch.tensor([.5, .25, .25])[None, :, None, None]
        data = (((predicted[:, 0]-before)/scale).square().mean()
                + ((predicted[:, 1]-after)/scale).square().mean())/2
        actual_step = float(case.probes['dt'][step])
        stepped = SWEOperator().step(predicted[:, 0], bed, actual_step,
            source=torch.tensor(case.probes['source'][step], dtype=torch.float32))
        physics = ((predicted[:, 1]-stepped)/scale).square().mean()
        gd = torch.autograd.grad(data, [hidden] + [p for _, p in shared], retain_graph=True, allow_unused=True)
        gp = torch.autograd.grad(physics, [hidden] + [p for _, p in shared], allow_unused=True)
        values = dict(time_s=step*5., actual_microstep_s=actual_step,
            data_loss=float(data.detach()), physics_loss=float(physics.detach()),
            data_to_latent_norm=norm(gd[:1]), physics_to_latent_norm=norm(gp[:1]),
            data_to_shared_norm=norm(gd[1:]), physics_to_shared_norm=norm(gp[1:]),
            positive_auxiliary_preactivation_fraction=float((pre.detach()>0).float().mean()),
            minimum_auxiliary_preactivation=float(pre.detach().min()),
            maximum_auxiliary_preactivation=float(pre.detach().max()))
        for prefix in PREFIXES:
            indices = [i+1 for i, (name, _) in enumerate(shared) if name.startswith(prefix)]
            values['data_to_' + prefix[:-1] + '_norm'] = norm([gd[i] for i in indices])
            values['physics_to_' + prefix[:-1] + '_norm'] = norm([gp[i] for i in indices])
    require(all(math.isfinite(v) for v in values.values()), 'Nonfinite checkpoint mechanism diagnostic')
    values['data_nonzero_loss_zero_shared_gradient'] = values['data_loss']>0 and values['data_to_shared_norm']==0
    values['physics_nonzero_loss_zero_shared_gradient'] = values['physics_loss']>0 and values['physics_to_shared_norm']==0
    return values, hidden.detach()


def checkpoint_probe(actor, case, task):
    before = fingerprint(actor)
    actor.eval()
    env = ScaleEnv(case, config(task, 'D', 11))
    view = env.reset()
    hidden = actor.hidden(2)
    previous = torch.zeros(2, 8)
    rows = []
    for step in range(181):
        if step in INDICES:
            value, hidden = probe_state(actor, case, view, previous, hidden, step)
            rows.append(dict(parent=case.case['parent_terrain'], **value))
        else:
            with torch.no_grad():
                _, hidden, _ = actor(torch.from_numpy(view.images.copy()), previous, step, hidden, False)
        if step < 180:
            actions = water_aware(env)
            view, _, _ = env.step(actions)
            previous = F.one_hot(torch.as_tensor(actions), 8).float()
    require(fingerprint(actor) == before and all(p.grad is None for p in actor.parameters()),
            'Read-only diagnostic changed model tensors or accumulated optimizer gradients')
    return rows


def execute(job_id, output):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    require(not output.exists(), 'Refusing to overwrite checkpoint mechanism evidence')
    plan = read(FORMAL / 'plan.json')
    validate_plan(plan)
    job = next((j for j in plan['jobs'] if j['id'] == job_id), None)
    require(job is not None and job['method'] in ('D', 'P'), 'Choose a registered D/P job')
    require(job_id in read(FORMAL / 'queue.json')['completed'], 'This formal job has not completed selection')
    model = FORMAL / job_id
    selected = check_selected(model, job, plan, digest(FORMAL / 'plan.json'))
    final = read(model / 'actor.json')
    validate_meta(final, plan['training_backend'])
    require((final['task'], final['method'], final['seed'], final['trained_episodes'], final['updates'], final['formal'])
            == (job['task'], job['method'], job['seed'], 1024, 1021, True)
            and final['actor_sha256'] == digest(model / 'actor.pt'), 'Wrong final checkpoint')
    cases = [FloodCase(row['path']) for row in plan['inputs'] if row['split'] == 'fit']
    results = []
    cache = {}
    for kind, folder, meta in (('selected', model / 'selected', selected), ('final', model, final)):
        sha = meta['actor_sha256']
        if sha not in cache:
            actor = make_actor(plan['training_backend'])
            actor.load_state_dict(torch.load(folder / 'actor.pt', weights_only=True, map_location='cpu'))
            actor.eval()
            cache[sha] = [row for case in cases for row in checkpoint_probe(actor, case, job['task'])]
        rows = cache[sha]
        results.append(dict(kind=kind, episode=meta['trained_episodes'], actor_sha256=sha, states=len(rows), rows=rows,
            nonzero_data_shared_states=sum(r['data_to_shared_norm']>0 for r in rows),
            nonzero_physics_shared_states=sum(r['physics_to_shared_norm']>0 for r in rows),
            data_nonzero_loss_zero_shared_states=sum(r['data_nonzero_loss_zero_shared_gradient'] for r in rows),
            physics_nonzero_loss_zero_shared_states=sum(r['physics_nonzero_loss_zero_shared_gradient'] for r in rows)))
    result = dict(schema='hyra-scale100-checkpoint-mechanism-v1', job=job, formal_plan_sha256=digest(FORMAL/'plan.json'),
        source_sha256=digest(__file__), checkpoint_results=results, split='fit', scenario_seed=11,
        training_or_selection_changed=False, heldout_fields_loaded=False, heldout_policy_scored=False,
        physics_used_in_training=job['method']=='P', history_gradient_last_sample_episode=256,
        limitations=['Fit-only diagnostic of saved selected/final weights; not every past update',
            'Previous recurrent state is held fixed for each probed decision; tests the current input/GRU path',
            'Nonzero possible gradient is not evidence of positive policy benefit',
            'For D, the physical gradient is only a potential path; no physics term was applied during D training',
            'All-negative auxiliary preactivations are not dead units under the registered LeakyReLU'])
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, result)
    print(json.dumps(dict(job=job_id, checkpoints=[{k:r[k] for k in ('kind','episode','states',
        'nonzero_data_shared_states','nonzero_physics_shared_states')} for r in results])), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    execute(args.job, args.output)
