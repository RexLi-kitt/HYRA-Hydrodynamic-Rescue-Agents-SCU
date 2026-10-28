"""Audit every saved preflight training transition and greedy evaluation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
import torch
from torch.nn import functional as F
from hyra_learning.networks import SearchAgent
from hyra_learning.observations import encode
from hyra_learning.team_env import TeamConfig, TeamEnv
from hyra_learning.training import central_state, source_files
from hyra_uav.core import ACTIONS, Config, FloodProcess, Memory, footprint
from hyra_uav.interfaces import merge_arrived


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    root = ROOT/'generated/sur-preflight-20261025'
    plan, summary = read(root/'plan.json'), read(root/'summary.json')
    assert summary['status'] == 'completed'
    assert plan['source_sha256'] == source_files()
    processes = {entry['id']: FloodProcess(ROOT/entry['path']) for entry in plan['data']}
    processes_by_hash = {p.digest: p for p in processes.values()}
    methods = []
    for result in summary['methods']:
        folder = root/result['method']
        pointer = read(folder/'latest.json')
        assert pointer['episode'] == 12
        assert sha256((folder/pointer['path']).read_bytes()).hexdigest() == pointer['sha256']
        state = torch.load(folder/pointer['path'], map_location='cpu', weights_only=True)
        assert state['episode'] == 12 and state['updates'] == 9 and len(state['replay']) == 12
        previous = torch.load(folder/'checkpoint-011.pt', map_location='cpu', weights_only=True)
        assert previous['updates'] == 8
        for module, key in (('actor', 'target_actor'), ('mixer', 'target_mixer')):
            for name, value in previous[module].items():
                torch.testing.assert_close(value, state[key][name], atol=0, rtol=0)
        transitions = 0
        for history, data in zip(state['history'], state['replay']):
            env = TeamEnv(processes[history['event']], Config(**plan['config']), TeamConfig(**plan['team']))
            view, _ = env.reset()
            assert data['images'].shape == (61, 2, 16, 7, 7)
            assert data['states'].shape == (61, 137)
            total = 0.
            for t in range(61):
                encoded = encode(view)
                np.testing.assert_array_equal(encoded['images'], data['images'][t].numpy())
                np.testing.assert_array_equal(encoded['poses'], data['poses'][t].numpy())
                np.testing.assert_array_equal(central_state(env), data['states'][t].numpy())
                if t == 60:
                    break
                view, reward, _, done, _ = env.step(data['actions'][t].numpy())
                np.testing.assert_allclose(reward, float(data['rewards'][t]), rtol=1e-6, atol=1e-9)
                assert bool(data['terminal'][t]) == done == (t == 59)
                total += reward
                transitions += 1
            assert abs(total-history['metrics']['return_']) < 1e-12
        actor = SearchAgent().to(plan['runtime']['device'])
        actor.load_state_dict(state['actor'])
        actor.eval()
        evaluated = []
        for path in sorted(folder.glob('evaluation-*.json')):
            replay = read(path)
            process = processes_by_hash[replay['source_sha256']]
            env = TeamEnv(process, Config(**plan['config']), TeamConfig(**plan['team']))
            view, _ = env.reset()
            hidden = actor.initial_hidden(2)
            previous_action = torch.zeros(2, 4, device=hidden.device)
            for t, action_row in enumerate(replay['actions']):
                encoded = encode(view)
                with torch.no_grad():
                    q, hidden, _, _ = actor(torch.as_tensor(encoded['images'], device=hidden.device),
                                             torch.as_tensor(encoded['poses'], device=hidden.device),
                                             previous_action, t, hidden, enhance=False)
                actions = np.array([ACTIONS.index(a) for a in action_row['actions']])
                np.testing.assert_array_equal(q.argmax(dim=-1).cpu().numpy(), actions)
                view, _, _, _, _ = env.step(actions)
                previous_action = F.one_hot(torch.as_tensor(actions, device=hidden.device), 4).float()
            rebuilt = env.result()
            assert rebuilt['summary'] == replay['summary']
            assert rebuilt['observations'] == replay['observations']
            assert rebuilt['snapshots'] == replay['snapshots']
            assert len(replay['snapshots']) == 121 and len(replay['observations']) == 242
            evaluated.append(path.name)
        assert transitions == 720 and len(evaluated) == 3
        methods.append(dict(method=result['method'], training_transitions_rebuilt=transitions,
                            evaluation_episodes_rebuilt=len(evaluated),
                            all_greedy_actions_from_clean_actor_reproduced=True,
                            target_schedule_verified=True, capacity=result['capacity']))
    output = ROOT/'docs/evidence/20261025-sur-adaptation'
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status='passed', methods=methods, training_joint_steps=1440, evaluation_joint_steps=360,
                  source_sha256=plan['source_sha256'],
                  checks=['all training images and poses rebuilt from legal union observations',
                          'all current training states rebuilt separately from actor encoding',
                          'all finite-horizon terminal flags and rewards', 'target update after optimizer step8',
                          'all greedy evaluation actions, actual-frame observations and snapshots',
                          'same actual core actor/mixer capacity and completed budget'],
                  limitation='Previously seen engineering fixtures and one seed; no policy effectiveness or physical-learning conclusion.')
    (output/'preflight-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(methods))


if __name__ == '__main__':
    main()
