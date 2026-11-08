"""Training-only demonstrations. All labels use public observations."""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
sys.path.insert(0, str(V2))
import demo
import policy_core as p
import replay
import teachers
import train

s = p.geometry.search


def parent(seed):
    return (V2 / 'pilot_rescue_1000_88126' if seed == 88126 else V2 / 'formal' / f'rescue_{seed}') / 'checkpoint.pt'


def collect_hard(model, cases=512, device='cuda:0'):
    micro = teachers.load_micro_teacher()
    examples, records = [], []
    for i in range(cases):
        seed = 300000 + i
        bridge, goal, agent = micro.fixture(seed, 'unaligned')
        hidden = model.init_hidden(4, device)
        previous = np.full(4, -1, np.int64)
        labels = abstentions = 0
        for _ in range(16):
            obs = bridge.observation()
            context, hidden = replay.capture(model, obs, previous, hidden)
            action = micro.teacher(obs, agent)
            if action is None:
                examples.append((context, agent, -1)); abstentions += 1
                break
            mask = np.unpackbits(context['mask_bits'], axis=1, count=585).astype(bool)
            if not mask[agent, action]:
                raise AssertionError('Illegal hard teacher label')
            examples.append((context, agent, int(action))); labels += 1
            previous = np.full(4, 520, np.int64); previous[agent] = action
            bridge.step(previous // 65, previous % 65 - 1, local_action_order=False)
            if bridge.env.supplied[goal] >= 0:
                break
        records.append(dict(seed=seed, kind='unaligned', goal=goal, agent=agent,
                            labels=labels, abstentions=abstentions,
                            goal_delivered=bool(bridge.env.supplied[goal] >= 0)))
        if (i + 1) % 64 == 0:
            print('HARD_DEMO', i + 1, 'labels', sum(r['labels'] for r in records), flush=True)
    return examples, records


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args(); path = HERE / 'demonstrations.pt'
    if path.exists(): raise FileExistsError(path)
    torch.set_num_threads(1)
    model, ck = demo.load_model(parent(88126), a.device)
    lock = demo.source_lock()
    lock.update({str(f): p.sha(f) for f in (Path(__file__).resolve(), HERE/'protocol.json', parent(88126))})
    old, old_records = teachers.rescue_examples(model, 512, a.device)
    if demo.data_hash(old) != ck['dataset_sha256']:
        raise AssertionError('Original demonstration regeneration differs')
    hard, hard_records = collect_hard(model, device=a.device)
    manifest = dict(schema='DQN_RETAINED_DEMO_DATA_V4', source_lock=lock,
                    base_sha256=train.module_hash(model.base),
                    old_sha256=demo.data_hash(old), hard_sha256=demo.data_hash(hard),
                    old_records=old_records, hard_records=hard_records,
                    old_positive=sum(x[2] >= 0 for x in old), old_abstention=sum(x[2] < 0 for x in old),
                    hard_positive=sum(x[2] >= 0 for x in hard), hard_abstention=sum(x[2] < 0 for x in hard))
    s.check_lock(lock)
    demo.save(path, dict(**manifest, old=old, hard=hard))
    s.json_write(HERE/'data_manifest.json', dict(**manifest, artifact_sha256=p.sha(path)))
    print('DATA_COMPLETE', manifest['old_positive'], manifest['hard_positive'], flush=True)


if __name__ == '__main__': main()
