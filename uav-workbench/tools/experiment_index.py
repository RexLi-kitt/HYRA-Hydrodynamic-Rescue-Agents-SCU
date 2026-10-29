"""Index actual local experiments without combining superseded runs as evidence."""
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hyra_uav.experiments import write_json

CANONICAL = {
    'p2-pilot-20261025-final': ('P2 development pilot', 'index.json', '', 54),
    'p2-formal-20261025': ('P2 formal scale selection', 'index.json', 'runs', 81),
    'p3-20261025': ('P3 historical linear screening', 'replay-index.json', 'replays', 18),
    'p3-nonlinear-round1': ('P3 nonlinear round 1', 'index.json', 'replays', 54),
    'p3-nonlinear-round2': ('P3 nonlinear round 2', 'index.json', 'replays', 54),
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def identity(path):
    return dict(path=str(path.relative_to(ROOT)).replace('\\', '/'), bytes=path.stat().st_size,
                sha256=sha256(path.read_bytes()).hexdigest())


def main():
    output = ROOT/'docs/evidence/20261025-experiment-index'
    output.mkdir(parents=True, exist_ok=True)
    phases, runs = [], []
    for folder, (meaning, filename, replay_dir, expected) in CANONICAL.items():
        root = ROOT/'generated'/folder
        index = read(root/filename)
        assert len(index) == expected, folder
        phase = dict(folder=folder, meaning=meaning, indexed_runs=len(index),
                     index=identity(root/filename), role='canonical for this phase, not independent events',
                     score_complete=(root/'summary.json').exists() if folder.startswith('p3-nonlinear') else True)
        for row in index:
            if row.get('replay'):
                path = root/row['replay']
            elif row.get('path'):
                path = Path(row['path'])
                if not path.is_absolute():
                    path = ROOT/path
            elif row.get('run_id'):
                path = root/replay_dir/row['run_id']/'replay.json'
            else:
                raise ValueError('Index row has no replay location: '+repr(row))
            assert path.is_file(), (folder, row)
            replay = read(path)
            final_time = replay['snapshots'][-1]['time_s']
            assert final_time == replay['config']['duration_s']
            runs.append(dict(phase=folder, run_id=path.parent.name, replay=identity(path),
                             source_sha256=replay['source_sha256'], scene=replay['scene']['id'],
                             family=replay['scene'].get('family'), method=replay['method'],
                             duration_s=final_time, actions=len(replay['actions']), samples=len(replay['snapshots']),
                             split=row.get('split'), cache_sha256=row.get('cache_sha256')))
        phases.append(phase)
    assert len(runs) == 261 and len({(r['phase'], r['run_id']) for r in runs}) == 261
    others = []
    for root in sorted((ROOT/'generated').iterdir()):
        if not root.is_dir() or root.name in CANONICAL or root.name=='sciml-round1':
            continue
        role = ('superseded pilot, excluded from canonical totals' if root.name in ('p2-pilot-20261025', 'p2-pilot-20261025-v2')
                else 'fit-only relation preflight, not another independent mechanism round' if 'preflight' in root.name
                else 'engineering demonstration or test artifacts; not independent research events')
        metadata = [identity(p) for p in sorted(root.glob('*.json'))]
        replay_files = [identity(p) for p in sorted(root.glob('*/replay.json'))]
        if (root/'replay.json').exists():
            replay_files.append(identity(root/'replay.json'))
        others.append(dict(folder=root.name, role=role, metadata=metadata, replay_files=replay_files))
    learning = []
    preflight = ROOT/'generated/sur-preflight-20261025'
    if (preflight/'summary.json').exists():
        summary = read(preflight/'summary.json')
        if summary['status'] == 'completed':
            for method in summary['methods']:
                folder = preflight/method['method']
                learning.append(dict(method=method['method'], training_episodes=method['episodes'],
                                     training_joint_steps=method['training_joint_steps'],
                                     greedy_evaluation_joint_steps=method['evaluation_joint_steps'],
                                     summary=identity(preflight/'summary.json'),
                                     checkpoint_pointer=identity(folder/'latest.json'),
                                     greedy_replays=[identity(p) for p in sorted(folder.glob('evaluation-*.json'))],
                                     role='one-seed engineering preflight, not a new P3 round or confirmatory physics result'))
    sciml = None
    study = ROOT/'generated/sciml-round1'
    if (study/'formal-plan.json').exists():
        plan=read(study/'formal-plan.json')
        cases=read(study/'index.json')
        jobs=[]
        for method in plan['methods']:
            for seed in plan['seeds']:
                folder=study/'formal'/(method.replace('+','-')+f'-s{seed}')
                summary=folder/'summary.json';progress=folder/'progress.json'
                jobs.append(dict(method=method,seed=seed,state='completed' if summary.exists() else 'started' if progress.exists() else 'not_started',
                    summary=identity(summary) if summary.exists() else None,
                    progress=read(progress) if progress.exists() else None,
                    history=identity(folder/'history.json') if (folder/'history.json').exists() else None,
                    initial_weights=identity(folder/'initial-weights.json') if (folder/'initial-weights.json').exists() else None))
        sciml=dict(role='One independent four-group physical auxiliary study, not a third P3 round; progress snapshot only',
            formal_plan=identity(study/'formal-plan.json'),evaluation_plan=identity(study/'evaluation-plan.json'),
            pilot_summary=identity(study/'pilot-summary.json'),hydraulic_index=identity(study/'index.json'),
            cases_by_split={split:sum(e['split']==split for e in cases) for split in ('fit','selection','holdout')},
            planned_training_episodes=24000,planned_joint_training_steps=1440000,completed_jobs=sum(j['state']=='completed' for j in jobs),jobs=jobs,
            actual_policy_replays=[identity(p) for p in sorted((study/'evaluation/policies').glob('*.json'))],
            actual_rule_replays=[identity(p) for p in sorted((study/'evaluation/rules').glob('*.json'))],
            forecast_shards=[identity(p) for p in sorted((study/'evaluation/forecasts').glob('*.jsonl'))],
            confirmatory_summary=identity(study/'confirmation-summary.json') if (study/'confirmation-summary.json').exists() else None)
    result = dict(status='indexed', phases=phases, canonical_replays=261, runs=runs, learning_preflights=learning,independent_sciml_study=sciml,
                  other_artifact_folders=others,
                  limits='Counts are completed route/budget replays, not independent hydraulic events. Score completeness is recorded per phase; this index does not by itself certify scientific success.')
    write_json(output/'index.json', result)
    lines = ['# 本地实验与工程运行总索引', '',
             '规范阶段共 261 条完成回放。这个数字包含不同策略、起点、时间预算和重用过程，不能称作 261 个独立事件。'
             '模型与缓存哈希见各阶段输入索引，完整路径、输入摘要、动作数和时刻见机器可读总表。', '',
             '| 目录 | 用途 | 完成回放数 | 留出评分完整 |', '|---|---|---:|---|']
    lines += [f"| {p['folder']} | {p['meaning']} | {p['indexed_runs']} | {'是' if p['score_complete'] else '尚未完成'} |" for p in phases]
    if learning:
        lines += ['', '## 独立学习贯通实验', '',
                  '原P3收尾后另完成QMIX与Sur-QMIX适配版各12训练回合/720联合步，合计24回合；'
                  '各3条贪心评估、共360联合评估步。检查点回放和6条评估索引单列，不并入上面的261条原阶段回放。'
                  '只有一个种子和已见fixture，不是正式物理辅助结果。详见[贯通报告](SUR_PREFLIGHT_20261025.md)。']
    if sciml:
        lines += ['', '## 独立物理辅助世界模型首轮', '',
            f"四组×三个种子×2000回合，计划24000回合/1440000联合训练步；当前完成{sciml['completed_jobs']}/12个任务。"
            f"实际水情索引：拟合{sciml['cases_by_split']['fit']}、选型{sciml['cases_by_split']['selection']}、确认留出{sciml['cases_by_split']['holdout']}。",
            '本段为生成索引时的快照，进程存活须另外检查。资源试跑与CLI拟合集验证单列，不并入确认性回放或独立事件数量。',
            '正式训练分片、每次恢复与十次选模检查点保留在generated/sciml-round1/formal；逐分片哈希见训练审计。',
            '[协议与证据入口](SCIML_PROTOCOL_20261025.md)。最终结论需要全部留出评价和独立审计，不由完成计数推断。']
    lines += ['', '## 历史、预检与工程产物', '']
    lines += [f"- `{r['folder']}`：{r['role']}。" for r in others]
    lines += ['', '- [逐条运行、摘要与全部阶段路径](evidence/20261025-experiment-index/index.json)',
              '- [验收表](CONTINUOUS_ACCEPTANCE.md)', '- [两轮 P3 结论](P3_FINAL_20261025.md)',
              '- [复现命令](REPRODUCE.md)', '',
              '完整过程、回放和模型保留在本机 generated 中；该目录不随代码入库。'
              '新克隆按复现命令重建，不能把本地缓存路径当作已经发布的数据包。', '']
    (ROOT/'docs/EXPERIMENT_INDEX_20261025.md').write_text('\n'.join(lines), encoding='utf-8')
    print('indexed', len(runs), 'canonical replays;', len(others), 'other artifact folders')


if __name__ == '__main__':
    main()
