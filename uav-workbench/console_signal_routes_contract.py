"""Lightweight UI contract for the completed, frozen signal-route study."""
from pathlib import Path
from console_scale100_contract import ROOT, ACTIONS, TRAINING_SEEDS, read, digest, require
from console_scale100_contract import default_config as base_config

STUDY = ROOT / 'generated/signal-routes-20261027-v2'
LABELS = {'R1':'路径一 · 有向SAR', 'R2':'路径二 · 物理协变量均值',
          'R3':'路径三 · 跨层CRF', 'Braw':'原始信号基线', 'Bpost':'共同后验基线',
          'C1':'路径一 · 水情置乱', 'C2':'路径二 · 水情置乱', 'C3':'路径三 · 水情置乱'}
DESCRIPTIONS = {'R1':'非对称结构系数，精度保持对称正定', 'R2':'水深、速度大小、坡度幅值和记录年龄',
    'R3':'结构与信号的联合二次势', 'Braw':'沿用实际采集的信号记录', 'Bpost':'不使用物理调制的共同后验',
    'C1':'有向SAR的匹配置乱对照', 'C2':'均值校正的匹配置乱对照', 'C3':'跨层CRF的匹配置乱对照'}


def release_record():
    complete=read(STUDY/'completion-audit.json'); delivery=read(STUDY/'delivery-audit.json')
    final=read(STUDY/'finalization-status.json'); seal=read(STUDY/'sealed-models.json')
    require(read(STUDY/'status.json')['status']==final['status']=='completed'
        and complete['passed'] and delivery['passed'], '三路径研究尚未完成全部验收')
    require(final['evidence']['audit_sha256']==delivery['independent_completion_audit_sha256']==digest(STUDY/'completion-audit.json'),
            '三路径完整验收记录已变化')
    require(complete['report_sha256']==delivery['report_sha256']==digest(STUDY/'report.md')
        and complete['results_sha256']==digest(STUDY/'confirmation/results.json'), '三路径结果摘要不匹配')
    require(complete['completed_formal_jobs']==24 and complete['confirmation_model_replays']==1728
        and complete['rule_evaluations']==216, '三路径评价矩阵不完整')
    require(seal['parameters_sha256']==digest(STUDY/'parameters.json')
        and seal['dataset_sha256']==digest(STUDY/'data/dataset.json'), '冻结参数或数据清单已变化')
    require(len(seal['models'])==24 and {(r['method'],r['seed']) for r in seal['models']}==
        {(m,s) for m in LABELS for s in TRAINING_SEEDS}, '三路径模型清单不完整')
    require(seal['sources']==complete['sources'] and all(digest(ROOT.parent/name)==sha
        for name,sha in seal['sources'].items()), '三路径冻结训练源码已变化')
    return seal


def model_record(method, training_seed=11):
    require(method in LABELS and type(training_seed) is int and training_seed in TRAINING_SEEDS, '路径或训练种子无效')
    seal=release_record(); row=next(r for r in seal['models'] if (r['method'],r['seed'])==(method,training_seed))
    actor=STUDY/'formal'/f'{method}-s{training_seed}'/f"actor-{row['episode']:04d}.pt"
    meta=read(actor.with_suffix('.json'))
    require(actor.resolve()==Path(row['actor']).resolve()
        and digest(actor)==row['actor_sha256']==meta['actor_sha256'], '所选路径模型权重摘要不匹配')
    require(meta['method']==method and meta['seed']==training_seed and meta['phase']=='formal'
        and meta['trained_episodes']==row['episode'] and meta['sources']==seal['sources']
        and meta['parameters_sha256']==seal['parameters_sha256'], '路径模型与冻结清单不一致')
    return actor,meta,row


def scene_records():
    return [dict(id=f"signal-routes:{r['parent_terrain']}:{r['mechanism']}",
        name=('三路径开发' if r['split']=='development' else '三路径训练')+f" / {r['parent_terrain']} / {r['mechanism']}",
        path=r['path'],source_sha256=r['sha256'],split=r['split'],parent=r['parent_terrain'],mechanism=r['mechanism'],
        grid=dict(shape=[20,20],dx_m=5.,axis_order=['z','x']),duration_s=900.,actual_frame_count=181,
        kind='synthetic_signal_routes_swe') for r in read(STUDY/'data/dataset.json')['cases']
        if r['split'] in ('fit','development')]


def default_config(task='exploration_rescue', group='R1', seed=11):
    require(group in LABELS, '请选择三路径研究中的已登记方法')
    return base_config(task,'B3',seed)


def validate(data):
    require(isinstance(data,dict) and not set(data)-{'scene','task','group','mode','config','training_seed'}, '三路径请求参数无效')
    require(data.get('task')=='exploration_rescue', '三路径模型仅训练了探索—搜救任务，请切换探索—搜救')
    method,mode=data.get('group'),data.get('mode')
    require(method in LABELS and mode in ('model','rule'), '路径或运行方式无效')
    values=data.get('config',{}); require(isinstance(values,dict), '任务配置必须是对象')
    expected=default_config(group=method,seed=values.get('seed',11))
    require(not set(values)-set(expected) and all(type(v) in (int,float,str) and v==expected[k] for k,v in values.items()),
        '三路径固定为100米、两机、900秒及登记的感知/施救配置；仅场景种子可变')
    scene=next((r for r in scene_records() if r['id']==data.get('scene')),None)
    require(scene is not None, '请选择三路径训练或开发场景；确认场景不用于交互运行')
    require(digest(scene['path'])==scene['source_sha256'], '洪水过程摘要已变化')
    seed=data.get('training_seed',11)
    require(type(seed) is int and seed in TRAINING_SEEDS, '训练种子必须是11、29或47')
    if mode=='model': model_record(method,seed)
    return dict(scene=scene,task='exploration_rescue',requested_group=method,group=method if mode=='model' else None,
        mode=mode,config=expected,training_seed=seed,source_sha256=scene['source_sha256'])


def catalog():
    try: release_record(); ready,reason=True,'已完成24项训练、1944项评价和独立验收；交互回放不计入正式结果'
    except (OSError,ValueError,KeyError,TypeError) as error: ready,reason=False,str(error)
    return dict(version='hyra-signal-routes-console-v1',profile='signal_routes',released=ready,reason=reason,
        actions=list(ACTIONS),default_group='R1',default_task='exploration_rescue',default_mode='model',
        training_progress=dict(completed=24 if ready else 0,expected=24),
        defaults={t:default_config(t) for t in ('exploration','exploration_rescue')},
        groups=[dict(id=m,label=label,description=DESCRIPTIONS[m],representation='dual_gmrf',surprise=True,auxiliary='none',
            tags=['Sur-QMIX','实际信号' if m=='Braw' else '信号后验', '冻结估计器'],
            tasks={'exploration':dict(ready=False,reason='本轮三路径仅提供探索—搜救模型'),
                   'exploration_rescue':dict(ready=ready,reason=reason)}) for m,label in LABELS.items()])
