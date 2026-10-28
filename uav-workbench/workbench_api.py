"""Task/group capabilities and exact checkpoint contracts; no silent fallback."""
from dataclasses import asdict
from pathlib import Path
import json,math,zipfile
import numpy as np
from sur_rescue.environment import RescueConfig,ACTIONS
from sur_rescue.hydraulics import metadata,file_digest
from workbench_tasks import ExplorationConfig
from rescue_console import model_record as pilot_record
ROOT=Path(__file__).resolve().parent
TASKS={'exploration':'自主探索','exploration_rescue':'探索—搜救'}
LABELS={'B0':'QMIX','B1':'GMRF + QMIX','B2':'Sur-QMIX','B3':'GMRF + Sur-QMIX','D':'GMRF + Sur-QMIX + Data','P':'GMRF + Sur-QMIX + Physics'}
PILOTS={'B1':'rescue:QMIX8','B3':'rescue:Sur-QMIX8'}

def group_record(group):
    if group not in PILOTS:raise ValueError('该实验组尚无八动作检查点，不能运行模型；可导出配置或切换规则核查')
    return pilot_record(PILOTS[group])

def catalog():
    plan=json.loads((ROOT/'research/dual_task_plan_20261026.json').read_text(encoding='utf-8'));groups=[]
    if {r['id'] for r in plan['method_specs']}!=set(LABELS):raise ValueError('六组实验登记与工作台不一致')
    for spec in plan['method_specs']:
        id_=spec['id'];ready=False;checkpoint=None
        if id_ in PILOTS:
            try:
                _,m=group_record(id_);ready=True;checkpoint=dict(seed=m['seed'],episodes=m['trained_episodes'],actor_sha256=m['actor_sha256'],config=m['config'])
                reason='探索—搜救已有12回合工程检查点，尚未验证收敛'
            except (OSError,ValueError,KeyError) as e:reason='检查点暂不可用：'+str(e)
        elif id_ in ('D','P'):reason='八向辅助训练分支尚未接入，暂无匹配检查点'
        else:reason='尚未训练同源原始观测编码的八动作模型'
        groups.append(dict(**spec,label=LABELS[id_],checkpoint=checkpoint,
            tasks=dict(exploration=dict(ready=False,reason='自主探索尚无专用检查点，可先运行规则核查'),exploration_rescue=dict(ready=ready,reason=reason))))
    return dict(version='hyra-eight-workbench-v3',tasks=[dict(id=k,label=v) for k,v in TASKS.items()],groups=groups,
        actions=list(ACTIONS),default_task='exploration',default_group='B3',default_mode='rule',
        defaults=dict(exploration=asdict(ExplorationConfig()),exploration_rescue=asdict(RescueConfig())),
        scope='八方向工程工作台。规则核查不计入六组模型结果；现有模型仅完成12回合贯通训练。')

def registered_scenes():
    from hyra_uav.experiments import available_scenes
    result=available_scenes()
    for s in result:
        if s['id']=='existing-demo':s['name']='洪水演示过程'
        elif s['id'].startswith('import-'):s['name']='已导入过程 '+s['id'][7:]
    index=ROOT/'generated/sciml-round1/index.json'
    if index.is_file():
        labels={'fit':'开发','selection':'选型','holdout':'历史留出'}
        for row in json.loads(index.read_text(encoding='utf-8')):
            if Path(row['path']).is_file():result.append(dict(id='study:'+row['id'],name=labels[row['split']]+' / '+row['id'],path=row['path'],kind='historical_interactive_only'))
    terrain=ROOT.parent/'terrain-workbench/terrain-lab/generated';seen={str(Path(s['path']).resolve()) for s in result}
    for p in sorted(terrain.glob('maps-*/maps/map-result.zip'),key=lambda p:p.stat().st_mtime,reverse=True):
        if str(p.resolve()) in seen:continue
        try:metadata(p)
        except (OSError,ValueError,KeyError,zipfile.BadZipFile):continue
        id_=p.parent.parent.name;result.append(dict(id='flood:'+id_,name='洪水工作台 / '+id_,path=str(p),kind='direct_hydraulic_workbench_export'))
    return result

def scene_metadata(scene):
    p=metadata(scene['path']);return dict(**{k:v for k,v in scene.items() if k!='path'},grid=p['grid'],duration_s=p['frames'][-1]['time_s'],actual_frame_count=len(p['frames']))

def validate(data):
    if not isinstance(data,dict):raise ValueError('任务参数必须是对象')
    if set(data)-{'scene','task','group','mode','config'}:raise ValueError('包含未知工作台参数')
    task=data.get('task');group=data.get('group');mode=data.get('mode')
    if task not in TASKS or group not in LABELS or mode not in ('rule','model'):raise ValueError('请选择有效任务、六组之一及运行方式')
    scene=next((s for s in registered_scenes() if s['id']==data.get('scene')),None)
    if scene is None:raise ValueError('洪水过程不存在，请刷新列表')
    values=data.get('config',{})
    if not isinstance(values,dict):raise ValueError('任务配置必须是对象')
    config=ExplorationConfig(**values) if task=='exploration' else RescueConfig(**values)
    if not all(math.isfinite(v) for v in asdict(config).values()):raise ValueError('任务参数必须为有限数')
    p=metadata(scene['path']);times=np.asarray([f['time_s'] for f in p['frames']],dtype=float)
    class Header:
        def index(self,t):
            found=np.flatnonzero(np.isclose(times,t,atol=1e-7,rtol=0))
            if len(found)!=1:raise ValueError(f'{t}s没有实际洪水帧，请缩短任务或在洪水工作台生成足够长的过程')
    config.validate(Header())
    if math.prod(p['grid']['shape'])*(round(config.duration_s/config.step_s)+1)>2_000_000:raise ValueError('回放超过200万格点时刻，请缩短任务；原始地图不会缩小')
    if mode=='model':
        if task!='exploration_rescue':raise ValueError('自主探索尚无专用八动作检查点，请切换规则核查；不会使用搜救模型替代')
        _,m=group_record(group)
        if p['grid']['shape']!=[32,32] or p['grid']['dx_m']!=5 or asdict(config)!=m['config']:raise ValueError('该工程检查点需要32×32、5米格距及登记的250秒配置，请应用检查点参数或使用规则核查')
    return dict(scene=scene,task=task,requested_group=group,group=group if mode=='model' else None,mode=mode,config=asdict(config),source_sha256=file_digest(scene['path']))

def history():
    root=ROOT/'generated/web'
    if not root.is_dir():return []
    rows=[]
    for folder in sorted(root.iterdir(),key=lambda p:p.stat().st_mtime,reverse=True)[:100]:
        manifest=folder/'manifest.json'
        if not manifest.is_file():continue
        try:r=json.loads(manifest.read_text(encoding='utf-8'))
        except (OSError,ValueError):continue
        if r.get('schema')!='hyra-eight-workbench-v1':continue
        rows.append(dict(run_id=folder.name,task=r['task'],task_label=TASKS[r['task']],mode=r['execution']['mode'],group=r.get('group'),requested_group=r.get('requested_group'),created_at=r.get('created_at'),summary=r['summary'],replay_url=f'/runs/{folder.name}/replay.json'))
        if len(rows)>=12:break
    return rows
