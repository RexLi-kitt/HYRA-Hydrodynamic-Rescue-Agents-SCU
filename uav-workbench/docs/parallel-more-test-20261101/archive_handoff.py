import hashlib,json,shutil
from pathlib import Path
from datetime import datetime
base=Path(__file__).resolve().parent;run=base.parent/'uav-strict-eval-20261101/run01'
out=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
repo=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
raw=(run/'results.jsonl').read_bytes();prefix=(base/'serial-results-prefix.jsonl').read_bytes();assert raw.startswith(prefix)
raw=raw[:raw.rfind(b'\n')+1];rows=[json.loads(x) for x in raw.splitlines()]
keys=[tuple(r[k] for k in ('candidate','case','assignment','variant','mode','action_seed')) for r in rows];assert len(keys)==len(set(keys))
status=read(run/'status.json');reg=read(base/'registration.json');pre=read(base/'preflight.json');selection=read(base/'selection.json')
assert status['workers']==selection['workers']==pre['selected_workers']==6 and status['status']=='evaluating_parallel'
assert sha(base/'parallel_eval.py')==pre['wrapper_sha256']==reg['wrapper_sha256']
audit=dict(timestamp=datetime.now().isoformat(),status='handoff_verified_evaluation_in_progress',records=len(rows),total=5630,
  preserved_prefix_records=reg['completed_prefix_records'],prefix_byte_identical=True,unique_keys=True,coordinator_pid=status['pid'],workers=6,
  selection_sha256=sha(base/'selection.json'),selection_matches_preflight=True,wrapper_matches_preflight=True)
(base/'handoff-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
report=f'''2026-11-01 更高并行度对照及六路接续
用户要求：也可以试试更多路。基线uav-parallel4-test-20261101/parallel_eval.py四路；新文件parallel-more-test的parallel_eval.py。
更改：run的worker由固定4改为预登记性能试验所选6，每进程仍2个Torch线程；最多在途8改12。新增available_memory及2GiB余量门槛，触发时保存失败/已完成结果，不自动重试。initialize/evaluate_task/原始trial数值行为不变，生产训练默认未改。
依据：CPU进程仍有余量；内存支持有界增并发。只改变独立试验调度，不改变水情或救援算法。此前直接RAM加载/几何缓存收益不足，GPU环境未安装或迁移。无模型质量提升结论。
预登记：相同24个已有TRAIN场任务，比较4/6/8路（按此顺序，含进程启动）。要求全部指标与全轨迹逐数组一致、速度比>=1.15、整个试验可用内存>=2GiB；合格中最快者优先，但8路比6路加速不足5%时优先6路。正式测试继续运行，负载/时序非完全相同，样本小，不能承诺完整队列提速相同。
实测：4路18.4162秒，6路15.5232秒，8路14.7915秒。六路相对四路1.1864倍；八路相对六路1.0495倍（不足1.05），按预定规则选六路。两候选各24项比较，指标与所有轨迹数组完全一致，72条原轨迹全部保存。
最低可用内存：4路{pre['minimum_free_bytes']['4']/1024**3:.3f}GiB，6路{pre['minimum_free_bytes']['6']/1024**3:.3f}GiB，8路{pre['minimum_free_bytes']['8']/1024**3:.3f}GiB。六路正式启动后快照约5.07GiB可用，各worker约350MiB；不是峰值保证。
交接：原32000按停止标记正常退出。2595已合并记录及8条已完成在途共2603条全部接回；前缀逐字节保留，无重跑这些记录。新协调器{status['pid']}；核对快照{len(rows)}/5630无重复，全部旧结果、失败及配置保留。
不变量：21冻结候选、70个父场、5630任务、种子、角色、评分、完整/no-cover、规则基线和抽样重复不变，原来源/权重/场哈希启动复核通过。每条权重/轨迹检查仍有效，最终按原任务序聚合。8小时预算从不晚于原测试开始的21:16:46累计，不重置；磁盘8GiB下限不变。
回归与跳过：48项候选对照通过；未用测试成绩选择并发或模型，未新跑训练、重生成测试场或改2048准入。全5630最终审计尚待完成，无最终排名结论。
回退：正常停止六路后保存所有worker完成记录并按原任务键对账；复制已验证四路调度建立新接续版本，重新登记已用时间。不得直接重启原串行main覆盖结果，不自动循环重试资源限制。
下一步：完成同一冻结测试，再统一对照21候选及规则。证据双份保存到outputs/uav-parallel-more-test-20261101和uav-workbench/docs/parallel-more-test-20261101；不推送Git。
'''
(base/'report.txt').write_text(report,encoding='utf8')
files=[p for p in base.rglob('*') if p.is_file() and not any(x in p.relative_to(base).parts for x in ('workers','__pycache__')) and p.name not in ('stdout.log','stderr.log','manifest.json')]
(base/'manifest.json').write_text(json.dumps({str(p.relative_to(base)):sha(p) for p in files},ensure_ascii=False,indent=2),encoding='utf8');files.append(base/'manifest.json')
for targetroot in (out/'uav-parallel-more-test-20261101',repo/'uav-workbench/docs/parallel-more-test-20261101'):
    for p in files:
        target=targetroot/p.relative_to(base);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);assert sha(p)==sha(target)
entry=f'\n\n## 2026-11-01 更多并行度对照，按预登记选择六路\n\n人工要求尝试更多路。4/6/8路24个TRAIN-only任务分别18.4162/15.5232/14.7915秒，候选各24项指标及轨迹完全一致。六路相对四路1.1864倍，八路较六路收益不足预定5%，选六路；最低可用内存均高于2GiB。原32000正常退出，2595合并及8条在途共2603条全部保留；新协调器{status["pid"]}六路×2线程，快照{len(rows)}/5630无重复。冻结模型/地图/任务/评分/8小时预算不变，新增2GiB可用RAM下限，不改GPU或训练。证据/72轨迹/配置哈希在outputs/uav-parallel-more-test-20261101及uav-workbench/docs/parallel-more-test-20261101；最终测试未完成，不以部分结果改排名，勿重复启动。\n'
for folder in (out,repo/'terrain-workbench/docs'):
    for name in ('HYRA自动续作进度.md','HYRA累计更改日志.md'):
        with (folder/name).open('a',encoding='utf8') as f:f.write(entry)
print(json.dumps(audit,ensure_ascii=False))
