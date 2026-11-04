import hashlib,json,shutil
from pathlib import Path
from datetime import datetime
base=Path(__file__).resolve().parent;run=base.parent/'uav-strict-eval-20261101/run01'
out=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
repo=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
raw=(run/'results.jsonl').read_bytes();prefix=(base/'serial-results-prefix.jsonl').read_bytes()
assert raw.startswith(prefix)
raw=raw[:raw.rfind(b'\n')+1];rows=[json.loads(x) for x in raw.splitlines()]
keys=[tuple(r[k] for k in ('candidate','case','assignment','variant','mode','action_seed')) for r in rows]
assert len(keys)==len(set(keys))
status=read(run/'status.json');reg=read(base/'registration.json');pre=read(base/'preflight.json')
assert status['workers']==4 and status['status']=='evaluating_parallel'
audit=dict(timestamp=datetime.now().isoformat(),status='handoff_verified_evaluation_in_progress',records=len(rows),total=5630,
    preserved_prefix_records=reg['completed_prefix_records'],prefix_byte_identical=True,unique_keys=True,coordinator_pid=status['pid'],workers=4)
(base/'handoff-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
report=f'''2026-11-01 严格测试四路并行接续
人工要求：可以尝试再加快一点。基线：uav-parallel-test-20261101/parallel_eval.py 两路进程；新文件：parallel4-test的parallel_eval.py。
更改：run调度max_workers=2改4，最多在途4改8；每worker仍2个Torch线程；initialize/evaluate_task/原始trial数值行为保持一致。生产训练默认和源模型未修改。
证据：preflight-protocol在测试前登记，仅已有训练场，24任务，两路/四路完整指标与每个轨迹数组完全一致。含启动耗时两路{pre['seconds']['2']:.4f}秒，四路{pre['seconds']['4']:.4f}秒，加速比{pre['speedup']:.4f}，通过预定>=1.15门槛。对照期间正式两路仍运行，时序负载不同，因此小样本加速比例不是完整队列保证。
原因与取舍：独立回放可利用空闲CPU进程，约6.5GiB可用内存支持四路。此前RAM和几何缓存没有足够收益；本轮不安装GPU环境、不改推理线程或算法。启动后可用内存约5.56GiB，四worker约1.31GiB RSS，未见换页压力证据；不声称持续峰值上限。
交接：6636按原停止机制正常退出，已合并2100条及已完成在途4条共2104条全部接回。旧前缀逐字节保留，旧异常信息是预期停止标记而非数值失败。新协调器{status['pid']}，当前核验{len(rows)}/5630条无重复；全量对比原样保留results.jsonl与完整原traces，新增登记同步原输出。
冻结与检查：原24个来源/21候选权重/70个场哈希启动时复核通过。原5630任务、种子、角色、评分、full/no-cover、规则基线、抽样重复不变。每条继续原权重/轨迹检查；最终按原任务顺序聚合。原8小时预算继续从不晚于初始评测开始的21:16:46保守计时，不重置；8GiB磁盘下限保留。
回归与跳过：24项完整等价通过；48条预检轨迹保存。不重跑训练，不生成新测试场，不根据测试重新选择模型。全5630完成和最终审计仍待运行；无模型质量提升结论。
回退：用原STOP_AFTER_CHECKPOINT正常退出，保存已完成在途worker记录，按任务键合并并验证；由已验证两路调度复制为新的接续版本，重新登记累计预算。禁止直接重启原strict_eval主程序覆盖或重复全部任务。
下一步：继续四路完成全部测试，再统一分析21候选与规则控制。资料/配置/哈希/预检轨迹双份保存，未推送仓库。
'''
(base/'report.txt').write_text(report,encoding='utf8')
files=[p for p in base.rglob('*') if p.is_file() and not any(x in p.relative_to(base).parts for x in ('workers','__pycache__')) and p.name not in ('stdout.log','stderr.log','manifest.json')]
(base/'manifest.json').write_text(json.dumps({str(p.relative_to(base)):sha(p) for p in files},ensure_ascii=False,indent=2),encoding='utf8');files.append(base/'manifest.json')
for targetroot in (out/'uav-parallel4-test-20261101',repo/'uav-workbench/docs/parallel4-test-20261101'):
    for p in files:
        target=targetroot/p.relative_to(base);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);assert sha(p)==sha(target)
entry=f'\n\n## 2026-11-01 严格测试四路并行接续\n\n用户要求进一步提速。24个TRAIN-only任务对照逐指标/逐轨迹一致；两路20.0253秒、四路14.6836秒（含启动），通过预定1.15加速门槛，实际1.3638。原6636正常退出，2100合并条及4条已完成在途全部恢复，2104条前缀不变。新协调器{status["pid"]}四路×2线程接续，当前{len(rows)}/5630无重复，原全场景/候选/种子/评分/8小时总预算不变。内存可用约5.56GiB，未改GPU/训练/模型。预检/原轨迹/配置/哈希在outputs/uav-parallel4-test-20261101与uav-workbench/docs/parallel4-test-20261101双份；全测试仍运行，未用部分结果选模型，不宣称质量提升，勿重复启动。\n'
for folder in (out,repo/'terrain-workbench/docs'):
    for name in ('HYRA自动续作进度.md','HYRA累计更改日志.md'):
        with (folder/name).open('a',encoding='utf8') as f:f.write(entry)
print(json.dumps(audit,ensure_ascii=False))
