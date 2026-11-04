import hashlib,json,shutil
from pathlib import Path
from datetime import datetime

base=Path(__file__).resolve().parent
run=base.parent/'uav-strict-eval-20261101/run01'
out=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
repo=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prefix=(base/'serial-results-prefix.jsonl').read_bytes()
raw=(run/'results.jsonl').read_bytes()
assert raw.startswith(prefix)
raw=raw[:raw.rfind(b'\n')+1]
rows=[json.loads(x) for x in raw.splitlines()]
keys=[tuple(r[k] for k in ('candidate','case','assignment','variant','mode','action_seed')) for r in rows]
assert len(keys)==len(set(keys))
status=read(run/'status.json');reg=read(base/'registration.json')
assert status['status']=='evaluating_parallel'
audit=dict(timestamp=datetime.now().isoformat(),status='handoff_verified_evaluation_in_progress',
  completed_snapshot=len(rows),old_prefix_records=reg['completed_prefix_records'],prefix_byte_identical=True,
  unique_keys=True,coordinator_pid=status['pid'],workers=2,full_test_complete=False)
(base/'handoff-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
shutil.copy2(base/'registration.json',run/'parallel-resume-registration.json')
report=f'''2026-11-01 两路并行测试交接报告
用户请求加快速度；仅更换测试调度。原strict_eval.py及所有冻结来源、模型、场景保持原哈希。
原串行程序40556按现有停止标记在完整记录后正常退出。manual_stopped及failure.json是预期交接诊断，已原样保存serial-stop；不是数值失败或重新训练。
已保留{reg['completed_prefix_records']}条完整原记录，字节前缀一致；新快照{len(rows)}/5630条，唯一任务键无重复。当前协调进程{status['pid']}、两个工作进程，每个2个Torch线程。
8项训练场预检，指标和每个轨迹数组逐项相同；串行10.5178秒，并行8.5774秒（含进程启动），约1.226倍。完整混合测试速度尚待持续记录，不能承诺同样提速。
物理内存核查：并行启动后可用约6.64GiB，两个worker各约270MiB，协调器约32MiB；后续模型切换可变化。
原8小时总预算从不晚于原测试开始的21:16:46计时，未重置；8GiB磁盘下限保留。最多4条在途、两条同时执行。
接续时全部原来源/模型/70个场哈希核验通过。每条原权重与轨迹对账继续，最终5630任务完整性和按原任务序聚合尚待完成。
未改算法、训练轮次、模型选择、评分、地图、种子或2048门槛，未重新安装环境、启用GPU或推送仓库。加速不是模型质量提升。
回退与原因、替代方案取舍见protocol.txt；禁止直接重启原串行main覆盖本批结果。下一步等待全测试完成后提供全部方案/规则对照；当前数据仍为部分结果。
'''
(base/'report.txt').write_text(report,encoding='utf8')
files=[p for p in base.rglob('*') if p.is_file() and not any(x in p.relative_to(base).parts for x in ('workers','__pycache__')) and p.name not in ('stdout.log','stderr.log','manifest.json')]
manifest={str(p.relative_to(base)):sha(p) for p in files}
(base/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
files.append(base/'manifest.json')
for dest in (out/'uav-parallel-test-20261101',repo/'uav-workbench/docs/parallel-test-20261101'):
    for p in files:
        target=dest/p.relative_to(base);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);assert sha(p)==sha(target)
entry=f'\n\n## 2026-11-01 严格测试两路并行接续\n\n用户要求提速。8项TRAIN-only等价预检指标/全轨迹一致，10.5178秒降至8.5774秒（含启动）；仅调度提速不代表质量改善。原40556在记录边界正常退出，{reg["completed_prefix_records"]}条前缀字节一致保留，新协调器{status["pid"]}、2worker×2线程接续剩余任务。当前核对{len(rows)}/5630条无重复，21候选及70场、种子、评分、总8小时预算不变，原manual_stopped是交接记录。全部资料与原预检轨迹/配置/哈希保存outputs/uav-parallel-test-20261101和uav-workbench/docs/parallel-test-20261101。最终完整性审计与全表仍待5630条完成；勿启动重复队列或把部分测试用于重新挑选模型。\n'
for folder in (out,repo/'terrain-workbench/docs'):
    for name in ('HYRA自动续作进度.md','HYRA累计更改日志.md'):
        with (folder/name).open('a',encoding='utf8') as f:f.write(entry)
print(json.dumps(audit,ensure_ascii=False))
