import json,hashlib,shutil
from pathlib import Path
import numpy as np
from preselect import summarize,sha,CANDIDATES
BASE=Path(__file__).parent;RUN=BASE/'run01'
OUTPUTS=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
REPO=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031')
OUT=OUTPUTS/'uav-plan-preselection-20261101';DOC=REPO/'uav-workbench/docs/plan-preselection-20261101'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
state=read(RUN/'status.json');assert state['status']=='completed'
protocol=read(RUN/'protocol.json');results=read(RUN/'results.json');selection=read(RUN/'selection.json')
assert sha(BASE/'HYRA无人机算法预选与训练计划书-v1.0.docx')==protocol['plan_sha256']
assert sum(map(len,results.values()))==208
all_checks=[];remaining=[]
for cid,rows in results.items():
 assert len(rows)==(16 if cid=='P0' else 48)
 for r in rows:
  folder=RUN/cid
  trace=[json.loads(line) for line in (folder/r['trace']).read_text(encoding='utf8').splitlines()]
  assert len(trace)==r['ST']
  assert sum(t['info']['conflicts'] for t in trace)==r['conflicts']
  assert sum(t['info']['rescued'] for t in trace)==r['rescued']
  target=read(folder/r['targets']);assert len(target)==20
  assert sum(t['status']==2 for t in target)==r['rescued']
  assert sum(r['failure_counts'].values())==20-r['rescued']
  final=np.load(folder/(r['trace'][:-6]+'-final.npz'),allow_pickle=False)
  assert abs(float(final['known'].mean())-r['coverage'])<1e-12
  assert len(final['missing_cells'])==10000-int(final['known'].sum())
  for t in target:
   if t['status']!=2:remaining.append(dict(candidate=cid,case=r['case'],layout=r['layout'],mode=r['mode'],action_seed=r['action_seed'],**t))
 all_checks.append(dict(candidate=cid,episodes=len(rows),traces_and_final_states_verified=True))
audit=dict(status='passed',total_rollouts=208,new_training_episodes=0,plan_unchanged=True,checks=all_checks,note='Verification of archived records, not performance acceptance')
write(RUN/'record-audit.json',audit);write(RUN/'all-unserved-target-records.json',remaining)

report=['# 无人机算法预选执行报告','',
 '详细计划书v1.0在运行前冻结；五候选208次回放已全部完成，新增训练0回合。使用2个既有合成父水场，旧4布局与新增4布局分开统计。不是50场最终测试，也不是现场准确率。','',
 f"实际回放耗时{state['elapsed_s']/60:.2f}分钟。计划文档SHA256：{protocol['plan_sha256']}。",'']
chosen=selection['selected_learning_candidate']
report += ['## 预选结论','',f"满足全部预登记条件的学习候选：{chosen or '无'}。"]
if chosen:
 report+=['下一阶段首先扩展独立开发父水场，冻结新高层接口后再做64回合检查；本轮没有把旧逐步策略当成已训练高层。']
else:
 report+=['触发计划中的停止扩训条件：当前组合不进入1024回合长训练。保留全部候选和失败，下一项工作应针对已复现的搜索遗漏或冲突协调另登记单因素试验。分层接口、扩展水场与多种子训练均尚未执行。']
report+=['纯规划仍是必要对照；其高完成比例不代表所有开发门槛已通过。网络与规则的贡献按接管比例区分，不将规则带来的完成度写成训练收益。','']
tables={}
for stratum,title in [('old','旧布局'),('new','新增布局')]:
 report += [f'## {title}完整对照','', '|候选|执行|完整任务|获救比例|覆盖率|截断步数|潜在冲突每步|规划接管机步|','|---|---|---:|---:|---:|---:|---:|---:|']
 for cid,rows in results.items():
  for mode in ['sample','fixed']:
   rr=[r for r in rows if r['stratum']==stratum and r['mode']==mode]
   if not rr:continue
   s=summarize(rr);tables[f'{cid}/{stratum}/{mode}']=s
   report.append(f"|{cid} {CANDIDATES[cid]['name']}|{'随机' if mode=='sample' else '固定'}|{s['complete_count']}/{s['n']}|{s['RCR']:.2%}|{s['coverage']:.4%}|{s['capped_steps']:.2f}|{s['ACC']:.5f}|{s['planner_agent_fraction']:.2%}|")
 report += ['','每组对应2父水场×4布局；随机方式增加2种动作种子，仍只有8种父水场与布局组合。失败任务截断步数记300，成功任务用实际联合步数；每步5秒。潜在冲突为环境阻止前的轨迹意图冲突，不是实际碰撞次数。','']
report+=['## 各项门槛及未入选原因','']
for cid,d in selection['decisions'].items():
 failures=[]
 for gate in d['gates']:
  bad=[{'completion_nonlower':'完成比例低于纯规划','rescue_nonlower':'获救比例低于纯规划','conflict_ok':'冲突超过0.02'}[k] for k in ('completion_nonlower','rescue_nonlower','conflict_ok') if not gate[k]]
  if cid!='P0' and gate['mode']=='sample' and not gate['material_gain']:bad.append('未达到预定完成量或时间收益')
  if bad:failures.append(f"{gate['stratum']}/{gate['mode']}：{'、'.join(bad)}")
 report.append(f"- {cid}：工程门槛{'通过' if d['engineering_gates_pass'] else '未通过'}，继续学习投入门槛{'通过' if d['learning_investment_pass'] else '未通过或不适用'}。"+'；'.join(failures)+'。')
report+=['','## 失败诊断','', '|候选与执行|未发现记录|已发现有后续窗口未服务|已发现无后续窗口未服务|冲突拦停机步|无新增覆盖步|','|---|---:|---:|---:|---:|---:|']
for cid,rows in results.items():
 for mode in ['sample','fixed']:
  rr=[r for r in rows if r['mode']==mode]
  if not rr:continue
  s=summarize(rr)
  report.append(f"|{cid} {mode}|{s['undiscovered']}|{s['discovered_with_window_unserved']}|{s['discovered_without_window_unserved']}|{s['blocked_agent_steps']}|{s['zero_gain_steps']}|")
report+=['','这是回放累计目标记录，可能涉及同一布局重复运行，不能当真实独立人数。合法服务窗口由回合结束后的真值核对，未输入控制器；有窗口不证明可及时到达，无发现后窗口不证明整场不可行。拦停和漏扫可以同时发生，不能将诊断计数相加成独立根因。',
 '全部未完成目标见 all-unserved-target-records.json；每次20人完整状态、漏格坐标及最终位置在对应候选目录的targets.json与final.npz；逐步原动作、最终动作、服务、规划、位移、冲突及中断见jsonl。','',
 '## 更改与检查','',
 '- 新增preselect.py，仅组合既有冻结控制器与权重。纯规划从第一步启用；其余补漏沿用连续5步无新增覆盖触发。重试记忆记录实际最终动作。原环境、奖励、服务阈值、300步及旧权重未修改。',
 '- 四项启动检查通过：P1原动作流等价、P0/P2合法动作与观测覆盖检查、相同观测下改变未来真值不改变第一步决策。完整运行持续检查覆盖不越过实际观测、服务动作和合法掩码。',
 '- P1旧布局随机回放与历史记录逐项一致；208次逐步冲突/服务累计与最终指标、20人状态及漏格数量一致。父模型及原源码哈希保持。',
 '- 计划书共8页，已使用本机Word渲染并逐页检查。打包渲染器因缺LibreOffice失败，使用Word完成同一文档的视觉核验；未安装软件或修改安全策略。',
 '- 本次跳过：新高层实现、权重更新、扩大父水场、3种子训练、50场最终测试、建筑通信续航和真实救援验证。跳过依据为分阶段预选协议；不能声称这些阶段完成。',
 '- 回退：继续使用原项目入口和已保存父模型；本轮代码、文档、输出均隔离。未推送PR或外发消息。','',
 '## 后续单元','',
 '先定位冲突最集中的旧/新布局和发生阶段，区分搜索时互相阻塞、接近服务目标冲突及服务占位；冻结一个只依赖公开意图的协调候选，保留原始意图冲突和协调后动作两套记录。随后复核新增布局上的搜索遗漏，再决定高层任务分配接口。不能通过屏蔽冲突计数、删除不可服务目标或缩小覆盖区域达标。']
write(RUN/'mode-stratum-summary.json',tables)
(BASE/'execution-report.md').write_text('\n'.join(report)+'\n',encoding='utf8')

for target in (OUT,DOC):
 target.mkdir(parents=True,exist_ok=True)
 for name in ['HYRA无人机算法预选与训练计划书-v1.0.docx','plan-content.json','build_plan.py','preselect.py','finalize.py','execution-report.md']:
  shutil.copy2(BASE/name,target/name)
 for path in RUN.rglob('*'):
  if path.is_file():
   dest=target/path.relative_to(RUN);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
 write(target/'live-run.json',dict(status='completed',live_path=str(RUN),added_training_episodes=0))
index=[dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact-index.json']
for target in (OUT,DOC):write(target/'artifact-index.json',index)
note=f"## 2026-11-01 无人机详细计划和首轮预选完成\n\n8页计划书已冻结并完成208次五候选回放，新增训练0。旧4布局和新4布局、随机及固定执行分列，仍仅2个合成父水场，封存50未用。入选学习候选：{chosen or '无，按预定规则停止当前组合扩训'}。四项启动检查、历史P1复现及208份轨迹末态审计通过。全部结果与计划在outputs/uav-plan-preselection-20261101及uav-workbench/docs/plan-preselection-20261101双份保存。分层接口、多种子与新水场训练未执行，原模型及失败保留。未推送PR，勿重复本配置。\n\n"
for root in (OUTPUTS,REPO/'terrain-workbench/docs'):
 for name in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
  path=root/name
  if name=='HYRA自动续作进度.md':path.write_text(note+path.read_text(encoding='utf8'),encoding='utf8')
  else:
   with path.open('a',encoding='utf8') as f:f.write('\n'+note)
print(json.dumps(dict(archived=str(OUT),rollouts=208,selected=chosen,records=len(index)),ensure_ascii=True))
