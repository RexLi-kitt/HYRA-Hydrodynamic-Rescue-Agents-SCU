import hashlib,json,shutil,sys,time
from pathlib import Path
BASE=Path(__file__).parent;RUN=BASE/'run01'
OUTPUTS=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
REPO=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261031')
OUT=OUTPUTS/'uav6-joint-training-20261101';DOC=REPO/'uav-workbench/docs/joint-training-20261101'
def read(p):return json.loads(p.read_text(encoding='utf8'))
state=read(RUN/'queue-status.json');finished=state['status'] in ('completed','failed','stopped')
OUT.mkdir(exist_ok=True);DOC.mkdir(parents=True,exist_ok=True)
report=['# 双模型联合适配训练（2026-11-01）','',
 '用户要求实际组合训练。两套候选为HASAC1024+MAT256、HASAC1024+HATRPO1024，每套计划新增256个联合任务回合。原有独立模型和冻结组合结果保留。','',
 '## 当前状态','',f"队列状态：{state['status']}。真实运行目录：{RUN}",'',
 '## 实现及依据','',
 '- 每架飞机的控制模型身份作为4个公开输入位提供。新增权重列初始为0，初始策略及值函数与父检查点一致性检查通过。',
 '- 两个模型共同采集同一集群的真实轨迹，每两个完整回合更新一次，采集期间保持双方策略固定。MAT/HATRPO仅对自己实际负责的动作使用新轨迹更新，其他动作掩码为单动作且旧对数概率为0。',
 '- HASAC使用真实混合队友生成下一步联合动作及价值目标，熵项只包含HASAC负责的飞机。逐个枚举动作时重新计算MAT后续动作的因果条件，避免把另一策略的动作概率当作自己的。',
 '- 两种对角分工轮换，每4个训练父场景切换一次，两套候选使用相同场景与布局序列。重置旧经验回放和优化器状态，避免沿用旧同质队友数据；已有网络权重保留。',
 '- 这是独立研究适配，混合两种更新机制没有继承论文的单调改善保证，不能称官方复现。',
 '- 参考：HARL https://www.jmlr.org/papers/v25/23-0488.html；HASAC https://proceedings.iclr.cc/paper_files/paper/2024/hash/fe066022bab2a6c6a3c57032a1623c70-Abstract-Conference.html；MAT https://arxiv.org/abs/2205.14953 。','',
 '## 验证与回退','',
 '- 父策略输入扩展一致性、所有权动作掩码、两模型确实更新、非负责HATRPO actor不被更新，以及真实文件保存/恢复后的后续优化逐项一致检查通过。详情preflight.json。',
 '- 每64个联合回合完成固定32次开发验证：2父水场×4布局×2动作种子×2分工。两分工全部保留，不按分工挑选有利结果。训练水场4个；封存50场测试不访问。',
 '- 自动施救、重试控制、奖励和环境物理沿用已有条件。没有覆盖补漏接管，没有真实续航/建筑/天气与吊运验证。',
 '- 最佳版条件：平均获救不降、覆盖最多下降0.01、潜在冲突/步<=0.02、最差分工获救不降，并有预定最小收益。完整跑满256，不因表现退步提前删除或结束。',
 '- 联合回合单独计数。256个联合回合不能被计成两个独立模型各新增256后再加总512。',
 '- 回退到原单模型入口/原冻结组合。源环境和学习器文件未改，原模型未覆盖。新代码只在独立目录，未推送Git或PR。','',
 '## 阶段结果','', '|组合|执行方式|联合回合|覆盖率|获救比例|潜在冲突/步|全员获救|全任务完成|','|---|---|---:|---:|---:|---:|---:|---:|']
for group in ['hasac_mat','hasac_hatrpo']:
 d=RUN/group
 if not d.exists():continue
 for p in sorted(d.glob('validation-*.json')):
  if not p.stem.split('-')[1].isdigit():continue
  v=read(p);s=v['summary'];ep=int(p.stem.split('-')[1]);mode='固定动作（补充诊断）' if 'greedy' in p.stem else '随机动作（预登记验证）';report.append(f"|{group}|{mode}|{ep}|{s['coverage']:.2%}|{s['RCR']:.2%}|{s['ACC']:.5f}|{s['CRT']['valid']}/{v['n']}|{v['whole_task_completed']}/{v['n']}|")
report+=['','固定动作诊断仅在两套训练结束后执行：训练前、已按随机验证选中的版本、最终256回合版本各2父水场×4布局×2分工=16次，相同版本不重复执行，不用于选择检查点。随机动作与固定动作结果分开解释；上述获救是模拟服务规则下的人数比例，不是真实救援成功率。']
if (BASE/'archive.stderr.log').exists():report+=['','归档辅助PowerShell脚本被系统执行策略拒绝，未修改或绕过该策略。训练正常继续；本报告由手动调用归档程序保存。']
if (RUN/'failure.json').exists():report+=['','运行失败已记录在failure.json，未反复重启。']
if state['status']=='completed':
 report+=['','## 最终选择与计算量','']
 for group in ['hasac_mat','hasac_hatrpo']:
  result=read(RUN/group/'status.json');best=result['best_joint_episode']
  choice='没有阶段满足预登记改善条件，保留训练前组合' if best==0 else f'按预登记开发验证条件保留第{best}联合回合；仍需独立测试'
  report.append(f"- {group}：新增{result['joint_episodes']}联合回合，耗时{result['elapsed_s']/60:.2f}分钟。{choice}。")
 report+=['','两套合计新增512个联合回合；这是每套一个训练种子的初步筛选。固定验证布局被反复用于选择，不能当未见场景泛化结论。各候选所有阶段和原检查点保留，不用两模型学习次数相加冒充任务回合。']
 report+=['','HASAC+HATRPO最终256回合的平均获救与覆盖高于第64回合，但潜在冲突/步0.022396超过0.02门槛，且较弱分工获救比例从0.94375降至0.940625；因此按预先固定规则不替换第64回合。HASAC+MAT所有训练阶段均未满足改善条件。','', '## 使用范围','', '这些结果仅支持当前模拟环境和开发验证布局。获救比例是20名模拟目标中完成服务的比例；全员获救计数要求全部20名完成服务；全任务完成还要求全部10000格被观察。三者不能互换。本轮全部随机验证的全任务完成数均为0。水情、建筑避障、通信、真实续航和实际救援动作仍不因本轮训练而获得现场验证。']
 if (RUN/'hasac_hatrpo/validation-0064-greedy.json').exists():
  g=read(RUN/'hasac_hatrpo/validation-0064-greedy.json')['summary']
  report.append(f"固定动作补充诊断也存在明确局限：保留的HASAC+HATRPO第64回合版获救{g['RCR']:.2%}、覆盖{g['coverage']:.2%}，但潜在冲突/步{g['ACC']:.5f}。不能将随机执行下的改善外推到固定动作执行；本轮并未证明组合优于原最佳单模型，也不支持直接替换现有默认。")
 if (RUN/'integrity-audit.json').exists():
  audit=read(RUN/'integrity-audit.json');report.append(f"完整性审计：{audit['status']}。父模型和冻结训练源码哈希未变；两套各256连续回合、场景/分工平衡、相同训练序列、最终参数/优化器/回放数值有限及检查点哈希已核验。该审计不代表性能通过。")
report_text='\n'.join(report)+'\n'
for target in [OUT,DOC]:
 for source in [BASE/'joint_train.py',BASE/'publish_joint.py',BASE/'final_greedy.py',BASE/'audit_joint.py',BASE/'archive.stderr.log']:
  if not source.exists():continue
  shutil.copy2(source,target/source.name)
 for p in RUN.rglob('*'):
  if not p.is_file() or p.name.endswith('.partial'):continue
  if p.suffix=='.pt' and (not finished or target==DOC):continue
  dest=target/p.relative_to(RUN);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 (target/'report.md').write_text(report_text,encoding='utf8')
 (target/'live-run.json').write_text(json.dumps(dict(path=str(RUN),pid=state.get('pid'),status=state['status']),ensure_ascii=False,indent=2),encoding='utf8')
index=[dict(file=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact-index.json']
for target in [OUT,DOC]:(target/'artifact-index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf8')
stage='完成归档' if state['status']=='completed' else '异常归档' if finished else '已启动'
count='各已完成256联合回合' if state['status']=='completed' else '各计划新增256联合回合'
conclusion=' HASAC+MAT无阶段通过，保留训练前组合；HASAC+HATRPO按预登记规则保留第64回合（获救94.53%、覆盖94.45%、潜在冲突/步0.01094），256回合虽达获救95.00%/覆盖95.45%但冲突及较弱分工退化，未替换。全部随机验证完整任务完成0。' if state['status']=='completed' else ''
note=f"## 2026-11-01 两套无人机组合联合训练{stage}\n\nHASAC1024+MAT256、HASAC1024+HATRPO1024{count}，当前{state['status']}。新增公开角色位、按实际所有权更新和混合队友目标采样；五项启动检查通过。每64回合验证两分工32次，封存50测试未用。{conclusion} 实时目录{RUN}；归档outputs/uav6-joint-training-20261101及uav-workbench/docs/joint-training-20261101。不要重复启动。无单调改进承诺，原模型和坏结果保留，未推送PR。\n\n"
for base in [OUTPUTS,REPO/'terrain-workbench/docs']:
 for name in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
  p=base/name
  if not p.exists():continue
  if name=='HYRA自动续作进度.md':p.write_text(note+p.read_text(encoding='utf8'),encoding='utf8')
  else:
   with p.open('a',encoding='utf8') as f:f.write('\n\n'+note)
print(json.dumps(dict(status=state['status'],archived=str(OUT),finished=finished),ensure_ascii=True))
