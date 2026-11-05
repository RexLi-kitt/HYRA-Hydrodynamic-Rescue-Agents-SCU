from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parent;REPO=ROOT/'hyra-uav-pr-20261101';RUN=ROOT/'uav-frozen-rigor-20261104/run01';OUT=Path('C:/Users/Lenovo/Documents/ChatGPT/无人机集群搜救大创/outputs')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
status=read(RUN/'status.json');protocol=read(RUN/'protocol.json');completed=status['state']=='completed'
lines=['冻结模型严谨性复核 — 2026-11-04','',
 '用户授权：先冻结当前93.125%版本，补充独立验证、简单模型对照、不同地形与落水/再漂流机制，随后以证据决定是否加训。当前GRU不改权重、不因测试换检查点。',
 '设计：40张全新同类SWE地图（1105501..5540）为主检验；4双河谷+4斜向河谷为结构泛化，分别报告；另外4个布局共用1个制造的干湿脉冲水场，单独报告临界落水/再次漂流，不冒充4个独立水动力事件。',
 '对照：完整coverage100、flow_coverage；简单统计预计人数；256维历史输入线性softplus预计人数；冻结FineGRU142-e16。三个预计人数模型使用同一CountCoupledEnv、30秒缓存、半权流动运输、系数8。全部使用共享避碰修复，同20m视野/40人4机/660秒个人窗口/2400秒任务/10秒投放/2m或1mps强制落水阈值。',
 '简单模型训练：旧24个FIT父场，每场48历史片段，1152张标签图，同一64维×4帧公开输入和FIT归一化。统计仅拟合1个总量系数；线性257参数，用同样D4变换，Adam.001、批32、64监督epoch，按8个旧CAL父场平均Poisson误差在8/16/32/64选一次。新的40+8地图在模型冻结之后才生成。不能把线性训练轮数或260回放说成GRU加训。',
 '判别：人数优先，主面板>=85%且不低于两规划，>=95%人数场景率至少领先15个百分点作为本批点估计；若要说稳定领先15个百分点，配对父场bootstrap95%区间下界也要达到15。额外比较GRU与统计、线性，不能只比弱对照。结构/再漂流不利项不通过混合平均隐藏。',
 '机械条件与数值：13项制造输入规则检查（9再次漂流+4初次落水）；SWE场有限、非负、闭域源体积账本与储水差核对，输出5秒、运动/观测1秒。制造再漂流场是外部规定h/u/v，明确不是求解器准确性验证；其水体变化来自给定干湿过程，不作真实洪水结论。',
 '当前状态：'+status['state']]
if completed:
 s=read(RUN/'summary.json');rows=[json.loads(v) for v in (RUN/'results.jsonl').read_text(encoding='utf-8').splitlines()];cases=read(RUN/'scenarios.json')
 assert len(rows)==260 and len({(r['case'],r['method']) for r in rows})==260
 for r in rows:
  assert len(r['persons'])==40 and sum(p['status']==2 for p in r['persons'])==r['supplied'];assert r['min_separation']>=5-1e-8
  for p in r['persons']:
   if p['status']==2:assert p['fall']+10<=p['supplied_at']<=p['deadline']
 for p,h in {**protocol['source_sha256'],**protocol['input_sha256'],**read(RUN/'input-lock.json')['water_sha256'],**read(RUN/'predictor-lock.json')}.items():assert sha(p)==h,p
 triggers={f:{t:sum(ev['trigger']==t for c in cases if c['family']==f for ev in c['scenario']['events']) for t in ['threshold','random']} for f in s['results']}
 audit=dict(rows=260,source_inputs_unchanged=True,fall_triggers=triggers,min_swept_separation=min(r['min_separation'] for r in rows),primary_independent_parents=40,structural_parents=8,manufactured_shared_fields=1,manufactured_layouts=4)
 write(RUN/'independent-audit.json',audit)
 lines+=['','面板 | 方法 | 人数/总人数 | 人数率 | 全员场景 | >=95%场景 | 实际再漂流次数']
 for family,methods in s['results'].items():
  for name,a in methods.items():lines.append(f"{family} | {name} | {a['supplied']}/{a['people']} | {a['supplied']/a['people']:.3%} | {a['all_scenes']}/{a['scenes']} | {a['p95_scenes']}/{a['scenes']} | {a['refloats']}")
 lines+=['','主面板配对地图区间：']
 for n,a in s['paired_primary'].items():lines.append(f"GRU相对{n}：人数差{a['people_pp']:+.3f}pp，95%区间{a['people_95']}；>=95%场景差{a['scene_pp']:+.3f}pp，95%区间{a['scene_95']}；人数胜{a['wins']}、负{a['losses']}（40父场）。")
 lines+=['',f"本批点估计目标={s['point_target']}；稳健领先15pp下界目标={s['stable_15pp_lower_bound']}。历史93.125%/45%为另一批20场结果，原样保留，不与本批混合挑分。",'实际初次落水原因：'+json.dumps(triggers,ensure_ascii=False)]
 a=s['results']['original'];g=a['gru-frozen']
 deep_dominates=all(g['supplied']>a[n]['supplied'] and s['paired_primary'][n]['people_95'][0]>0 for n in ['statistic','linear'])
 structural_ok=all(s['results'][f]['gru-frozen']['supplied']>=max(s['results'][f][n]['supplied'] for n in ['coverage100','flow_coverage']) for f in ['two_valleys','diagonal'])
 mech=s['results']['manufactured_rewet'];mechanism_ok=mech['gru-frozen']['supplied']>=max(mech[n]['supplied'] for n in ['coverage100','flow_coverage'])
 decision=dict(gru_beats_simple_with_positive_people_interval=deep_dominates,structural_people_not_below_planners=structural_ok,manufactured_mechanism_people_not_below_planners=mechanism_ok,gru_extra_training_started=False)
 if not deep_dominates:decision['next']='GRU特有收益尚未证实，先分析同接口简单模型；不盲目增加GRU轮数。'
 elif not structural_ok or not mechanism_ok:decision['next']='同类场景已证明冻结GRU相对本轮简单对照的收益，但压力条件泛化不足。先在另设开发场隔离预测加权导致的重复巡查与探索不足，再设计只依赖当前观测的探索保底；随后补充有数值依据的多地形/涨退水FIT数据，开展多种子8/16/32监督epoch小试，仅CAL改善才延长。不在旧分布直接堆轮数，不拿本批测试选新模型，不把制造场当真实水动力依据。'
 else:decision['next']='保留冻结GRU版本；如继续提升，采用新增多地形FIT数据及多种子8/16/32小试，仅在独立CAL改善后才延长。当前验证结果本身不构成必须加训的证据。'
 write(RUN/'decision.json',decision);lines+=['','训练决策：'+decision['next'],f"成本：{s['elapsed_s']:.1f}秒；48SWE场+1制造场，8校准收集、260任务回放、线性64监督epoch、GRU0epoch。一个实验任务使用4CPU工作进程，每进程torch1线程；未用GPU，环境回放为主要成本。",'回归：全部10400人员记录、投放期限与10秒规则、安全距离、源码配置权重哈希；每个部署方法禁用真实区域计数调用，检查隐藏人员和未来水场不改变公开输入。旧数据、权重、模型和生产默认不变。','局限：同类主面板仍来自单一合成生成器；不同地形每类4图样本小；制造干湿场只能做规则与压力测试，不代表真实涨退物理。初始阈值与医学安全阈值无对应保证；完成定义为物资投放，不是现实生存率。40人同图相关，不能按1600独立人计算显著性。只有一个冻结GRU训练种子/检查点，未估计训练种子差异。各95%区间为逐对比较，不是多比较同时置信界。','下一步必须依据decision.json；不把本次看过的新TEST偷偷用于选检查点。没有自动新增长训队列、远程推送或PR。']
 secondary=read(RUN/'secondary-audit.json')
 lines+=['','第二份独立统计核验：全部260行/10400人逐条重算、所有汇总与配对区间复算相符；48份新SWE场哈希与旧32份FIT/CAL场无重合。制造场4布局共用1场，独立性不虚增。','失败原因按截止前是否被发现核对（期满后才看见不算及时发现）：']
 for family,mm in secondary['failure_and_refloat'].items():
  for name,v in mm.items():lines.append(f"{family}/{name}: 失败{v['failed']}，截止前未发现{v['failed_not_seen_by_deadline']}，期满{v['expired']}，漂出区域{v['outside']}；实际再漂流人数{v['people_refloated']}，经历再漂流后完成{v['supplied_after_any_refloat']}。")
 replay=read(RUN/'pressure-replay/result.json')
 lines+=['','压力失效事后诊断：单独精确复放manufactured-0的coverage100/统计/GRU，三者逐人结果与锁定测试完全一致；这3次是复现，不增加独立样本量。']
 for r in replay['results']:
  p=next(x for x in r['snapshots'] if x['t']==479)
  lines.append(f"{r['method']}: 最终{r['supplied']}/40，观测覆盖{r['final_observed_cells']}/10000格，总飞行距离{sum(r['distance_m']):.1f}m；首次退水前479s已观测{p['ever_observed_cells']}格、发现{p['discovered_by_now']}人、完成{p['supplied']}人、再漂流{p['refloats']}次。")
 lines+=['观察到GRU在第一次退水前已偏向外围重复巡查，中部未观测区预测人数很低；连续飞行却覆盖不足，不能归为之前已修复的避碰停滞。统计对照也受影响，需隔离共享预测/风险加权与探索反馈，不能假定多训GRU就能解决。该人工场同时改变水深、湿区范围、人员分布和干湿过程，不能据此声称再次漂流是唯一因果。',f"额外3次只读复现成本{replay['elapsed_s']:.1f}秒。运行统计与初次落水规则检查脚本也一并留档。",'', '全量逐场人数（40人为一场）：面板 | 场景 | coverage | flow | statistic | linear | GRU']
 for c in cases:
  rr={r['method']:r for r in rows if r['case']==c['id']};lines.append(c['family']+' | '+c['id']+' | '+' | '.join(str(rr[n]['supplied']) for n in ['coverage100','flow_coverage','statistic','linear','gru-frozen']))
else:lines+=['','预计48个新SWE场、260任务回放，固定样本量，不按暂时达标提前停止；当前仅一项任务运行。']
(RUN/'严谨性复核与全部数据.txt').write_text('\n'.join(lines),encoding='utf-8-sig')
(RUN/'source').mkdir(exist_ok=True)
for n in ['rigor_support.py','rigor_trial.py']:shutil.copy2(REPO/'uav-workbench/experiments/flood-drop-search-return-v1'/n,RUN/'source'/n)
shutil.copy2(ROOT/'audit_fall_thresholds.py',RUN/'source/audit_fall_thresholds.py')
for n in ['audit_rigor_results.py','inspect_rigor_pressure.py','archive_rigor.py']:shutil.copy2(ROOT/n,RUN/'source'/n)
files=[p for p in RUN.rglob('*') if p.is_file() and 'fields' not in p.relative_to(RUN).parts and p.suffix!='.npz' and p.name!='artifact-sha256.json']
write(RUN/'artifact-sha256.json',{str(p.relative_to(RUN)):sha(p) for p in files})
for dest in [OUT/'uav-frozen-rigor-20261104',REPO/'terrain-workbench/docs/uav-frozen-rigor-20261104',REPO/'uav-workbench/docs/uav-frozen-rigor-20261104']:
 (dest/'run01').mkdir(parents=True,exist_ok=True)
 for p in files+[RUN/'artifact-sha256.json']:
  target=dest/'run01'/p.relative_to(RUN);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
 shutil.copy2(RUN/'严谨性复核与全部数据.txt',dest/'严谨性复核与全部数据.txt')
marker='冻结模型严谨性复核'+('完成' if completed else '启动')
entry='\n\n## 2026-11-04 '+marker+'\n'
if completed:
 entry+='40个全新同生成器父场主检验：'+json.dumps(s['results']['original'],ensure_ascii=False)+'\n'
 entry+='点估计目标='+str(s['point_target'])+'；两规划配对95%区间下界均领先15个百分点='+str(s['stable_15pp_lower_bound'])+'。\n'
 entry+='结构/机制面板分别保留，不混合掩盖失败：'+json.dumps({k:v for k,v in s['results'].items() if k!='original'},ensure_ascii=False)+'\n'
 entry+='260任务回放、48新SWE场、1制造干湿水场4布局；线性对照64监督epoch、GRU新增0epoch；完整逐人记录与源码/输入/权重哈希检查通过。\n'
 entry+='决策：'+read(RUN/'decision.json')['next']+'\n本轮无待跑训练队列；生产默认未替换，未推送。\n'
else:entry+=lines[-1]+'\n'
entry+='完整记录：uav-frozen-rigor-20261104/严谨性复核与全部数据.txt\n'
for directory in [OUT,REPO/'terrain-workbench/docs']:
 for name in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
  p=directory/name
  if marker not in p.read_text(encoding='utf-8-sig'):
   with p.open('a',encoding='utf-8') as f:f.write(entry)
print(json.dumps(dict(state=status['state'],report=str(RUN/'严谨性复核与全部数据.txt'),summary=s if completed else None),ensure_ascii=False))
