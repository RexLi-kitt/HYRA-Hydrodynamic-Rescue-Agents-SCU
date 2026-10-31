from pathlib import Path
import hashlib,json,shutil,zipfile
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round87';S=Path(__file__).parent
D=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029\terrain-workbench\docs')
status=json.loads((R/'status.json').read_text());assert status['state']!='running'
reg=json.loads((R/'regressions.json').read_text());passed=status.get('internal_gate_pass',False)
cmp=json.loads((R/'extension-comparison.json').read_text()) if passed else None
note=f"有摩阻3类静水及两时间步均匀坡流对照全部通过；2400s延长状态{status['state']}，实际到达{status['end_s']}s，内部关口{passed}。"
table='未通过内部关口，不评分不完整场。'
if cmp:
    last=cmp['frames'][-1]['thresholds'][1]
    note+=f"RK拒绝{status['rejections']}次，600s旧候选前缀逐字段/实际步一致；2400s同CFL下1mm湿格由旧式{last['old_wet']}变为候选{last['candidate_wet']}，两者有{last['changed_cells']}格判断不同。这不是实测改善率。"
    rows=['|时刻s|同CFL最大深差m|1mm旧湿/新湿|1mm旧独有/新独有|1cm变化格|','|---|---|---|---|---|']
    for f in cmp['frames']:
        a=f['thresholds'][1];b=f['thresholds'][2];rows.append(f"|{f['time_s']}|{f['max_abs_depth_change_m']:.8g}|{a['old_wet']} / {a['candidate_wet']}|{a['old_only_wet']} / {a['candidate_only_wet']}|{b['changed_cells']}|")
    table='\n'.join(rows)
report=f'''# 第87轮：有摩阻独立回归与候选2400s后段验证

## 更改日志

2026-10-30。基线round83（原对角摩阻）与round86（解析向量摩阻隔离候选）；本轮不改任何求解器源码，candidate exe SHA=884a074e36474ae86949579525af295a0e9550c24d2ca138cd2612dcd4e7a771。新增regress_friction87、extend_friction87、analyze_extension87及本发布脚本。3类静水fixture复用round79，糙率0改.06；均匀坡流为新解析fixture。实际归档候选只把sim_time600改2400，CFL.33/dtmax1/output300及全部输入不变。生产默认、用户网页、PR保持原样；回退为继续使用round83/生产exe，候选和全部旧结果保存。

## 原因、预登记与物理依据

上一轮只修正固定水深摩阻子步方向保持，尚未检验它与床坡源项、静水平衡和旧负深时段的耦合。本轮先过回归门槛，失败则不得长跑。静水阈值沿用既有double内部检查：H漂移<1e-10m、体积差<1e-8m³，有限非负。全部记录是数值回归，不是现场验收标准。

均匀坡流H=1m，n=.06，床坡S=.001，解析u=H^(2/3)sqrt(S)/n=0.5270462767m/s。80×8个1m格，模拟2s，固定核心区距x边界30m，物理最快波传播约7.4m；封闭墙会扰动边界格，完整场全部保留，不能把边界反射后的区域当无限均匀流。只在预登记核心区检查源项平衡，dt=.02/.01的允许漂移2gSdt是保守的重力步量级内部截断检验，非真实流速容差；另要求细步误差下降和候选不劣于基线。斜率、核区、阈值均先登记后运行。

## 结果与目标可行性

{note}

静水最大漂移约2.8e-16m，两实现均保持；均匀坡流dt=.02时速度误差从1.40625e-5降至7.03372e-6m/s，dt=.01从7.03246e-6降至3.51685e-6m/s。细步约减半，仍有分裂截断误差，不是精确整体平衡或已证明二阶。全部边界区漂移保留在regressions.json，未据核心区正确删去边界反射。

{table}

表中是同一CFL的新旧公式预测差异，不是方向准确率、漏淹率或新候选的时间步收敛结果。0.165候选仅有600s，目前没有2400s配对；严禁把新.33与旧.165相近就算新公式稳定。所有8帧三个预登记阈值的完整变化掩膜和深差保存full-extension-comparison.npz。湿区改变必须用独立同域同阶段观测核验后，才可判断改善或退化。

```json
{json.dumps(status,indent=2)}
```

实现状态：局部摩阻机制修正及有摩阻耦合回归已通过；后段事件结果按上述状态记录；完整164000s及现实受淹/主流向/道路/起降点仍未验证，现场新增通过0。原9次拒绝的round83失败处理和round81原退出42均保留，不把旧错误清掉。由于当前不支持重启，延长必须重算前600s，已核对其逐输出/实际步前缀，不作为新增独立证据。

回归范围10个小运行：2实现×3静水＋2实现×2步长坡流；无雨、正点源、无重启、OMP1限定不变。跳过旧40点/HLLC/墙位（不相关不重跑），未做新网格敏感性、非零雨/抽水、多线程或新的实验室实测。代价见各run.log、regressions.json和status.json，本轮只有一项较重任务且已结束。代码输入哈希、全原始结果、脚本双份归档。

输入证据核验：已读本地Zenodo7064320元数据，说明是论文测试参数归档，但没有绝对UTC或高程基准。另检索到Parkes等2013论文有关水文图的起时线索；Reading全文入口被访问拒绝，AMS入口403，未绕过，不能用搜索摘要绑定本归档时轴。作者Rong博士论文入口超过网页工具大小限制，也未取得所需正文。故原HVAR/free、绝对起时、初态/河宽缺口仍未闭合；无输入变更。检索URL和访问结果另见input-evidence-attempts.json，不将检索算作实际验收。

下一步：若本轮内部关口通过，冻结同exe、同2400s/300s输出、同输入，仅CFL.165补候选后段的真正配对（原.33已完成不重跑）；预登记900s墙钟上限、失败不重启。比较全部8帧/3阈值和实际步分布，保留不利项；仅有600s改善不得自动外推。若内部门槛失败，保存并定位本轮失败，不运行配对。完整实事件长跑暂不启动；同时继续获取能确证归档边界和时间的原始资料，不挑更合评分的输入。
'''
(R/'audit-round87-20261030.md').write_text(report,encoding='utf-8')
attempts={'date':'2026-10-30','local_metadata':'round73/zenodo7064320.json: test parameter files; no absolute UTC/datum metadata','attempts':[{'url':'https://centaur.reading.ac.uk/35476/1/2012ei000475%252E1.pdf','outcome':'access denied; no bypass'},{'url':'https://journals.ametsoc.org/view/journals/eint/17/6/2012ei000475.1.xml','outcome':'403; no bypass'},{'url':'https://research-information.bris.ac.uk/ws/portalfiles/portal/430852101/Final_Copy_2024_10_15_Rong_Y_T_PhD.pdf','outcome':'web tool content-length limit; needed body not verified'}],'input_changed':False,'epoch_closed':False}
(R/'input-evidence-attempts.json').write_text(json.dumps(attempts,indent=2))
for n in ['regress_friction87.py','extend_friction87.py','analyze_extension87.py','publish_extension87.py']:shutil.copy2(S/n,R/n)
with zipfile.ZipFile(R/'raw-experiments.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for folder in ['baseline','candidate','carlisle_2400_cfl033']:
        for p in (R/folder).rglob('*'):
            if p.is_file():z.write(p,p.relative_to(R))
manifest=[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(R.iterdir()) if p.is_file() and p.name!='manifest.json']
(R/'manifest.json').write_text(json.dumps(manifest,indent=2));dest=D/'round87-artifacts';dest.mkdir(exist_ok=True)
for p in R.iterdir():
    if p.is_file():shutil.copy2(p,dest/p.name)
shutil.copy2(R/'audit-round87-20261030.md',D/'audit-round87-20261030.md')
update=f'''## 2026-10-30 第87轮：有摩阻回归与2400s候选后段收尾

{note}

仅round86隔离exe，生产默认/网页/PR不变，源码未新增修改。原始结果、全8帧三个阈值比较及输入配置哈希、10个有摩阻回归保存在round87与docs/round87-artifacts。新增现场通过0；没有新的绝对UTC或HVAR基准闭合，不能说真实事件已复现。

下一单元若内部关口通过：同exe补CFL.165的2400s配对，.33已完成不得重复；同输入/输出/阈值，900s墙钟，失败不重启；先核查实际进程和最新记录。若内部关口失败则诊断，不启动配对。当前延长任务已退出，网页8766保留；不启动整夜完整事件。

'''
for n in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
    for root in [O,D]:
        p=root/n
        if p.exists():p.write_text(update+p.read_text(encoding='utf-8-sig'),encoding='utf-8')
assert all(hashlib.sha256((dest/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in manifest)
print(json.dumps({'note':note,'files':len(manifest),'dual_hashes_match':True},ensure_ascii=False))
