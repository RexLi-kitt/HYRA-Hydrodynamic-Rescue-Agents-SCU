from pathlib import Path
import json,shutil,hashlib,zipfile
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round83'
REPO=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029');DOC=REPO/'terrain-workbench/docs'
S=Path(__file__).parent
status=json.loads((R/'status.json').read_text())
assert status['state']!='running'
verification=json.loads((R/'independent-verification.json').read_text())
controls=json.loads((R/'controls.json').read_text())
probe=json.loads((R/'stage2_probe/result.json').read_text())
summary=f'''## 2026-10-30 第83轮：RK回滚候选诊断已收尾

独立候选round83，不改生产，不自动推送PR #10。2400s运行状态={status['state']}，退出码={status['exit_code']}，实际到达={status.get('end_s')}s，墙钟={status['wall_s']:.2f}s，独立内部关口={verification['independent_gate_pass']}。现场新增通过0，不能继承旧HYRA评分或称总体60%。

3个无触发回归保持原输出/实际步，含源折点合成失败由原退出42变为候选成功，4次拒绝后入水恰为.003m³、体积误差约2e-15m³。第二阶段专门探针未触发拒绝，但Carlisle实际在2197.6531s的第二次尝试触发RK2拒绝，之后成功接受。Carlisle实际拒绝次数={status.get('rejections',0)}，最小接受步={status.get('minimum_dt')}s；原始全部输出场/尝试日志/源码和哈希见round83及docs/round83-artifacts，拒绝前9场仅保存最新一次而非每次，限制明确保留。

候选有限适用范围：CPU64/OMP1，无雨无重启，正QFIX/QVAR。回滚原始9场，重算摩阻/源/边界/两阶段；所有接受账目只写一次。prestate.bin为诊断完整场，不是已支持的重启文件。未接网页，网页8766仍是原HYRA服务，勿把常驻server.py进程当实验。

下一步先读audit-round83-20261030.md的结果与否决条件，不重复本配置；本次两阶段均有拒绝记录且内部关口通过，下一单因素用dtmax=.5对同2400s做实际时间步敏感性，比较源体积/输出区域传播/有限非负和成本后再决定完整作者时窗。完整事件稳定性尚未关闭，不盲重启。

'''
report='''# 第83轮：全域RK尝试与接受事务候选

## 更改日志

日期2026-10-30；基线round77独立CPU64，Git研究快照fc7584c。只在round83/lisflood8-rk-retry/swe/dg2.cpp隔离修改，未改变网页、原HYRA、原观测、床面、HLL通量、糙率或生产默认。

新增SavedFlow83保存/恢复H、HU、HV及其x/y坡度共9个原始场；checked_increments83在每阶段写入任一单元前计算全域增量，检查前向Euler水深及SSP结果有限非负，失败用异常返回外层。rk_stage1/2使用已核算增量。solve内最多8次尝试；拒绝后从原始9场重算源、水静力变量、通量和摩阻，派生中间场由后续阶段完整覆盖。时间、最大场、统计、accepted_timesteps和accepted_source_volumes均仅在接受后提交。accepted_positive_point_step接受外部dt上限，仍精确积分折点。

回退为使用round77 exe和原输入；新候选独立编译，不升级默认。候选拒绝步长选择min(原步长一半, .9×失败局部冻结通量排空界)，每次重新计算，绝不将冻结界当下一次通量正确性的证明。每步最多8次；dt<1e-10s或不推进时间则退出51并保存诊断原始场，不能循环重启。该因子是回溯安全余量，不是依据观测拟合。原无雨/无restart/仅正点源/OMP1限制保留。

## 原因与复现依据

第82轮独立水量回放已经复现原RK1向外流出多于可用水，且均值CFL未覆盖异常重构面。候选先隔离时间步接受问题而不同时改通量/床面。所有输入仍为原尺度；2400s是明确跨原2179.0073s失败位置的诊断终点，不是现场完整事件。为了保留状态，输出间隔改为300s；原质量账目已有300s事件，此改动不额外增加早期事件限制。原164000s目标和此前失败保留。

源码审查：摩阻会修改均值动量及动量坡度，单独回滚H不充分；坡度限制也改原系数，所以保存全部9场。Qin/Qout每次尝试重置，累计边界体积在最终接受后才更新。中间场和星值/通量为重新派生量。前态二进制存xsz/ysz/9、t/dt和9个float64场，字段顺序见SavedFlow83；它不是可直接被原求解器读取的checkpoint，尚不支持恢复长算例。

替代取舍：未采用截负水深或直接限速，以免隐藏水量/动量问题；未换NS或GPU。若通过正性仍出现巨量回溯或成本恶化，应根据状态转向有依据的动量/速度重构，不能一直缩步。文献依据及其不直接适用于本DG2的限制沿用第82轮报告。

## 目标可行性与结果

已实现隔离事务控制和编译；相关小测试通过。3个无触发闭水/开边界/源+边界例保留原水深、坡度输出和实际步长（逐字比较见独立核验）。25格人为动量坡度压力例有独立水量期望：原版退出42，新版4次拒绝、15接受步至.2s、总源体积.003m³、总水量.253m³，未重复注水。它是数值回归而非实验室或现场观测。

补充探针将首步置于原冻结排空界下方，希望覆盖RK2拒绝；完成但未触发拒绝，不能靠该探针算通过。实际Carlisle运行在t=2197.6530538s的第二次尝试触发RK2拒绝，将dt=.09948854缩至.00372677后重新完成，实际轨迹提供该分支覆盖。总共8次RK1拒绝、1次RK2拒绝，9个失败尝试均有时间/阶段/单元/dt记录；last-rejection-prestate.bin仅保留最近一次首次尝试的9场，不包含所有拒绝状态，未来若需逐拒绝重放必须改为唯一文件名，不能称本轮已具备全部重放状态。实际诊断结果如下。

```json
'''+json.dumps(status,ensure_ascii=False,indent=2)+'''
```

独立复核：
```json
'''+json.dumps(verification,ensure_ascii=False,indent=2)+'''
```

没有新增现场定性验收；原263点、5范围和所有历史不一致不变。本轮输入/时间基准、HVAR与自由出口、初态和河宽表示证据缺口仍未闭合。完整164000s未在本轮运行，不能把2400s稳定当完整洪水正确。

下一步：本次独立内部关口通过，且实际轨迹触发RK2拒绝。下一单因素固定原输入、2400s和300s输出，dt上限1→.5；原全部8输出场比较水量、有限非负、dt分布、数值湿格和传播差异，数值湿阈值只用内部敏感性诊断，不是现场干湿真值。不按更好看的结果挑配置；同样15分钟墙钟上限，不循环重启。得到实际时间步敏感性后再决定完整条件研究，原164000s不能因本2400s通过而视为完成。若失败、触及8次或资源上限，保存状态后否决继续长跑，按错误阶段定位或转向动量重构。旧40点/墙位/全库290测试跳过，因为没有改其生产路径；不得把此跳过误称新全库通过。回归目录保留全部输入，source/step/boundary账目均独立核对。

所有本轮变化仅本地，已上传PR #10不自动更新。原始小测试、2400s输入/输出/拒绝记录打包双份，manifest记录SHA256。当前后台无本诊断实验；网页服务可能仍常驻。
'''
(R/'audit-round83-20261030.md').write_text(report,encoding='utf-8')
for name in ['build_retry83.py','test_retry83.py','run_retry83.py','verify_retry83.py','publish_retry83.py']:
    shutil.copy2(S/name,R/name)
shutil.copy2(R/'lisflood8-rk-retry/swe/dg2.cpp',R/'dg2-candidate.cpp')
archive=R/'raw-experiments.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for sub in ['controls','retry_source_baseline','retry_source_candidate','stage2_probe','carlisle_2400']:
        for p in sorted((R/sub).rglob('*')):
            if p.is_file():z.write(p,p.relative_to(R))
manifest=[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(R.iterdir()) if p.is_file() and p.name!='manifest.json']
(R/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
dest=DOC/'round83-artifacts';dest.mkdir(exist_ok=True)
for p in R.iterdir():
    if p.is_file():shutil.copy2(p,dest/p.name)
shutil.copy2(R/'audit-round83-20261030.md',DOC/'audit-round83-20261030.md')
for name in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
    for root in [O,DOC]:
        p=root/name
        if p.exists():p.write_text(summary+p.read_text(encoding='utf-8-sig'),encoding='utf-8')
assert all(hashlib.sha256((dest/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in manifest)
print(json.dumps({'published':str(dest),'files':len(manifest),'archive_bytes':archive.stat().st_size,'all_hashes_match':True}))
