from pathlib import Path
import json,hashlib,shutil,zipfile
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round86';S=Path(__file__).parent
D=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029\terrain-workbench\docs')
diag=json.loads((R/'diagnosis.json').read_text());cmp=json.loads((R/'coupled-comparison.json').read_text());runs=json.loads((R/'runs.json').read_text());op=json.loads((R/'operator-verification.json').read_text())
assert json.loads((R/'status.json').read_text())['state']=='completed'
lines=['|时刻s|原/新CFL对照最大深差m|原/新L1深差体积m³|1e-6m变化格原→新|1mm变化格原→新|1cm变化格原→新|','|---|---|---|---|---|---|']
for f in cmp['frames']:
    a=f['old_cfl_pair'];b=f['new_cfl_pair'];counts=f['sensitivity_change']['changed_cells_before_after']
    lines.append(f"|{f['time_s']}|{a['max_abs_depth_m']:.8g} / {b['max_abs_depth_m']:.8g}|{a['L1_depth_volume_m3']:.8g} / {b['L1_depth_volume_m3']:.8g}|{counts[0][0]}→{counts[0][1]}|{counts[1][0]}→{counts[1][1]}|{counts[2][0]}→{counts[2][1]}|")
table='\n'.join(lines)
f=cmp['frames'][-1];counts=f['sensitivity_change']['changed_cells_before_after'];improves=all(x['sensitivity_change']['L1_reduction_fraction']>0 and all(b<=a for a,b in x['sensitivity_change']['changed_cells_before_after']) for x in cmp['frames'])
decision=('限定两时刻的L1与全部预定阈值差异均未恶化，保留隔离候选进入后续独立回归；不据此升级生产或宣布现场提升。' if improves else '耦合敏感性没有在全部预定项目上改善，不能宣布整场改进；保留局部方向修复证据和全部不利结果，不升级生产，先解释剩余分裂/限幅耦合。')
note=f"原摩阻在300/600s保存场的独立局部探针最大额外转向15.8428°/15.4184°；共享解析衰减候选降至约6e-15°。两组600s均通过内部关口；600s的1mm时间步分类差异{counts[1][0]}→{counts[1][1]}格，1cm差异{counts[2][0]}→{counts[2][1]}格。{decision}"
report=f'''# 第86轮：早期分流定位与各向同性摩阻隔离修正

## 更改日志

2026-10-30。基线round83隔离LF8 CPU64候选及round85细步结果；源码仅改复制目录round86/lisflood8-isotropic-friction/swe/dg2/friction.cpp中的dg2::friction。逐文件检查仅此文件不同，构建SHA见candidate-build.json。原来的x/y独立对角分母改为同一解析衰减系数，roughness=.06、DepthThresh=.001、SpeedThresh=1e-9、原apply_friction顺序和Gauss斜率处理、点源、通量、边界、RK回滚均保留。生产源码/默认/网页/PR没有修改。回退使用round83 exe和原配置。新诊断、候选、分析脚本及全部原结果双份归档。

## 更改原因、物理依据与取舍

先复用原300/600s全场；三个最近源分区的体积各自一致到约1e-9m³，而水深分布存在差异。300s最大深差点离源1约103m，600s约197m，均落在归档width>0的名义河道格。最近源分区仅几何诊断，不是已验证河道流域；width>0不是真实湿区，所有非河道差点仍保留。

原dg2::friction针对x/y采用不同的线性化分母。各向同性Manning摩阻在固定水深的单独子步应保持动量方向，但使用原C++函数（函数体不改，单独编译）对300/600s所有活动格逐个测试，出现最大15.8428°/15.4184°额外转向。它说明这一离散子步有方向误差，不证明主流向实测误差就是15°，也不证明全部淹没差异由摩阻造成。两半步也未消除该机制，故不再靠不断缩小CFL掩盖问题。

设q=(hu,hv)，固定h和n的摩阻子问题为dq/dt=-g n² |q|q/h^(7/3)，可积分为q_new=q_old/(1+dt g n² |q_old|/h^(7/3))，两分量同乘正数。候选仅实现这个子步。Deng等2013第3.2节式(3.12)给出同式，并讨论保持方向；其整套方法使用Strang分裂和有限体积。这里没有照搬其整体方法、干湿阈值或精度结论，现有DG2调用顺序仍在，因此不能称整体二阶/无条件稳定。[论文原文](https://www.global-sci.com/nmtma/article/view/14029)

备选是保留原对角近似进一步减步，但会增加成本且不满足子步精确方向不变性；另一个备选是完整向量隐式求解，本子问题已有闭式解，不必增加非线性迭代。改变物理糙率、地形、阈值或全域NS都不能针对这个已复现的离散原因，本轮不采用。

## 目标可行性、对照与决定

{note}

已实现：原函数C++复现与候选函数解析一致性、零糙率恒等、固定水深一步/两半步一致、有限耗散、3项旧无摩阻回归；只证明局部摩阻数值性质。两个相同600s案例固定300/600s输出，唯一源码因素是摩阻更新；分别保留CFL .33/.165，不能挑更好得分参数。所有8个原输入文件哈希、所有场和接受步留存。

{table}

上述格数表示两个时间步数值解的分类差别，不能当漏淹/误淹实测数量或现场正确率。1e-6/.001/.01m均是预先登记内部诊断门槛，不能当人员安全判据。原始旧曲线、局部失败、所有不同阈值均保留；最大场差、L1、每一格类别和同CFL修改前后对照见coupled-comparison.json及coupled-full-comparisons.npz。

必须保留的变化：600s在同CFL .33下，h>1mm湿格从252增至338；同CFL .165下从290增至336。扩大湿区可能改变搜救搜索范围，也可能增加误报；缺同时间独立湿/干观测，不能判断这就是现实改善。改前与改后的数值场不相同，不能只报告两个新解更接近而隐去预测范围扩大。

```json
{json.dumps(runs,indent=2)}
```

成本：构建及全部运行时间见candidate-build.json与runs.json；每次600s上限300秒墙钟，串行，失败不自动重启。三个小回归为n=0，仅覆盖无摩阻分支不受影响，不能替代粗糙斜坡稳态与静水平衡回归。局部解析探针包括108+252个已有活动格，不是360份独立物理实验。

状态：局部机制修正已验证；耦合早期敏感性已完成；现场受淹/涨退/主流向/道路/起降点仍未新验收，新增现场通过0。Carlisle绝对起时、HVAR/free、初态/河宽证据缺口仍在。2400s及完整164000s候选未运行，不能把600s场与最大淹没图配对。降雨、负点源、重启、多线程不在本候选验证范围。原apply_friction先更新均值再构造Gauss点的次序及点源分裂误差仍需关注，不随本子步修正自动解决。

下一单元以本轮不利结果决定：如耦合敏感性未普遍改善，先复用600s差点检查摩阻均值/斜率更新与源项分裂，不直接叠加多个改动；如有一致改善，先做有摩阻的静水/均匀坡流独立数值回归，再按固定协议跑后段至2400s验证旧失败区。不是无休止缩CFL或扫糙率。两条路径都不允许据软件相似性宣布现实通过，不启动不可恢复的整夜全事件。资料核验继续独立推进。
'''
(R/'audit-round86-20261030.md').write_text(report,encoding='utf-8')
(R/'literature-source.json').write_text(json.dumps({'title':'Robust a Simulation for Shallow Flows with Friction on Rough Topography','authors':'Deng, Li, Sun, Wu','year':2013,'doi':'10.4208/nmtma.2013.1129nm','url':'https://www.global-sci.com/nmtma/article/view/14029','verified_location':'section3.2 equations3.10-3.12 PDF page9 (printed393)','access_date':'2026-10-30','pdf_saved_locally':False,'claim_scope':'fixed-depth split Manning ODE only; not transfer of whole-scheme stability or field accuracy'},indent=2))
for n in ['diagnose_routing86.py','candidate_friction86.py','analyze_friction86.py','publish_friction86.py']:shutil.copy2(S/n,R/n)
with zipfile.ZipFile(R/'raw-experiments.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for folder in ['controls','cfl033','cfl0165']:
        for p in (R/folder).rglob('*'):
            if p.is_file():z.write(p,p.relative_to(R))
    for p in (R/'candidate-probe').glob('*'):
        if p.is_file():z.write(p,p.relative_to(R))
with zipfile.ZipFile(R/'candidate-source.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for p in (R/'lisflood8-isotropic-friction').rglob('*'):
        if p.is_file():z.write(p,p.relative_to(R/'lisflood8-isotropic-friction'))
manifest=[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(R.iterdir()) if p.is_file() and p.name!='manifest.json']
(R/'manifest.json').write_text(json.dumps(manifest,indent=2));dest=D/'round86-artifacts';dest.mkdir(exist_ok=True)
for p in R.iterdir():
    if p.is_file():shutil.copy2(p,dest/p.name)
shutil.copy2(R/'audit-round86-20261030.md',D/'audit-round86-20261030.md')
update=f'''## 2026-10-30 第86轮：摩阻方向误差复现及隔离修正600s对照收尾

{note}

报告audit-round86-20261030.md，原始场/源码/哈希/局部探针/两组CFL对照全部保存在round86及docs/round86-artifacts。只有复制源的dg2::friction改变，不改生产默认/网页，不自动更新PR。现场通过新增0；短时格数改善不是现实准确率。两组均结束，无后台实验，网页server.py8766保留。

下一步：根据coupled-comparison.json保留全部不利项；不能仅靠局部解析正确就启动整夜长跑。如耦合仍恶化，先定位摩阻均值/斜率顺序和源项分裂；如均改善，先做有摩阻静水/均匀坡流独立回归，再固定参数验证后段2400s。起时/HVAR/free/初态和河宽缺口继续核验，不通过观测调参或改变阈值。

'''
for n in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
    for root in [O,D]:
        p=root/n
        if p.exists():p.write_text(update+p.read_text(encoding='utf-8-sig'),encoding='utf-8')
assert all(hashlib.sha256((dest/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in manifest)
print(json.dumps({'note':note,'files':len(manifest),'dual_hashes_match':True},ensure_ascii=False))
