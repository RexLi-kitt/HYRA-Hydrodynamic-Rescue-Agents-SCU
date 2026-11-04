from pathlib import Path
import json,hashlib,shutil,zipfile
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round88';S=Path(__file__).parent;D=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029\terrain-workbench\docs')
st=json.loads((R/'status.json').read_text());assert st['state']!='running'
co=json.loads((R/'coupled-comparison.json').read_text()) if st.get('internal_gate_pass') else None
note=f"CFL.165候选2400s状态{st['state']}，内部关口{st.get('internal_gate_pass',False)}。"
table='内部门槛未通过，保存失败，不比较残缺窗口。';adverse=[]
if co:
    rows=['|时间s|最大深差旧/新m|L1深差体积旧/新m³|1e-6变化格旧→新|1mm变化格旧→新|1cm变化格旧→新|','|---|---|---|---|---|---|']
    for f in co['frames']:
        a=f['old_cfl_pair'];b=f['new_cfl_pair'];cs=f['sensitivity_change']['changed_cells_before_after']
        rows.append(f"|{f['time_s']}|{a['max_abs_depth_m']:.7g} / {b['max_abs_depth_m']:.7g}|{a['L1_depth_volume_m3']:.7g} / {b['L1_depth_volume_m3']:.7g}|{cs[0][0]}→{cs[0][1]}|{cs[1][0]}→{cs[1][1]}|{cs[2][0]}→{cs[2][1]}|")
        for key in ['max_abs_depth_m','L1_depth_volume_m3']:
            if b[key]>a[key]:adverse.append({'time_s':f['time_s'],'metric':key,'old':a[key],'new':b[key]})
        for k,(x,y) in enumerate(cs):
            if y>x:adverse.append({'time_s':f['time_s'],'metric':'changed_cells','threshold_m':[1e-6,.001,.01][k],'old':x,'new':y})
    table='\n'.join(rows);last=co['frames'][-1];cs=last['sensitivity_change']['changed_cells_before_after']
    note+=f"全8帧配对完成；2400s两个时间步的1mm判断差异{cs[1][0]}→{cs[1][1]}格，1cm差异{cs[2][0]}→{cs[2][1]}格。全窗口预定指标中恶化项{len(adverse)}个，全部保留。软件稳定性不等于现场正确率，默认不升级。"
(R/'all-adverse-comparisons.json').write_text(json.dumps(adverse,indent=2))
report=f'''# 第88轮：解析摩阻候选的2400s双时间步对照及作者资料核验

## 更改日志

2026-10-30。沿用round86隔离候选exe SHA884a074e36474ae86949579525af295a0e9550c24d2ca138cd2612dcd4e7a771，无源码变动。基线候选round87/CFL.33已完成，本轮只补CFL.165到2400s。dx10m、dtmax1、全部源/地形/边界/糙率.06/干初态/测点、300s输出及1e-6/.001/.01内部阈值保持不变；同一公式的旧式配对为round83/85。当前无重启支持，因此从零延长已完成600s细步，早期输出/接受步必须逐位复现，不重复计为独立证据。新增运行/分析/论文审查/发布脚本，原生产默认、网页和PR不动，回退使用原生产版本或保留的旧候选。

## 原因、证据与否决条件

round86的600s结果只能说明初期，不能外推后段。round87粗步完成后，本轮补齐真正的同候选配对，核查有限非负、水量、实际时间步、完整8帧和每个预登记阈值；任何失败停止，不调阈值、不重启。旧式与新式的时间步敏感性分别计算，不能把“新粗步接近旧细步”当成新式稳定。全域L1、最大深差、逐格分类全部比较，不只报有利终点。

## 可行性、结果与限制

{note}

{table}

```json
{json.dumps(st,indent=2)}
```

不利结果全表见all-adverse-comparisons.json，完整新旧4套场及分类在coupled-full-comparisons.npz。任一指标变好都不能当实测精度提升；剩余不同湿格须结合有观测支持的区域尺度评估，不能默认无搜救影响。当前内部毫米阈值不是安全阈值或现实湿区定义。新公式改变预测湿区而未获得现场真假标注；现场新增通过0。

已实现：局部摩阻方向修正、独立有摩阻静水/坡流回归（round87）、本轮全2400s接受步及水量关口和双步比较。未验证：完整164000s、网格收敛、现实淹没/涨退/道路/起降点，以及降雨/抽水/重启/多线程。没有代码变动，不重复round87回归；未恢复HLLC或旧800s/墙位。所有失败保留。运行成本约见status.json；完整事件按短程吞吐只能粗略估算数小时，后期湿区和CFL会变化，不能承诺完成时间。

## 作者原始资料核验（与数值实验独立）

公开作者博士论文已从Bristol下载15,333,752字节，249页，SHA5355c7548df26bc2b67610431a4feb788d60bb2cb2f6bed07da8dd793c1ff283；此前网页工具仅因文件大小无法读取。本地使用pypdf抽取并渲染核查PDF135/139页，没有访问先前403站点或绕过策略。原PDF、索引、逐页文字和图均保存。

核验结论：论文仍称自由出口，与归档HVAR不一致；章4文字45小时而图4.7横轴到70小时，未建立归档UTC；文字8点与图6点也不一致。图4.7的黑线是全分辨率二维模型，不是新增实测曲线。不能据其移时、选边界或补造观测。P100的GPU加速结果属于另一个SGC/非均匀网格实现，不能作为本DG2代码的加速实证。见thesis-evidence-audit.json中的原页码和待补原始文件清单。[公开论文原文件](https://research-information.bris.ac.uk/ws/portalfiles/portal/430852101/Final_Copy_2024_10_15_Rong_Y_T_PhD.pdf)

不改输入，以上核验没有使本案例自动通过。需要归档运行版本对应的出口设置、全部边界绝对时刻/预热和高程基准、初态/糙率赋值、原始测站与有效干湿观测范围；不自行外发联系。

本机只读盘点：i9-14900HX，24核32线程；nvidia-smi识别RTX5060 Laptop、8151MiB显存，驱动573.24。确有可研究的加速硬件，但当前候选仍强制OMP1，回滚/记账/摩阻修改未经GPU移植与一致性核验。未安装或更改驱动，不承诺论文P100加速倍数。硬件证据见hardware88.json，后续先核对已有工具链和限定同配置性能/结果一致性，再决定是否值得移植。

## 下一步

本轮结束时间步扫描，不再为了更好看的格数追加CFL试验。若某些后段指标恶化，先按保存差点定位均值/斜率摩阻调用和源项分裂，不能回避失败叠加修改；若整体有收益，保留隔离候选，优先转同事件输入与有完整条件的独立实测案例。完整现场窗口仍是目标，但不能把未闭合输入、无恢复长跑或软件一致性包装为现实验收。长程启动前需要明确可用成本和可靠保存/恢复方案，不能失败后反复从零跑；不直接改线程数绕过当前OMP1限定。
'''
(R/'audit-round88-20261030.md').write_text(report,encoding='utf-8')
for n in ['extend_friction88.py','analyze_friction88.py','read_thesis88.py','audit_thesis88.py','publish_pair88.py']:shutil.copy2(S/n,R/n)
with zipfile.ZipFile(R/'raw-experiment.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for p in (R/'carlisle_2400_cfl0165').rglob('*'):
        if p.is_file():z.write(p,p.relative_to(R))
manifest=[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(R.iterdir()) if p.is_file() and p.name!='manifest.json']
(R/'manifest.json').write_text(json.dumps(manifest,indent=2));dest=D/'round88-artifacts';dest.mkdir(exist_ok=True)
for p in R.iterdir():
    if p.is_file():shutil.copy2(p,dest/p.name)
shutil.copy2(R/'audit-round88-20261030.md',D/'audit-round88-20261030.md')
update=f'''## 2026-10-30 第88轮：2400s双时间步配对收尾

{note}

候选.33在round87，.165在round88，旧式.33/.165仍round83/85；不重复跑。原始8帧、完整分类、全部不利项和源配置哈希双份round88及docs/round88-artifacts。原生产、网页、PR保持。新论文已本地核查但出口/时窗未闭合，图4.7参考曲线是软件解，不能补作实测；详见thesis-evidence-audit.json。新增现场通过0。

结束无针对性的CFL扫描，先据all-adverse-comparisons.json决定是否需定位后段耦合；否则转输入闭合和独立实测案例。完整长跑须先有成本/保存恢复方案，不重复从零失败长跑或把软件一致率叫现实正确率。当前本轮已退出，无重负载；server.py8766是用户网页服务。

'''
for n in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
    for root in [O,D]:
        p=root/n
        if p.exists():p.write_text(update+p.read_text(encoding='utf-8-sig'),encoding='utf-8')
assert all(hashlib.sha256((dest/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in manifest)
print(json.dumps({'note':note,'files':len(manifest),'dual_hashes_match':True},ensure_ascii=False))
