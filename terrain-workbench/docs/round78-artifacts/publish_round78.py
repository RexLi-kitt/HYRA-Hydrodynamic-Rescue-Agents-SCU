"""Publish geometry-only findings without altering solver/input archives."""
from pathlib import Path
import ast, hashlib, json, shutil

BASE=Path(r'C:\Users\Lenovo\.codex\visualizations\2026\10\23\01a16912-ff90-73e3-a2db-13d580b278d3')
OUT=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
REPO=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029')
DOC=REPO/'terrain-workbench/docs'
R=OUT/'round78'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def new(p,b):
    if isinstance(b,str):b=b.encode('utf-8')
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if p.read_bytes()!=b:raise RuntimeError('Refusing to overwrite different bytes: '+str(p))
    else:p.write_bytes(b)
def js(p,v):new(p,json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

report='''# 第78轮：Carlisle河床表示与观测覆盖审查

日期：2026-10-30。阶段：Q2输入核验、Q4候选准备；定性模板v1.3不变。本轮找到一个与区域受淹有关的几何机制，准备了可回退候选，但**没有新增现场定性通过，也没有证明模型精度提高**。没有运行新的水动力实验或修改生产求解器。

## 一、更改日志、基线与回退

基线Git HEAD为7043b2c5a25e763491bdf4ce29650266407dff1f，既有9项跟踪修改全部保留。几何基线是round73/original/Carlisle的作者2023归档10m输入；代码解释基线为round77/lisflood8-rk-ledger。不是T07原5m复现。此前失败、旧40点和完整时序不变。

|新增文件/函数|此前|本轮行为|
|---|---|---|
|audit_carlisle_geometry78.py：grid、faces、bottleneck、gpkg_native、main|尚未全河网核对归档转换与观测覆盖|原坐标读全1579河道格、全部3入口、5个范围及263调查点，复现DG2转换和运行时边界坡度，保存全量结果|
|prepare_meanpreserving_bed78.py：mm、slopes|仅有归档DG2均值/坡度|另建mean-preserving-geometry，保留原raw格值为Z0，全域minmod重建坡度，不只改失败位置|
|verify_geometry_paths78.py：connected_at|单一最小最大路径算法|另用阈值BFS核验3入口×3表示，在阈值处连通、略低时不连通|

三个脚本在outputs/round78及项目flood-platform/benchmarks双份保存。原数据与求解源码哈希在geometry-coverage-audit.json；候选配置在mean-preserving-geometry/protocol.json，文件哈希在verification.json。两份科学图已检查，数据与图的镜像在docs/round78-artifacts。

候选无拟合参数：Z0=原始栅格值，Z1x=minmod(Zi−Zi-1,Zi+1−Zi)/(2√3)，Z1y=minmod(Znorth−Zi,Zi−Zsouth)/(2√3)，相应周边坡度置零与现代码一致。物理坐标、域、10m网格、源与曲线、糙率及观测不变。原栅格值不自动等于真实面积平均高程；该解释仍是待验证候选。生产默认不启用；回退只需继续使用归档DG2三文件，无须删除任何候选或旧结果。

## 二、复现证据、原因与取舍

### 2.1 全河网几何差异

原raw与CarlislefixDEMwithbedDG2.asc.raw逐字相同。读实际generateDG2DEM.cpp，按当前格及东、南、东南四值作角点转换，复现归档均值误差0、两坡度最大误差6.94e−18。证明了转换来源，不等于证明作者本来把raw定义为格中心或节点，也不能据此移动半格。

按width>0固定河道掩膜共1579格，四/八邻域各只有一个连通分量，全部连到西侧唯一河道边界格。DG2均值减raw的中位数1.65725m、最大7.3265m、最小0；相邻河道面最高床面差的中位数1.8295m，范围−0.8015至5.949m，不能只报告均值抬高而忽略局部降低。

下表为限定河道掩膜四邻接图上，从西出口到入口所需跨越的最低最高面高程。包含现代码实际周边坡度置零。它只是静态连通诊断，**不是实测水位误差、洪水到达时间、过流能力或完整洪泛流路**；真实流动也可能离开河道掩膜。

|原入口|raw分片常量(m)|归档DG2(m)|候选(m)|候选−归档(m)|
|---|---:|---:|---:|---:|
|Eden upstream1|10.3190|12.6110|10.3190|−2.2920|
|Petteril upstream2|18.0390|20.0195|18.0390|−1.9805|
|Caldew upstream3|13.5180|16.5725|13.5180|−3.0545|

BFS独立算法对上述9个阈值检查均通过。候选均值逐字等于raw、坡度有限、内点线性平面精确、每个面值不越过中心与对应邻格极值。通过的是几何门槛，不是水动力正性/静水平衡或现实验收。

物理理由：河道与洪泛区之间的床面和连接会影响分流与外溢，值得先隔离输入表示。Rong等2023论文3.5同样讨论Carlisle中河床平滑与模型表现，支持检查该机制，但论文结论不能直接当本候选成果。[原论文](https://gmd.copernicus.org/articles/16/3291/2023/index.html)

Esri格式将XLLCORNER/YLLCORNER定义为左下格左下角；该规范不能独自证明这个作者预处理文件的采样意图。因此维持原坐标，只登记另一种床面表示，不宣称发现了确定的5m错位。[官方格式说明](https://doc.esri.com/en/arcgis-pro/latest/tool-reference/conversion/ascii-to-raster.html)

### 2.2 河宽、SGC和出口仍有缺口

1552/1579河道格记录河宽>10m，中位43.2589m，最大85.35029m。本候选没有恢复真实横断面或河宽，降低静态障碍不意味着已恢复正确输水能力。不能靠猜测拓宽河道。

实际本地SGC_UpdateH对规则格QVAR乘dx和dt，证实单位宽输入语义与此DG2配置相容；这不证明本地SGC就是论文改进版本。归档SGCbank与DEM都指raw，raw与独立bed只差约0.0005m；代入本地CalcSGCz后全部河道bankfull深度约0.01m。宽河格有专用体积处理，故这里是输入/版本兼容性疑点，不能直接定为SGC物理错误，更不能当已复现论文。暂不另开SGC长跑。

HVAR有274条，0–245700s，值3.456–8.992m，跨过西侧最低raw床面5.76m。因此不能说指定水位始终在床面以下、可与自由出口等同。归档HVAR与论文自由出口、下游高程基准、绝对起时、干初态依据仍未闭合，不能按更合评分的配置择优。原参数164000s与完整曲线245700s的事件时窗关系也须登记清楚。

### 2.3 观测覆盖，不作事后删项

固定域为476×306格，E338490–343250、N554700–557760，原生BNG/EPSG:27700，不移动调查点。全部263点都在模型域内；此前79点在记录淹没边界之外是另一种关系，两者不能混为一谈。145条G;Ed保持原属性。

|原始淹没范围|覆盖关系|模型域内面积(m²)|保留的限制|
|---|---|---:|---|
|1494 Low Crosby|部分|293244.585|大部分域外，不能算干区或判通过|
|20431 Dalston|域外|0|保留为范围未覆盖|
|25550 Harraby Green|域外|0|保留为范围未覆盖|
|26602 Carlisle drainage|域内|585.124|原证据是排水/地表水，单纯河流源可能缺过程，不能删除|
|4078483 2005 Carlisle Event|部分|5259676.685|域外1252506.722m²仍保留|

这些是事件最大记录范围，不能作某时刻图，也不能把边界外白区当独立干地证据。all-outline-coverage.json保存全部原面积、相交面积、域外面积与来源；所有项均未获得模型对照通过。域内相交只作覆盖诊断，未变更原验收范围。

## 三、目标可行性、成本与后续门槛

|目标/状态|本轮可用证据|尚缺/预定判别|
|---|---|---|
|几何机制：已完成静态审查|全河网、全入口、运行时面高程、两种独立图算法|不能替代动态输水或实测|
|候选：部分实现|三份新床面文件与有限/线性/界面约束|仍须静水、有限非负、动态守恒兼容检查；失败即拒绝，不调糙率补救|
|区域受淹：待验证|5份原范围及263点定位完整|需要同事件完整结果、输入时窗与观测适用范围；保持所有未覆盖和已知失败|
|涨退、流向：已有旧限定实验支持|旧Okushiri与Merewether全部标签原样保留|本候选无新证据，不继承旧成绩|
|道路、起降、人员优先级：未验证|无新增现场证据|需要明确时间和位置的独立记录，不能从速度点推断|

成本：轻量数组/图搜索与覆盖审查，主审查脚本实测1.016s（不含资料查阅、绘图与归档）；无新长跑、无GPU/NS投入。GIS使用现有独立环境，不重装。

下一步按next-candidate-protocol.json执行：只改变地形表示，先做限定兼容关口；同时闭合作者数据时间、出口与初态。若有数值失败，保留失败并拒绝或定位机制；若静态/动态改善不能改变目标区域判断，结束此分支，转有断面/河宽证据的河道—洪泛耦合或条件更完整的独立案例。输入证据仍缺时，可明确标记条件研究场景，不能包装成验证通过；不无限追加无关小测试。完整区域对照须事前冻结规则，保留原范围及未知，不用事后缩域或调阈值全通过。

回归与跳过：原文件哈希全部未变；3脚本语法检查；几何转换、面界限与BFS校验通过；两张科学PNG已查看。未重跑旧800s、墙位、HLLC或第77轮无变化的账目套件；未做新候选水动力、独立案例/现场回归、UI预览。求解源码不变无需重新编译。内置浏览器原file策略限制未绕过。

定性前后：本轮新增现场通过0；Merewether19-J漏淹、Okushiri反向/峰序错误仍保留。新几何表示尚不能称真实水情改善，也没有整体99%正确率。生产默认和既有结果不变。

原始成果主目录：C:\\Users\\Lenovo\\Documents\\ChatGPT\\无人机集群搜救大创\\outputs\\round78。报告镜像：terrain-workbench/docs/audit-round78-20261030.md。全量原结果/协议/来源哈希/科学图镜像在docs/round78-artifacts；文件清单manifest.json用于逐文件复核。
'''

protocol={
 'status':'next unit only; no hydrodynamic run started in round78',
 'candidate':'round78/mean-preserving-geometry, raw cell values retained as means, full-domain minmod slopes; not certified physical truth',
 'hypothesis':'The archive corner averaging increases static channel barriers; preserving cell values may change source propagation and river/floodplain allocation without fitting measurements.',
 'solver':'round77/build-rk-ledger-cpu64/Release/lisflood.exe, unchanged CPU64 OMP1',
 'must_freeze_before_run':['executable and baseline/candidate input hashes','identical source locations, hydrographs, roughness, boundary configuration and duration','static-water and dry/partly-wet compatibility tolerances with numerical rationale','actual accepted steps, complete depth/momentum fields, source integrals and RK boundary ledger','whole-event observation objects/time/qualitative rules before inspecting new full-event output'],
 'minimum_next_experiment':['Static-water compatibility over sloping/discontinuous beds with exactly corresponding initial depth/slope fields; do not merely copy a startfile for a different DEM.','If the numerical gate passes, one paired author-configuration short run varying only bed representation; reuse archived unchanged baseline when hashes establish equivalence. The new candidate is not an identical old rerun.','Save wet footprint/flow paths and input-volume ledger, finite/nonnegative and wall-clock cost. Short-run support is computational only, not field validation.'],
 'internal_acceptance':'No NaN/Inf or negative mean depth, no unexplained water-volume creation/loss; register floating-point-scaled tolerances before run and fail closed. No post-result threshold tuning.',
 'field_acceptance':'All preregistered observation-supported regional/phase items must agree. Unknown/outside/partial/disagree remain unpassed; maximum extent not an instantaneous map. Preserve drainage-related outline even if its mechanism is absent.',
 'evidence_priority':['absolute epoch and relation of 164000s run to 245700s boundary coverage','HVAR datum/initial-state provenance and archive vs paper free-outlet difference','raw cell/node intent and physical channel width/bed compatibility','independent dry-area/road/site evidence'],
 'cost_limit':'No heavy run until short compatibility gate and source scope are recorded. Estimate full-run cost from accepted steps; at most one heavy process. Do not introduce GPU/NS without demonstrated need.',
 'reject_or_redirect':['Reject candidate on static-water/conservation/finite/nonnegative failure; diagnose once from retained output, no looped restart.','If no rescue-scale benefit on full fixed scope, retire this geometry route, retain bad cases, consider width-aware conservative channel/floodplain treatment only with provenance.','If author metadata cannot be closed, label an explicit conditional research scenario or use a more complete independent experiment; no claim of validated Carlisle reconstruction.'],
 'do_not':['move observation or source points','select outlet or friction by score','restore canceled HLLC queue','rerun unchanged Merewether or walls','claim that lower static barriers imply accurate flooding'],
 'production_adopted':False,'field_passes_added':0}

entry='''## 2026-10-30 第78轮：河床表示机制已复现，隔离几何候选已备；现场通过仍0

本轮未跑水动力、未改生产默认，原9项跟踪修改及旧结果保留。round78保存全河网/原始覆盖核验与候选；模板v1.3不变，不能称精度提升。无后台长跑。

作者10m raw→归档DG2四角转换逐值复现（均值误差0，坡度≤6.94e−18）。1579河道格全部四邻接连到西侧；DG2均值相对raw中位升1.65725m，面障碍差中位1.8295m且含局部降低。三个源的限定河道静态连通门槛raw/DG2分别10.319/12.611、18.039/20.0195、13.518/16.5725m；不是实际水位误差或输水能力。

新候选mean-preserving-geometry全域保留raw格值、minmod重建坡度，不移点、不为坏点改地形。均值/有限/线性/面界限检查通过；三个静态门槛回到raw，独立BFS9项核验通过。尚无水动力兼容或现场收益证据，不采用生产。1552河道格宽度超过dx，本候选未恢复河宽；本地SGC单位宽QVAR语义已核，归档bank/bed搭配在本地代码触发0.01m最小bankfull，宽河处理与论文改进版本仍需核验，不能直接判SGC错或开长跑。

263调查点全部在模型域内；旧79点在淹没记录边界外是不同概念，全部保留。5范围=1域内(排水源)、2部分、2域外；不删不移，白区未知，最大范围非瞬时。HVAR3.456–8.992m跨过西床5.76m，不可当始终干或等价自由出口。绝对起时/出口基准/初态仍未闭合。没有新现场通过，19-J等旧失败保持。

下一步严格按round78/next-candidate-protocol.json：准备这一个床面候选的静水/动态兼容关口，再按固定全对象做区域淹没；不重复旧墙/800s/账目套件。失败则拒绝或有依据归因；全程无救援尺度收益则结束分支，转有河宽/断面证据的方向；资料不足明确条件研究范围或选择完整独立案例，不能无限小测试。报告outputs/round78/河床表示与观测覆盖审查-20261030.md及docs/audit-round78-20261030.md，代码/全部结果/图/哈希双份可审查。

历史记录保留如下。

'''

if __name__=='__main__':
    a=json.loads((R/'geometry-coverage-audit.json').read_text('utf-8'))
    for p,v in a['all_input_hashes'].items():assert sha(Path(p))==v,p
    for name in ['audit_carlisle_geometry78.py','prepare_meanpreserving_bed78.py','verify_geometry_paths78.py']:
        b=(BASE/name).read_bytes();ast.parse(b)
        new(R/name,b);new(REPO/'flood-platform/benchmarks'/name,b)
    new(R/'publish_round78.py',Path(__file__).read_bytes())
    new(R/'河床表示与观测覆盖审查-20261030.md',report)
    new(DOC/'audit-round78-20261030.md',report)
    js(R/'next-candidate-protocol.json',protocol)
    js(R/'source-notes.json',{
      'accessed':'2026-10-30',
      'paper':{'url':'https://gmd.copernicus.org/articles/16/3291/2023/index.html','authors':'Rong, Bates, Neal','doi':'10.5194/gmd-16-3291-2023','use':'section3.5 motivates channel-geometry audit; no inheritance of published performance'},
      'format':{'url':'https://doc.esri.com/en/arcgis-pro/latest/tool-reference/conversion/ascii-to-raster.html','use':'lower-left-cell corner header convention only; not proof of author sampling intent'},
      'local_primary_inputs':'round73/original/Carlisle (already obtained author archive); round68 observation originals, unchanged',
      'local_code':'round77/lisflood8-rk-ledger: preprocess/generateDG2DEM.cpp, swe/dg2/fields.cpp and modifiedvars.cpp, sgc.cpp and input.cpp; hashes in geometry-coverage-audit.json',
      'no_external_message':True})
    # Mirror compact round results, including numerical arrays, not old input archives.
    mirrors=[]
    for p in sorted(R.rglob('*')):
        if not p.is_file() or p.name in {'manifest.json','publication-verification.json'} or 'logs-before' in p.parts:continue
        d=DOC/'round78-artifacts'/p.relative_to(R);new(d,p.read_bytes());mirrors.append((p,d))
    logs=[OUT/'HYRA自动续作进度.md',OUT/'HYRA累计更改日志.md',OUT/'HYRA现实贴合证据与改进工作表-20261029.md',DOC/'HYRA自动续作进度.md',DOC/'HYRA累计更改日志.md',DOC/'HYRA现实贴合证据与改进工作表-20261029.md',DOC/'reality-validation-workplan-20261029.md']
    for i,p in enumerate(logs):
        old=p.read_bytes()
        if old.startswith(entry.encode('utf-8')):continue
        new(R/'logs-before'/f'{i}-{p.name}',old)
        p.write_bytes(entry.encode('utf-8')+old)
    manifest={str(p.relative_to(R)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(R.rglob('*')) if p.is_file() and p.name not in {'manifest.json','publication-verification.json'}}
    js(R/'manifest.json',manifest);new(DOC/'round78-artifacts/manifest.json',(R/'manifest.json').read_bytes())
    assert all(sha(p)==sha(d) for p,d in mirrors)
    assert (R/'河床表示与观测覆盖审查-20261030.md').read_bytes()==(DOC/'audit-round78-20261030.md').read_bytes()
    for p,v in a['all_input_hashes'].items():assert sha(Path(p))==v,p
    result={'round':78,'manifest_files':len(manifest),'mirrored_result_files':len(mirrors),'manifest_verified':all(sha(R/k)==v['sha256'] for k,v in manifest.items()),'mirrors_byte_identical':True,'original_hashes_unchanged':True,'syntax_checks':3,'logs_updated':len(logs),'production_adopted':False,'new_hydrodynamic_runs':0,'field_passes_added':0}
    js(R/'publication-verification.json',result);new(DOC/'round78-artifacts/publication-verification.json',(R/'publication-verification.json').read_bytes())
    print(json.dumps(result,ensure_ascii=False))
