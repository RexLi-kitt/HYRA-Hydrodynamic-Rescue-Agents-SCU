from pathlib import Path
import json,hashlib,shutil,zipfile
S=Path(__file__).parent;O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round90';D=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029\terrain-workbench\docs');A=D/'round90-artifacts'
start=json.loads((R/'startup-analysis.json').read_text());cost=json.loads((R/'cost-analysis.json').read_text());par=json.loads((R/'parallel-status.json').read_text());obs=json.loads((R/'observation-registration.json').read_text())
assert start['gate'] and cost['left_rule_match_within_time_precision'] and par['equivalence']
report=f'''# 第90轮：5米案例入口、观测登记与CPU并行成本关口

## 更改日志
2026-10-31。沿用round86的独立CPU64可执行文件，源码未改，实际走Fast_MainStart加速浅水近似路径；DG2解析摩阻/RK回退不在该路径中，不能继承其成绩。原始5m地形、23点、n=.055、FREE .0006、drycheckoff和sgc_enable不变；仅按第89轮完整数列对应使用Run7.bdy表示2005事件CSV。新增60s两步长、7200s单线程与4线程成本/一致性检查，输出增加binary_out。生产、网页、PR不变。回退是保留原后端与文件，不采用新案例；不删除本次证据。

## 原因与复现依据
源码核实：SGC_enable选择快速路径；无SGC宽度文件时宽度全零、床面复制DEM，无额外亚格河道。H初值清零，未指定startfile；这是归档配置的干初态，不等于现场已证实初态。投影网格的点源按dx换算，23点/三曲线与CSV合计一致。drycheckoff不代表可以忽略负深；保留原开关并检查输出与账目。源码片段和哈希保存在source-path-evidence.json。

60s的dt2/dt1共30/60步。原CSV线性积分191.806759458m³，左端离散入水190.0532007894/190.9299801237m³，与日志水量在文本舍入精度内一致；短少1.7535586686/0.8767793343m³来自原后端一阶源积分，未修饰成精确积分。逐步累计绝对Verror为约6.57e-13/1.13e-12m³；Qerror被极小massint放大，只用有物理意义的Verror核查。

快速ASCII写出硬编码三位小数，不响应output_precision12，因此体积与薄水判断不能拿其舍入误差作求解误差。7200s改为二进制：实际末时7200.369s，保留0.369s越界，不移时假称精确终点。10565步、中位.599s；全部5场有限非负。末水量{cost['binary_volume_m3']:.9f}m³，与独立左端积分差{cost['difference_m3']:.9f}m³，处于日志时标三位小数传播界{cost['reported_time_rounding_bound_m3']:.9f}m³内。相对连续线性积分仍短少{cost['integration_shortfall_m3']:.6f}m³；总净Verror约{cost['sum_signed_internal_mass_residual_m3']:.3g}m³，绝对累计约{cost['sum_abs_internal_mass_residual_m3']:.3g}m³。没有场景正确率结论。

## 可行性与阶段结果
固定同7200s场，OMP1→4的墙钟37.101254→13.979004s，实测{par['speedup_vs_same_case']:.3f}倍；全部5个二进制水深场逐数组相同，实际时间/步长/步数列相同。只对该独立快速路径采用4线程；不外推到GPU、DG2或其他数据。湿区随时间增加，整场线性约8分钟只是早期外推，不能保证。快速路径checkpoint调用被注释，尚无可信恢复；完整事件仅一次有明确墙钟上限的运行，失败保留，不循环从零重启。

独立实测范围已在整场运行前固定：原5个多边形均保留，其中3个与新域相交；{obs['reported_maximum_inundation_fully_covered_cells']}个5m格完全被记录范围覆盖，{obs['edge_intersection_ambiguous_cells']}个边界相交格标几何不确定，另{obs['outside_unknown_cells']}格是未知、不是干地。263调查点全部在新域内且全保留。0.1m为原归档extent.py使用的分析指示量，另预登记0.01/0.3m敏感性；均非人员安全线，图界测量不确定性仍缺。只可给报告湿区内命中/漏项与未知覆盖，缺干区证据时不能报总体正确率或误报率。水迹高程不是水深，最大图不是任意时刻真值，旧失败项不删除。

已完成：入口/初态代码语义、量纲、启动、早期水量、并行一致性、固定观测对象。未完成：完整事件及现实对照，新增现场通过0。下一项以同原始5m输入、4线程、245700s完整窗口、二进制输出及固定观测协议启动一次条件性复现；所有未确定物理初态/观测边界误差保留，不称严格原v6.3.1软件复现。

回归/跳过：无源码变更，不重复旧40点、墙位或DG2长跑。覆盖探针没有读取模型场；原多边形自身已EPSG27700，撤去不必要的投影依赖，未安装环境。两次JSON布尔类型序列化修复只重跑分析，无模拟重启。未开展道路/起降/人员分布验收。所有原始场和配置打包双份，生产默认未升级。
'''
(R/'audit-round90-20261031.md').write_text(report,encoding='utf8')
for n in ['run_kabir90.py','analyze_start90.py','cost_kabir90.py','parallel_kabir90.py','register_observation90.py','finish_audit90.py','publish_path90.py']:shutil.copy2(S/n,R/n)
with zipfile.ZipFile(R/'raw-cases.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
 for n in ['cap2','cap1','cost7200','cost7200_omp4']:
  for p in (R/n).rglob('*'):
   if p.is_file():z.write(p,p.relative_to(R))
A.mkdir(exist_ok=False)
for p in R.iterdir():
 if p.is_file():shutil.copy2(p,A/p.name)
shutil.copy2(R/'audit-round90-20261031.md',D/'audit-round90-20261031.md')
entry='''## 2026-10-31 第90轮：5米入口与完整事件前关口

新5m案例实际为SGC_enable快速加速浅水近似，不调用DG2摩阻候选。无亚格河道、干初态、23单位宽源已核实；60s两步长水量符合原左端源积分。7200s二进制全场有限非负，净水量残差约-5.68e-9m³；源积分与连续曲线仍有78.59m³差，不伪称精确。实际末时7200.369s保留。fast ASCII硬编码三位小数，后续用binary_out。

同7200s的4线程13.98s对1线程37.10s，2.65倍；5个水深场逐数组相同，时间/步长相同。只采用到独立SGC参考，不是算法拟合提升。整场前冻结217464完全覆盖观测格、9318边界未知和354279外部未知，全部263点保留；不能算总体准确率。完整事件尚未启动，下一步一次4线程245700s条件性复现，先查round91状态/进程。快速后端无可信checkpoint，不循环重启失败长跑。报告/原始场/配置/观测协议双份round90和docs/round90-artifacts；现场通过新增0，网页与PR不变。

'''
for root in [O,D]:
 for n in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
  p=root/n;p.write_text(entry+p.read_text(encoding='utf8'),encoding='utf8')
assert all(hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256((A/p.name).read_bytes()).digest() for p in R.iterdir() if p.is_file())
print(json.dumps({'dual_verified':True,'reference_threads':4,'speedup':par['speedup_vs_same_case'],'field_passes_added':0}))
