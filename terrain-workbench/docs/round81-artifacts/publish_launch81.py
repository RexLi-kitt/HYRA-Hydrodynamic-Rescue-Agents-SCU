from pathlib import Path
import json,hashlib,shutil,ast
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round81'
P=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029');D=P/'terrain-workbench/docs';S=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,b):
 if isinstance(b,str):b=b.encode('utf-8')
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==b,p
 else:p.write_bytes(b)
status=json.loads((R/'status.json').read_text());protocol=json.loads((R/'protocol.json').read_text())
for n,v in protocol['inputs'].items():assert sha(R/'archive_conditional'/n)==v
put(R/'launch-snapshot.json',json.dumps({'session_id':4148,'status':status,'snapshot_only':True,'live_state':'status.json and actual process commandline, not this snapshot'},ensure_ascii=False,indent=2))
report=f'''# 第81轮：开始完整作者时窗的条件研究基线

日期2026-10-30。状态为启动登记而非完成结果：快照进程{status['pid']}，包装进程{status['wrapper_pid']}，会话4148，状态{status['state']}，已写出接受步约到{status['last_buffered_accepted_end_s']}秒。实时以round81/status.json、run.log、实际进程和completion.json为准。禁止重复开跑。

## 一、更改日志、参数与回退

基线为第79轮archive短例/第77轮独立CPU64 exe，Git HEAD7043b2c5a25e763491bdf4ce29650266407dff1f。原9项跟踪修改保留。新增run_conditional81.py，仅编排运行/记录/收尾，不修改依赖的求解器。三入口、归档DG2床面、HVAR西出口、原900秒曲线、糙率.06、域/坐标、归档隐含干初态均保留；不运行第78轮新床面候选、不按观测挑选配置。

从已验证60秒扩到作者par指定164000秒；时间步上限仍1秒（作者par原10），不是逐字复现论文。massint从10改300秒、saveint从60改4100秒，预计40份快照；开启qoutput保存动量均值/坡度供核验。输出事件截断接受步，可能影响离散误差，故完整保存实际步长，不能声称只是无影响日志变动。最大水深在本地代码每个接受步后更新，与旧版本手册所述mint_hk语义不能盲套。

CPU64/OMP1，最多一个重任务。包装器每15秒写状态；8小时仅为资源上限，超限/负RK/非零退出均保留失败，无自动重启、无候选队列。源码/原输入哈希冻结在protocol.json，跟踪会话和PID在launch-snapshot.json。回退为停止使用此条件配置；生产默认、旧项目和候选文件不受影响。运行依赖不得中途修改。

## 二、为何现在跑，以及仍未闭合什么

静水/启动/采样关口已完成，继续重复这些测试不能回答区域是否受淹。本轮采用明确的条件研究配置获取长时空场；它既不证明作者输入完全正确，也不等于现场验收。论文3.5写自由出口，而归档用HVAR；本文保留归档并披露差异，不在两者之间按分数择优。[论文原文](https://gmd.copernicus.org/articles/16/3291/2023/index.html)

当前曲线独立审查显示：164000秒Eden单位宽流量仍77.61833m²/s（规则格乘10m后776.1833m³/s），出口水位值仍7.89578m；来水记录延续到245700秒。因此完成164000秒只是完成作者登记时窗，**不是完整退水、更不是保证覆盖现实最大淹没**。不把延长到曲线末端也自动叫完全退水。初始三流量为零，说明归档有启动处理，但不证明真实河道初始全干。

绝对UTC起时、HVAR高程基准/来源、实际初态、河宽表示及论文/归档设置差异继续未闭合。旧文献起时线索不能未经逐曲线来源绑定就套用。本轮不改变观测时间，不对比18无观测站的“现实准确率”。

## 三、可行性、预定检查及后续

已实现：条件配置、完整时窗运行、实际接受步/源体积/RK边界账目与输出。运行中：164000秒场。未验证：全域有限非负/守恒、完整动量场、现场定性吻合。现场新增通过0。

完成后包装器自动核查所有40份格均水深有限非负、动量文件有限、无负RK、结束时间与源体积，并按两RK各半边界通量核水量；容差max(1e−4m³,总接受毛水量×1e−9)，考虑双精度累计和12位小数栅格输出，只是内部账目容差。任何失败保留，不重启。completion.json不表示定性现场通过。

观察对象冻结为第78轮全部5范围1494/20431/25550/26602/4078483及263调查点。域外、部分覆盖、排水过程与79个范围外调查点全部保留；不把外侧当干地。不用未经限制的格内多项式扩出更细边界，只明确用10m格均。未来区域比较仍须登记观测支持的湿润规则/可见范围；现在不选阈值以制造全通过，不合并已判断子集为99%。来源时窗未对应时，比较只能是条件诊断，现场验收维持未通过。

成本：启动时磁盘剩余约132GB，快照与审计记录预计GB以内到数GB量级，湿域增长与CFL可能显著增加时间，不能按60秒干域耗时线性保证完成时间。保守预留数小时，单任务运行。下一触发先核实际进程与status，结束读completion和全场；有进程无推进先诊断，失败不循环重启。并行仅做不改依赖的来源核查/观察协议准备，不开第二重任务。

回归：无需重跑旧800秒、墙位、HLLC或已过的读取套件。原配置输入哈希核对；原候选暂不升级；19-J和Okushiri旧失败保留。UI未预览，浏览器限制未绕过。报告镜像与协议/启动快照双份保存；运行中的原始结果暂留唯一写入目录，完成后再冻结哈希和双份成果，避免把写到一半的文件当完成证据。
'''
put(R/'条件研究完整时窗启动-20261030.md',report);put(D/'audit-round81-20261030.md',report)
for n in ['run_conditional81.py','publish_launch81.py']:
 b=(S/n).read_bytes();ast.parse(b);put(R/n,b)
put(P/'flood-platform/benchmarks/run_conditional81.py',(S/'run_conditional81.py').read_bytes())
fixed=['protocol.json','curve-window-diagnostics.json','launch-snapshot.json','条件研究完整时窗启动-20261030.md','run_conditional81.py','publish_launch81.py']
for n in fixed:put(D/'round81-artifacts'/n,(R/n).read_bytes())
entry=f'''## 2026-10-30 第81轮：条件研究基线已启动，目标164000秒，勿重复运行

会话4148，PID{status['pid']}、包装PID{status['wrapper_pid']}；实际状态以round81/status.json及命令行核验，不按旧77提示重做。只跑archive_conditional一个重任务，CPU64/OMP1、round77不变exe。完整作者时窗0–164000s，归档床面/HVAR/3源/糙率.06及干初态保留；dt上限1、mass300、save4100并输出动量。不是与论文/原par逐字一致的复现；源码依赖运行中勿改。8h资源上限，失败退出保存，不自动重启。

原始数据在round81/archive_conditional，协议/哈希/会话快照/报告已双份。164000秒Eden仍776.183m³/s、出口7.89578m，曲线到245700，因此本目标不是完全退水或保证现实最大范围。绝对起时、HVAR基准/free差别、初态、河宽仍未闭合；明确条件研究，不声称现场验收或99%。全部5范围263点保留，范围外未知，旧失败不改。

下一触发先收完成结果：包装器完成会写completion.json（全40快照均值有限非负、动量、源积分/边界账目），它仅内部关口；现场比较仍未评分。若还在运行，推进不改依赖的元数据/观察协议，禁止另一重任务；失败先诊断不循环重启。当前生产默认未变，原9项修改保留，现场通过仍0。报告round81/条件研究完整时窗启动-20261030.md与docs/audit-round81-20261030.md。

'''
logs=[O/'HYRA自动续作进度.md',O/'HYRA累计更改日志.md',O/'HYRA现实贴合证据与改进工作表-20261029.md',D/'HYRA自动续作进度.md',D/'HYRA累计更改日志.md',D/'HYRA现实贴合证据与改进工作表-20261029.md',D/'reality-validation-workplan-20261029.md']
for i,p in enumerate(logs):
 b=p.read_bytes();put(R/'logs-before'/f'{i}-{p.name}',b);p.write_bytes(entry.encode('utf-8')+b)
manifest={n:sha(R/n) for n in fixed};put(R/'launch-manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
assert all(sha(R/n)==v for n,v in manifest.items())
print(json.dumps({'launch_files_verified':len(manifest),'status_snapshot':status,'report_dual':True,'live_results_not_frozen':True},ensure_ascii=False))
