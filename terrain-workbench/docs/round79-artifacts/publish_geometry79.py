from pathlib import Path
import json,hashlib,ast,shutil,sys
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round79'
P=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029');D=P/'terrain-workbench/docs'
S=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,b):
 if isinstance(b,str):b=b.encode('utf-8')
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==b,p
 else:p.write_bytes(b)
def dump(p,v):save(p,json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

v=json.loads((R/'verification.json').read_text())
assert v['static_pass'] and len(v['actual'])==2 and all(x['passed'] for x in v['actual'].values())
checks={}
for n in ['archive','mean_preserving']:
 c=R/n;inputs=json.loads((c/'inputs.json').read_text());assert all(sha(c/k)==val for k,val in inputs.items())
 fields={}
 for f in (c/'results').glob('check-0000.*'):
  a=np.loadtxt(f,skiprows=6);fields[f.name]={'finite':bool(np.isfinite(a).all()),'min':float(a.min()),'max':float(a.max())}
 assert all(x['finite'] for x in fields.values())
 h=np.loadtxt(c/'results/check-0000.wd',skiprows=6)
 sx=np.loadtxt(c/'results/check-0000.wd1x',skiprows=6);sy=np.loadtxt(c/'results/check-0000.wd1y',skiprows=6)
 checks[n]={'fields':fields,'minimum_reconstructed_depth':float(min((h-np.sqrt(3)*abs(sx)).min(),(h-np.sqrt(3)*abs(sy)).min())),'input_hashes_unchanged':True,'momentum_export':'not available in this unchanged configuration; no complete momentum-field verification claim'}
old=json.loads((O/'round78/geometry-coverage-audit.json').read_text('utf-8'))['all_input_hashes']
assert all(sha(Path(k))==val for k,val in old.items())
assert sha(O/'round77/build-rk-ledger-cpu64/Release/lisflood.exe')==json.loads((R/'protocol.json').read_text())['exe_sha256']
checks['original_inputs_and_exe_unchanged']=True
dump(R/'independent-verification.json',checks)
report='''# 第79轮：保原始高程候选通过限定水动力兼容检查

日期2026-10-30。基线：第78轮河床表示协议、第77轮CPU64独立求解器，Git HEAD 7043b2c5a25e763491bdf4ce29650266407dff1f。原9项跟踪修改保留，生产默认不变。**本轮是候选数值兼容关口，不是现场定性通过；新增现场通过0。**

## 一、更改日志与回退

新增run_geometry79.py：run串行执行；每种新床面通过round78的slopes重新生成坡度，静水初始水深坡度由新床面推导，不能复制不同床面下的水深坡度。先写protocol.json，再跑三组3秒静水；全部通过后才以相同第77轮exe跑归档床面与新床面各60秒。两组物理参数、坐标、源曲线、边界、采样及启动水深不变，只换全域床面三文件；通过输入哈希确认。未改求解公式，不编译新exe，不重装。

参数：10m网格，476×306，fpfric=.06，initial_tstep=1，sim_time=60，CPU64/OMP1；每进程180秒资源超时，失败停止，不循环重启。内部门槛沿用第75/76轮浮点诊断：静水深漂移<1e−10m、体积误差<1e−8m³；实际启动源量/水量账误差<1e−6m³，无非有限或负均值水深、无负RK记录。不是实测精度阈值。candidate bed均值/坡度文件完全复用round78，无调参。

回退：候选只在round79/mean_preserving使用，继续用原归档即可；不删除任何新旧结果。脚本在项目benchmarks和outputs双份，完整输入/结果哈希在manifest.json，报告及结果镜像在docs/round79-artifacts。

## 二、结果、机制与不利信息

三组平衡例：全湿斜床、干湿台阶、平面岸线均完成3秒。最大水深漂移分别2.776e−16、0、1.110e−16m，体积差0、0、−1.776e−15m³，均有限非负、没有负RK记录。它们只验证这几类兼容性，不是任意地形/CFL正性定理。

|同一作者配置0–60秒|归档DG2|保原始高程候选|
|---|---:|---:|
|实际接受步|87|60|
|运行墙钟秒|3.501|2.674|
|接受源体积(m³)|111.0936|111.0936|
|净边界交换(m³)|0|0|
|导出水深积分−源−边界(m³)|−3.00e−10|0|
|最大水深(m)|0.203170|0.328657|
|数值湿润格(h>1e−6m)|40|9|
|负深/非有限/负RK记录|无|无|

两组均完整到60秒，源量与接受时间一一对应，源账积分没有漂移；边界交换从两个RK阶段各半独立求和，点源不重复计入。归档对照最终水深逐位复现第76轮，说明第77轮仅账目修改未影响该启动例。

候选与归档最大水深差0.253588m。原先31格数值湿润在候选变为未湿，候选没有新增数值湿润格。这是**真实计算输出变化，但不是已知改善**：同水量更集中，既可能反映不同的河道约束，也可能暴露新的传播问题。60秒很短且初期来水很小，不能用湿格变少判漏淹修好，或用积水变深判错误；该时刻没有配套现场范围。1e−6仅是内部足迹诊断，不能拿来作道路安全或现场受淹门槛。

第78轮静态通路门槛降低并不保证动态传播更快，这一实算差异否定了“只要降静态障碍就必然改好”的推断。候选仍未恢复实际河宽；1552河道格的原记录宽度超过10m，输水横断面仍有证据缺口。不能为了扩大湿格数改糙率或湿阈值。

独立复核：原始输入、exe及本轮配置前后哈希一致；所有已导出的水深/水面高程及水深坡度有限，重构面深最小值另存independent-verification.json。当前不变配置不导出完整动量场，因此不宣称完成全场动量核验。完整时序、stage、mass、接受步、点源体积和开放边界账目全部保留。

## 三、可行性、后续与否决条件

已完成：一个物理输入表示候选的静水与真实配置启动兼容检查，3组小例+配对60秒，求解总墙钟约6.29秒。没有理由重复本套检查；不能按初期干域速度线性外推整场耗时。

部分实现：河道地形表示候选确实影响水动力，内部启动关口通过。未验证：完整淹没区域、涨退/流向、独立实验回归、现场道路/起降与人员优先级。旧19-J漏淹及Okushiri反向/峰序保持，不能被参考后端启动成功替代。

下一单元直接回到同事件输入与完整区域对照准备，不继续扩大无信息增益的小测试。必须闭合或明确条件化：作者曲线绝对起时、164000秒参数与245700秒曲线覆盖、HVAR高程/自由出口差别、干初态依据。保留所有5份范围与263点及其覆盖/排水机制差异，最大图不能按某瞬时图评分，范围外未知不算干。资料仍缺则先给条件研究场景边界与可回答问题，不能宣称完整Carlisle复现，也不靠移时匹配测站。

采用/否决：目前仅保留候选、不开默认。下一整场须先固定区域/时窗/观察证据和数值湿润可见性规则，并评估计算成本；若候选导致关键漏淹/反向或没有搜救尺度收益，则拒绝这一表示作为升级，转有河宽/断面依据的河道—洪泛耦合或条件完整独立案例。没有证据不全域换NS、不为分数切换出口。

跳过与局限：未重跑旧Merewether800s、墙位或取消HLLC；未跑完整Carlisle、GPU、SGC、非静水/3D；未改模板分类/旧失败。未做UI浏览，原file策略拒绝未绕过。本轮全部进程已完成，没有残留长实验。

主结果目录：C:\\Users\\Lenovo\\Documents\\ChatGPT\\无人机集群搜救大创\\outputs\\round79。主要文件：protocol.json、verification.json、independent-verification.json及两组case.par、inputs.json、results与run.log。源码和配置基线引用第78轮及第77轮已保存的原始来源，不新增外部下载。
'''
entry='''## 2026-10-30 第79轮：床面候选通过静水与配对60秒启动；场分布改变但实测收益未定

最新以round79为准。第78轮保原raw均值/minmod坡度候选，用未改round77 CPU64 exe完成3组3秒平衡和配对60秒。全湿斜床/干湿台阶/平面岸线漂移≤2.78e−16m；归档87步、候选60步，均有限非负、无负RK、入水111.0936m³，独立水量账误差≤3.00e−10m³。归档最终水深逐位等于round76，生产不升级，原9项跟踪修改保留。

关键不利/未知：候选数值湿格40→9、最大水深.20317→.328657m，同水量更集中；原31数值湿格变未湿，没有候选新增湿格。1e−6仅内部诊断，不能称漏淹改善或真实准确率提高；60秒无独立瞬时范围。静态门槛下降不保证动态传播更好，河宽表示仍缺。完整动量场未导出，限制已记录。新增现场通过0，19-J与Okushiri旧失败保持。

本轮已完成，无后台长跑，不重复该套或旧800秒。下一步优先闭合/明确条件化作者epoch、164000/245700时窗、HVAR基准/free差别与初态，准备全部既定区域的完整过程对照；不再扩大小测试，不按分数挑配置。完整对照无收益则拒绝该表示升级、转有河宽/断面证据方向或独立完整案例。协议/全部输入结果与哈希在round79，报告outputs/round79/床面候选水动力兼容检查-20261030.md、docs/audit-round79-20261030.md及结果镜像双份。模板v1.3不变。

历史记录保留如下。

'''
for name in ['run_geometry79.py','publish_geometry79.py']:
 b=(S/name).read_bytes();ast.parse(b);save(R/name,b)
save(P/'flood-platform/benchmarks/run_geometry79.py',(S/'run_geometry79.py').read_bytes())
save(R/'床面候选水动力兼容检查-20261030.md',report);save(D/'audit-round79-20261030.md',report)
dump(R/'next-unit.json',{'task':'Close or explicitly condition event metadata, then preregister whole-event regional comparison; do not repeat startup checks','unresolved':['absolute epoch','164000s run vs245700s curve event coverage','HVAR datum vs paper free boundary','dry initial state basis','real width/bed support','independent dry/route/site evidence'],'candidate_status':'numerical startup gate passed, physical superiority unverified','fallback':'Conditional research scenario clearly labeled or complete independent case if provenance unavailable; no fabricated alignment','no_running_experiment':True})
mirrors=[]
for f in sorted(R.rglob('*')):
 if f.is_file() and 'logs-before' not in f.parts and f.name!='manifest.json':
  g=D/'round79-artifacts'/f.relative_to(R);save(g,f.read_bytes());mirrors.append(g)
logs=[O/'HYRA自动续作进度.md',O/'HYRA累计更改日志.md',O/'HYRA现实贴合证据与改进工作表-20261029.md',D/'HYRA自动续作进度.md',D/'HYRA累计更改日志.md',D/'HYRA现实贴合证据与改进工作表-20261029.md',D/'reality-validation-workplan-20261029.md']
for i,p in enumerate(logs):
 b=p.read_bytes();save(R/'logs-before'/f'{i}-{p.name}',b);p.write_bytes(entry.encode('utf-8')+b)
manifest={str(f.relative_to(R)).replace('\\','/'):sha(f) for f in sorted(R.rglob('*')) if f.is_file() and f.name!='manifest.json'}
dump(R/'manifest.json',manifest)
assert all(sha(R/k)==h for k,h in manifest.items())
assert (R/'床面候选水动力兼容检查-20261030.md').read_bytes()==(D/'audit-round79-20261030.md').read_bytes()
print(json.dumps({'files_hashed':len(manifest),'result_files_mirrored':len(mirrors),'logs_updated':7,'original_hashes_unchanged':True,'all_checks_passed':True,'minimum_face_depths':{n:checks[n]['minimum_reconstructed_depth'] for n in ['archive','mean_preserving']}},ensure_ascii=False))
