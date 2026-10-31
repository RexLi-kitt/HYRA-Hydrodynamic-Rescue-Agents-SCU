from pathlib import Path
import shutil, hashlib, json, ast
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs')
D=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029\terrain-workbench\docs')
R=O/'round91'; A=D/'round91-artifacts'
p=R/'analyze_full91.py'; before=p.read_text(); backup=R/'analyze_full91.before-output-audit.py'
assert not backup.exists(), 'Do not apply preparation twice'
shutil.copy2(p,backup)
s=before.replace("assert sha(O/'round90/observation-footprint-mask.npz')", "assert all(sha(C/n)==h for n,h in p['configuration_hashes'].items()), 'Simulation inputs changed'\nassert sha(O/'round86/build-isotropic-cpu64/Release/lisflood.exe')==p['exe_sha256']\nassert sha(O/'round90/observation-footprint-mask.npz')")
s=s.replace("frames=[]", "mx=read(res/'check.maxb')\nframes=[]")
s=s.replace("'volume_m3':float(h.sum()*25)", "'volume_m3':float(h.sum()*25),'maximum_dominates_saved_depth_above_1mm':bool(np.all(mx[h>.001]+1e-12>=h[h>.001]))")
s=s.replace("'per_step_mass_guard':massgate", "'per_step_mass_guard':massgate,'maximum_dominates_all_snapshots':all(f['maximum_dominates_saved_depth_above_1mm'] for f in frames)")
s=s.replace("model_zero_max_depth=bool(mx[row,col]==0)", "model_zero_recorded_max_depth=bool(mx[row,col]==0),zero_recorded_max_meaning='no recorded depth above 0.001m; not proof of exact dry state'")
s=s.replace("'zero_max_depth_in_reported_fully_covered_cells':", "'zero_recorded_max_depth_meaning':'no recorded depth above 0.001m, not exact dryness','zero_recorded_max_depth_in_reported_fully_covered_cells':")
assert s!=before
ast.parse(s)
p.write_text(s)
for f in [p,backup]:shutil.copy2(f,A/f.name)
shutil.copy2(__file__,R/Path(__file__).name);shutil.copy2(__file__,A/Path(__file__).name)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
entry='''## 2026-10-31 第91轮补充：完整事件运行期间的输出语义核查

完整事件仍由原会话24476运行，未重启或改其依赖。已核实SGC最大水深每实际步累计，六小时保存不会清零；最大水深只累计>1mm，故输出0不能解释为严格干燥。原0.01/0.1/0.3m评分规则、观测范围和物理输入均未改变。既有7200秒五个全场均满足累计最大值覆盖已保存的>1mm水深，0个违例。

整场评分器增加输入/exe哈希与累计最大场支配检查，并修正“零值”标签；旧脚本保留。未读取正在运行场评分、现场新增通过0。待status completed后运行analyze_full91.py并独立审核，不能以中间进度评分。完整原始结果双份归档仍待收尾。

'''
for root in [O,D]:
 for n in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
  f=root/n;f.write_text(entry+f.read_text(encoding='utf8'),encoding='utf8')
report='''# 第91轮补充：最大淹没输出语义与评分准备（2026-10-31）

## 更改日志
基线：正在运行的5m Carlisle完整事件，原配置/源码/exe未改。仅analyze_full91.py增加文件身份和最大场支配检查，将零最大值明确为“未记录超过1mm”，不再暗示严格干燥。旧评分脚本before-output-audit.py保留；无需改变求解器即可回退。全部评分阈值及默认保持。

## 原因
源码SGC2_Inundation_block只在h>DepthThresh=0.001m时更新最大值；mint_hk OFF且massint=1e-9使本配置每步更新。maxint/saveint_max均OFF，定期保存不清零最大值。因此最大场可用于整场最大范围比较，却不能以0证明从未有薄水。已有7200秒五个二进制场对累计最大场检查全部通过；未重跑求解。证据含源码行号与SHA256。

## 可行性
主验收仍是冻结的实测最大范围内>0.1m指示覆盖；另报0.01/0.3m。此审查只保证读取含义，不提高物理精度或增加现场通过项。所有原40点、旧不一致及未知保留。已完成5帧内部回归；完整245700秒结果、全事件守恒和现实对照待运行结束。未做道路/流向/涨退验证，缺相应独立证据。成本是小文件读取与5帧比较，不争抢重计算资源。

下一步：只收尾现有会话24476，核验全时窗、守恒、全部已保存场与累计最大场，输出所有已知湿区漏项及未知，双份归档。失败不循环重启。
'''
for root in [O,D]:(root/'audit-round91-output-semantics-20261031.md').write_text(report,encoding='utf8')
manifest={n:sha(R/n) for n in ['analyze_full91.py','analyze_full91.before-output-audit.py','maximum-output-semantics.json','audit_max_output91.py',Path(__file__).name]}
for root in [R,A]:(root/'scoring-preparation-hashes.json').write_text(json.dumps(manifest,indent=2))
assert all(sha(R/n)==sha(A/n) for n in manifest)
print(json.dumps({'copied_and_verified':len(manifest),'changed_solver':False,'changed_scoring_thresholds':False,'new_field_passes':0}))
