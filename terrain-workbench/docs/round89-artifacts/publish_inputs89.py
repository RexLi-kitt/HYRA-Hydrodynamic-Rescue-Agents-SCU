from pathlib import Path
import json,hashlib,shutil,datetime
S=Path(__file__).parent;O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round89'
D=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029\terrain-workbench\docs');A=D/'round89-artifacts'
audit=json.loads((R/'input-audit.json').read_text(encoding='utf8'));fp=json.loads((R/'all-run-fingerprints.json').read_text())
matches=[{'bdy':x['bdy_run'],'csv':x['csv_run']} for x in fp['rows'] if x['all_three_match']]
assert {'bdy':7,'csv':1} in matches and {'bdy':1,'csv':7} in matches
manifest=json.loads((R/'download-manifest.json').read_text());tree=json.loads((R/'repository-tree.json').read_text(encoding='utf-8-sig'))
for x in manifest:
 p=R/'original'/Path(x['path']).name;data=p.read_bytes()
 assert hashlib.sha256(data).hexdigest()==x['sha256']
 assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==x['git_blob']
contract={'case_id':'Carlisle2005-Kabir2020-5m-independent-input-candidate','status':'input identity audited; not simulated or field accepted','author_revision':tree['sha'],'dem':'original/Carlisle_5m.asc','boundary_locations':'original/carlisle.bci','event_total_flow':'original/Upstream_Flows_Run1.csv','matching_unit_width_boundary':'original/carlisle_run7.bdy','original_mislabeled_boundary_preserved':'original/carlisle_run1.bdy','mapping_basis':'274 original times and all 3x274 samples match within export rounding 1e-5 m3/s, using 11/6/6 points at 5m; no time shift or fitted scaling','date_origin_source':'Kabir2020 Figure4: 2005-01-07 00:00 for 2005 hydrographs; retain civil-time label and confirm timezone before UTC gauge overlay','grid':audit['grid'],'duration_s':245700,'friction_n':.055,'outlet':'W 554000 558000 FREE 0.0006','not_the_rong10m_or_T07_reproduction':True,'observations':'existing round68 independent maximum extents and survey points; maximum-event comparison only after completed event and valid coverage check; no dry assumption outside surveyed extent','remaining_gates':['verify acceleration + sgc_enable point-source semantics in chosen existing backend; no inherited DG2 scores','verify DEM represents same event and channel/initial-state handling; retain authoritative flags, do not silently change drycheckoff','freeze spatial observation objects and unknown/coverage rules before running','measure bounded runtime/memory and checkpoint support before full event; no from-zero overnight loop','derive UTC mapping only with source evidence; 18Points files are locations not time series'],'next_minimum_test':'static unit/path audit, then bounded inlet/finite/nonnegative/mass probe only if original flag semantics are understood','field_passes_added':0,'production_changed':False}
(R/'new-case-contract.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2),encoding='utf8')
notes={'primary_sources':[{'url':'https://arxiv.org/pdf/2006.11555v2','locations':'Figure4 p17; section3.1.2 pp14-15; code availability p40','summary':'2005 event origin Jan7 midnight; 15-minute routed inflows; CNN uses hydraulic-model targets. Not independent flood-accuracy evidence.'},{'url':'https://github.com/SRKabir/Rapid_FloodModelling_CNN/tree/'+tree['sha'],'locations':'README and Data','summary':'README assigns Run1 to 2005 and Run2-9 to synthetic training; fixed version data identity resolves bdy1/7 mismatch.'}],'read_failures':['Heriot-Watt PDF web403: not retried through alternate access to that endpoint; separately published author arXiv copy read normally','arXiv HTML404, PDF read succeeds; screenshot cache miss, no visual table claim used','one TLS EOF downloading two final small files; one resume with default TLS and all old blobs reverified succeeded'],'simulation_started':False,'license_note':'Public access is not a blanket redistribution license. Local source provenance copies only; no upload or PR.'}
(R/'source-notes.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2),encoding='utf8')
report='''# 第89轮：公开5米Carlisle案例与来水文件错配核验

## 更改日志
2026-10-31（本地时间；触发UTC为2026-10-30）。新增只读获取/核验脚本及独立案例登记，生产求解器、网页、PR、旧10米案例均未变。取得作者仓库固定提交eadfc848a6d37cb3476da2fe6ec7087262d9ef54的28个文件，全部长度、Git blob SHA1及SHA256核验。回退无需修改旧项目：不采用新登记即可。原文件逐字保留，不覆盖或改名伪装原始发布。

## 原因与可复现证据
旧10米案例的出口与绝对时间还未闭合，停止重复CFL扫描，转向作者公开输入。新地形为951×611、5m、原点338500/554700；无NoData、有效值有限，23个源点均落在有效且不同的格内。配套PRJ为英国国家格网。新域不同于旧10m域，不能以更换输入后的差别宣称单因素算法提升。

按原BCI统计11/6/6个点，单位宽QVAR分别乘55/30/30m，与CSV原时间逐行核对。Run1.bdy与Run1.csv三条曲线分别274/273/274行不一致；主源峰值854.2856与1272.994m³/s。全9×9对应检查发现Run1.bdy严格对应Run7.csv，Run7.bdy严格对应Run1.csv（每组3×274原时刻；容差1e-5m³/s仅为文本小数舍入）。因此依据README对2005事件Run1的定义，可登记Run7.bdy为该CSV的单位宽表达；不按观测拟合、移时或缩放。Run6未找到三条同时一致的CSV，完整81组结果保留，不能说整个仓库全部核验通过。

作者论文[arXiv版本](https://arxiv.org/pdf/2006.11555v2)图4给出2005曲线从1月7日00:00开始；该日期只适用于此事件资料，不能追溯套到旧归档。原运行245700s覆盖274个900s样本，FREE .0006、n=.055；这些与旧10m配置有差异。论文神经网络准确率主要是对软件输出，不继承为本项目实测正确率。原Run1配置的acceleration、sgc_enable、drycheckoff仍需结合实际后端核实，不直接启动重负载。

## 可行性、阶段状态与后续
已实现：公开5米地形获取、全部下载哈希、23源定位、9×9无移时数列对应及新案例输入登记。尚未实现：新案例求解、实际淹没/涨退/道路对照；新增现场通过0，不能报拟合度提高。

按顺序执行：①核实既有后端acceleration/sgc_enable入口、单位和初态，不安装新环境；②登记观测有效区域及最大范围比较规则，未知区不当干地；③若输入及数值开关可解释，做有成本上限的入口/守恒探针；④确认时间步成本和保存恢复方案后才完整事件。任何数值失败先定位，不自动重复跑；输入仍矛盾则保留未知并请求所缺原件。拟议目标为受淹区域定性证据，不追逐旧40点5%。

回归与跳过：本轮无算法改动，未重复水动力回归或旧2400s，未改阈值和糙率；已做数据完整性、网格、源落格及全部数列一致性检查。没有将样点位置文件当观测过程。文件公开不等于无限再分发许可，本轮仅本地保存，不推送。一次TLS断连后仅补缺文件一次，标准TLS不变；失败记录见source-notes.json。
'''
(R/'audit-round89-20261031.md').write_text(report,encoding='utf8')
for name in ['acquire_kabir89.py','audit_kabir89.py','fingerprint_kabir89.py','publish_inputs89.py']:shutil.copy2(S/name,R/name)
A.mkdir(exist_ok=False)
for p in R.rglob('*'):
 if p.is_file():dest=A/p.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
shutil.copy2(R/'audit-round89-20261031.md',D/'audit-round89-20261031.md')
entry='''## 2026-10-31 第89轮：5米独立输入来源与Run1/7错配闭合

取得Kabir2020作者固定仓库5m地形及28文件，全部Git blob与SHA256通过；951×611、23源格有效。Run1.bdy与2005事件Run1.csv不一致，完整9×9核验发现Run7.bdy对应Run1.csv、Run1.bdy对应Run7.csv（全部3×274原时刻，无移时/拟合）。原数据保留，Run6仍无完整对应。新案例登记outputs/round89/new-case-contract.json；不替换旧10m、不继承旧评分。

新源运行245700s、FREE .0006、n=.055；论文2005起时Jan7 00:00仅为本案例日期线索，UTC叠加仍核时区。先核实原acceleration/sgc_enable/drycheckoff的后端语义和初态，再冻结观测覆盖及成本/恢复门槛；不要立即开整夜长跑。现场新增通过0、无算法改变、无后台实验，网页服务保留。报告及全原始文件双份round89与docs/round89-artifacts；不再说所有可用地形只有10m/50m，但这不是T07原件复现。

'''
for root in [O,D]:
 for name in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
  p=root/name;old=p.read_text(encoding='utf8');p.write_text(entry+old,encoding='utf8')
checks=[]
for p in R.rglob('*'):
 if p.is_file():checks.append(hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256((A/p.relative_to(R)).read_bytes()).digest())
assert all(checks)
print(json.dumps({'dual_files_verified':len(checks),'source_files':len(manifest),'run_identity_matches':matches,'field_passes_added':0}))
