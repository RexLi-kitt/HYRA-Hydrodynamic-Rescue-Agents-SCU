from pathlib import Path
import json,hashlib,sys,shutil,ast
import numpy as np
from dg2_depth_reader import DepthReader
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round80'
P=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029');D=P/'terrain-workbench/docs';S=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
R.mkdir(exist_ok=False)
dump(R/'protocol.json',dict(hypothesis='Negative raw terminal polynomial does not necessarily affect current mean-value station outputs; audit consumers before changing solver.',default='cell_mean with explicit support and raw polynomial flag',rejection='raw_polynomial rejects entire cell if any corner negative; no clipping or observed wetness inference',tests=['all original263 coordinates retained','all negative-polynomial cells reject raw mode','all263 default samples exactly mean values','finite/shape/negative mean/outside/header checks'],scope='offline reference-backend reader only, not wired into production UI, no solver runs',field_passes_added=0))
marks=json.loads((O/'round68/prepared/all-survey-marks.json').read_text('utf-8'));result={}
for name in ['archive','mean_preserving']:
 path=O/'round79'/name/'results/check-0000.wd';rd=DepthReader.read(path);records=[]
 for mark in marks:
  a=mark['source_attributes'];x,y=a['E'],a['N'];s=rd.sample(x,y)
  assert s['depth_m']==rd.h[s['row'],s['col']]
  records.append(dict(source_attributes=a,sample=s,simulation_time_s=60,comparison='unscored_peak_observation_not_instantaneous'))
 bad=np.argwhere(rd.corner_min<0);rejected=0
 for j,i in bad:
  x=rd.header['xllcorner']+(i+.5)*10;y=rd.header['yllcorner']+(306-j-.5)*10
  try:rd.sample(x,y,'raw_polynomial')
  except ValueError:rejected+=1
 assert rejected==len(bad)
 # Current executable stage output: direct means with 4 decimal serialization, not slopes.
 stage=np.loadtxt(O/'round79'/name/'carlisle.stage',skiprows=1)
 last=(O/'round79'/name/'results/check.stage').read_text().splitlines()[-1].split()
 predicted=[rd.sample(x,y)['depth_m'] for x,y in stage]
 assert np.allclose(predicted,np.array(last[1:],float),rtol=0,atol=.00005000001)
 result[name]=dict(all_marks=len(records),invalid_polynomial_cells=len(bad),raw_requests_rejected=rejected,mean_samples_exact=True,stage18_matches_serialized_means=True,corner_min=float(rd.corner_min.min()))
 dump(R/(name+'-all263-samples.json'),records)
np0=np.zeros((2,3));header=dict(nrows=2,ncols=3,cellsize=1,xllcorner=0,yllcorner=0)
test=DepthReader(np0,np0,np0,header);failures=0
for x,y in [(-.1,0),(3,0),(0,2),(float('nan'),0)]:
 try:test.sample(x,y)
 except ValueError:failures+=1
assert failures==4 and test.sample(0,0)['row']==1 and test.sample(1,1)['row']==0
for a in [np.full((2,3),-.01),np.full((2,3),np.nan),np.zeros((3,2))]:
 try:DepthReader(a,np0,np0,header)
 except ValueError:failures+=1
assert failures==7
result['guard_checks']=7;result['scope']='No change to numerical fields, station files, historical scoring or production defaults';result['field_passes_added']=0
src=O/'round77/lisflood8-rk-ledger/swe/output.cpp';dg=O/'round77/lisflood8-rk-ledger/swe/dg2.cpp'
dump(R/'source-provenance.json',{str(p):sha(p) for p in [src,dg]})
dump(R/'verification.json',result)
report=f'''# 第80轮：明确采样支持范围，隔离不合法格内重构

2026-10-30，沿用第79轮完成结果和第77轮参考exe；没有新求解。生产默认不变，原9项跟踪修改保留，模板v1.3及所有失败保留，现场新增通过0。

## 一、更改日志

新增flood-platform/benchmarks/dg2_depth_reader.py的DepthReader.read/sample。默认cell_mean，明确输出支持范围、原始多项式有效性以及“不推断湿润/速度”。显式raw_polynomial在任一格角为负时拒绝该格，不截负、不改原文件；非有限、负均值、坐标头不同、NODATA与域外均拒绝。采用半开格区间，北/东外边界不擅自夹到边界格。原始栅格所有值保留。

新增audit_sampling80.py在原263调查点和两组完整终态上验证。模块是隔离离线读取入口，尚未接入生产UI；回退为不用该模块继续显式读取格均值，原求解和测站输出无需变动。不是把粗格均值当点真值，仍有空间支持误差。

## 二、根因与证据

实际源码swe/output.cpp::write_depth_samples直接读Arrptr->H；write_speed_samples用H/HU/HV均值和DepthThresh；write_solution的wd是均值。它们没有读取H1x/H1y。swe/dg2.cpp两个RK阶段在通量之前限制坡度，最终系数输出前没有再限制。因此第79轮发现的负原始重构不能用于宣布既有测站样本受污染，更不能借此改写历史分数。

两组18测站的60秒末输出均与原坐标所在格均值的4位小数序列一致，原263坐标默认样本均逐值等于对应均值。原始全属性与样本分别保存，峰值调查数据与60秒输出没有评分。

全场检查结果：归档存在{result['archive']['invalid_polynomial_cells']}个角点重构不合法格，候选{result['mean_preserving']['invalid_polynomial_cells']}格；对应raw模式请求全部拒绝。角点最低深度分别{result['archive']['corner_min']:.9g}、{result['mean_preserving']['corner_min']:.9g}m。该计数是内部表示问题，不是实际漏淹数。原始结果没有被“修好”或删除。

为何不改求解器：现有通量前限制已经在运行，当前采样只用均值。为了文件看起来非负而改变最终状态，可能改变随后演化；本轮没有依据这样做。也不提供只有水深限幅而动量未对应的速度。七项非法输入/域外检查及南西边界/内格边界约定通过。

## 三、可行性、回归与下一步

已完成：现有样本路径审查、参考后端显式支持读取器与拒绝保护；两组全格/全263点原结果对照，无长跑，计算成本秒级。部分实现：离线接口可用，生产UI未接入；未验证：真实淹没、道路、起降、涨退与人员优先级。旧19-J等失败仍在。场景峰值观测不是60秒真值，不生成伪通过。

输出重构问题的用途限制已明确，当前格均淹没分析不因此整体阻塞。下一优先回到Carlisle输入时窗/初态/HVAR及河宽依据，冻结完整事件条件与观察范围；不再扩展本读取器测试或重复60秒。必要时登记条件研究场景，明确不能称验证复现；之后才以全原区域做最大淹没对照，不以原始坡度硬算更细淹没边界。

本轮重新检索时间来源，Parkes论文搜索条目仍有2005-01-07 00:45UTC线索，但未证明与作者2023这组900秒曲线同一时间零点，未采用。其它检索中的不同事件初态不能借用。已有全文获取失败记录不循环尝试；不移时套峰。来源线索与局限保存web-leads.json。

跳过：不重跑800秒/墙位/HLLC或静水套件，不改地形、边界、糙率，不做NS/GPU或UI预览。没有通过替换入口绕过浏览器限制。完整动量场未导出仍不声称验证；调查点之外、域外和最大范围以外未知保留。
'''
(R/'采样支持范围与重构保护-20261030.md').write_text(report,encoding='utf-8');(D/'audit-round80-20261030.md').write_text(report,encoding='utf-8')
dump(R/'web-leads.json',{'searched':'2026-10-30','primary_thesis':'https://kings-qa.qa.elsevierpure.com/files/51218432/2015_Parkes_Brandon_0838465_ethesis.pdf','status':'search snippet only in this turn; not a closed epoch for author2023 curves; earlier full retrieval failures not repeated','claimed_epoch_adopted':False})
for n in ['dg2_depth_reader.py','audit_sampling80.py']:
 b=(S/n).read_bytes();ast.parse(b);(R/n).write_bytes(b);(P/'flood-platform/benchmarks'/n).write_bytes(b)
mirror=D/'round80-artifacts';mirror.mkdir(exist_ok=False)
for f in R.iterdir():
 if f.is_file():shutil.copy2(f,mirror/f.name)
entry=f'''## 2026-10-30 第80轮：采样路径核清，显式格均读取与重构拒绝保护已完成

第79轮原始坡度负重构不会直接污染现有测站：源码write_depth_samples只用H均值，速度用均值H/HU/HV。两组18测站末样本按4位小数复核一致；全部263原点保留，60秒不能与峰值观测评分。原始多项式角点不合法格归档{result['archive']['invalid_polynomial_cells']}、候选{result['mean_preserving']['invalid_polynomial_cells']}，隔离DepthReader的显式raw模式全部拒绝、不截负。默认格均返回支持范围和不推断湿润/速度标记；7项非法输入/域外检查通过。模块未接生产UI，不改变求解或旧成绩，现场新增通过0。

无需再重复此采样/60秒套件；输出限制已有明确处理，格均区域分析不因此整体阻塞。下一主任务仍为同事件时间零点、164000/245700覆盖、HVAR/free与初态/河宽依据，登记完整区域对照或明确条件研究范围。Parkes00:45UTC仍只是不能绑定2023曲线的线索，不移时套峰。原9项修改、19-J等旧失败和模板v1.3保留，无后台实验。报告round80/采样支持范围与重构保护-20261030.md及docs/audit-round80-20261030.md，原点样本/源码/哈希双份。

'''
logs=[O/'HYRA自动续作进度.md',O/'HYRA累计更改日志.md',O/'HYRA现实贴合证据与改进工作表-20261029.md',D/'HYRA自动续作进度.md',D/'HYRA累计更改日志.md',D/'HYRA现实贴合证据与改进工作表-20261029.md',D/'reality-validation-workplan-20261029.md'];(R/'logs-before').mkdir()
for i,p in enumerate(logs):
 b=p.read_bytes();(R/'logs-before'/f'{i}-{p.name}').write_bytes(b);p.write_bytes(entry.encode('utf-8')+b)
manifest={str(p.relative_to(R)).replace('\\','/'):sha(p) for p in R.rglob('*') if p.is_file()};dump(R/'manifest.json',manifest)
assert all(sha(R/k)==v for k,v in manifest.items())
print(json.dumps(result,ensure_ascii=False))
