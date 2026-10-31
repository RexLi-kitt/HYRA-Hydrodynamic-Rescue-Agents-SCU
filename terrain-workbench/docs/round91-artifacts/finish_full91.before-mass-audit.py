"""Postprocess a completed event only; never launches/restarts a solver."""
from pathlib import Path
import json,hashlib,subprocess,sys,shutil,zipfile
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round91';C=R/'full_event';F=C/'results'
D=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029\terrain-workbench\docs');A=D/'round91-artifacts'
status=json.loads((R/'status.json').read_text())
if status['state']!='completed' or status.get('exit_code')!=0:
 print(json.dumps({'postprocess':'not_started','reason':'event_not_successfully_completed','solver_launched':False}));sys.exit(3)
assert not status.get('stop_reason')
subprocess.run([sys.executable,'-B',str(R/'analyze_full91.py')],check=True,capture_output=True,text=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_grid(path):
 dims=np.fromfile(path,dtype='<i4',count=2);geo=np.fromfile(path,dtype='<f8',count=4,offset=8);a=np.fromfile(path,dtype='<f8',offset=40)
 assert tuple(dims)==(951,611) and tuple(geo[:3])==(338500.,554700.,5.) and a.size==581061
 return a.reshape(611,951)
mx=read_grid(F/'check.maxb');mask=np.load(O/'round90/observation-footprint-mask.npz')['mask']
summary=json.loads((R/'maximum-outline-comparison.json').read_text())
counts=[]
for t in [.01,.1,.3]:
 table=np.bincount((mask.ravel()*2+(mx.ravel()>t)).astype(int),minlength=6).reshape(3,2)
 prior=next(a for a in summary['rows'] if a['threshold_m']==t)
 assert int(table[1,1])==prior['model_above_indicator'] and int(table[1,0])==prior['model_below_or_equal_indicator']
 assert int(table.sum())==581061 and int(table[1].sum())==217464
 counts.append({'threshold_m':t,'rows_outside_unknown_known_wet_edge_unknown_columns_below_above':table.tolist()})
raw_marks=json.loads((O/'round68/prepared/all-survey-marks.json').read_text())
marks=json.loads((R/'all263-mark-comparison.json').read_text())
assert len(marks)==263 and len({m['fid'] for m in marks})==263
elev=json.loads((R/'all263-elevation-compatibility.json').read_text());elev_by_id={a['source_attributes']['fid']:a for a in elev}
for row in marks:
 e=elev_by_id[row['fid']];row['source_attributes']=e['source_attributes'];row['survey_below_archived_grid']=e['survey_Z_minus_grid_elevation_m']<0
 assert row['sim_max_h_m']==float(mx[e['grid_row'],e['grid_col']])
assert sum(m['survey_below_archived_grid'] for m in marks)==81
groups=[]
for flag in [True,False]:
 group=[m for m in marks if m['survey_below_archived_grid']==flag]
 groups.append({'survey_below_grid':flag,'points':len(group),'model_above_main_indicator':sum(m['above_main_analysis_indicator'] for m in group),'model_not_above_main_indicator':sum(not m['above_main_analysis_indicator'] for m in group),'interpretation':'diagnostic association only; no points excluded and no causal conclusion'})
ind=np.load(R/'individual-observation-masks.npz');polys=[]
for fid,pm in zip(ind['fids'],ind['covered_by_polygon']):
 n=int(pm.sum());hit=int((pm&(mx>.1)).sum());polys.append({'fid':int(fid),'eligible_cells':n,'above_main_indicator':hit,'below_main_indicator':n-hit,'coverage':hit/n if n else None,'zero_cells_meaning':'no eligible fully covered cells, NOT pass'})
tm=read_grid(F/'check.maxtmb');wet=mx>.001
assert np.isfinite(tm[wet]).all() and np.all((tm[wet]>=0)&(tm[wet]<=245702/3600))
late={'known_wet_cells_with_model_max_above_indicator_after_48h':int(((mask==1)&(mx>.1)&(tm>48)).sum()),'meaning':'timing diagnostic, not proof observation ends at48h; frozen full-window score unchanged'}
num=json.loads((R/'numerical-verification.json').read_text());assert all(num['checks'].values())
independent={'count_checks':counts,'all263_mark_index_checks':True,'elevation_flag_groups':groups,'all5_polygon_diagnostics':polys,'maximum_time_diagnostic':late,'all_internal_checks':num['checks'],'primary_score_recomputed_from_binary':True,'overall_accuracy':None,'production_adopted':False}
(R/'independent-completion-verification.json').write_text(json.dumps(independent,indent=2))
(R/'all263-mark-comparison-with-source.json').write_text(json.dumps(marks,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
classes=np.zeros(mask.shape,dtype='uint8');classes[mask==2]=1;classes[(mask==1)&(mx>.1)]=2;classes[(mask==1)&(mx<=.1)]=3
colours=['#e5e7eb','#f4ca64','#2485b4','#c64343']
fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
extent=[338500,343255,554700,557755]
axes[0].imshow(classes,origin='upper',extent=extent,cmap=ListedColormap(colours),norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4),interpolation='nearest')
axes[0].set_title('Observed maximum outline vs full-event model')
axes[0].legend(handles=[Patch(color=c,label=l) for c,l in zip(colours,['Outside outline: unknown','Boundary cells: ambiguous','Known wet: model > 0.1 m','Known wet: model <= 0.1 m'])],loc='lower left',fontsize=8)
axes[1].imshow(mx>.1,origin='upper',extent=extent,cmap=ListedColormap(['#f3f4f6','#a3cee4']),vmin=0,vmax=1,interpolation='nearest')
for flag,colour,label in [(False,'#246446','182 other survey points'),(True,'#9b1d37','81 survey elevations below grid')]:
 group=[m for m in marks if m['survey_below_archived_grid']==flag]
 axes[1].scatter([m['E'] for m in group],[m['N'] for m in group],s=11,c=colour,label=label,edgecolors='white',linewidths=.25)
axes[1].set_title('All 263 marks retained; elevation flags are diagnostic')
axes[1].legend(loc='lower left',fontsize=8)
for ax in axes:
 ax.set_xlabel('OSGB Easting (m)');ax.set_ylabel('OSGB Northing (m)');ax.ticklabel_format(style='plain',useOffset=False);ax.set_aspect('equal')
fig.suptitle('Carlisle 2005 | independent LF8 fast reference | no overall accuracy or route-safety claim',fontsize=12)
fig.savefig(R/'full-event-observation-comparison.png',dpi=170);plt.close(fig)
main=next(x for x in summary['rows'] if x['main_indicator']);n=main['observed_fully_covered_cells'];hit=main['model_above_indicator'];miss=main['model_below_or_equal_indicator'];frac=hit/n
report=f'''# 第91轮完整事件结果（2026-10-31）

## 更改日志
独立LF8快速SGC参考后端，原5m格951×611，23源，n=.055、FREE .0006、干初态，固定Run7.bdy对应2005 Run1.csv。本次完整窗口245700s；实际末时{num['actual_end_s']}s，{num['steps']}步，墙钟{num['wall_s']:.1f}s。原HYRA生产、网页和PR均未更新，未运行新DG2摩阻或HLLC。仅启用已有核验通过的4线程及二进制结果保存。回退无需替换生产。

## 原因与完整结果
冻结的实测最大范围内共{n}个完整5m格，模型最大水深>0.1m为{hit}格，未达到指示为{miss}格；正例覆盖{frac:.2%}。这是限定观测湿区的条件性覆盖，不是总体准确率，无法算误报或真干率。若存在漏项，全一致门槛未通过。首个完整参考结果没有相同新案例旧版本基线，不能报告算法提高多少。

原始三阈值结果、所有格和263点均保留。边界9318格、外部354279格仍未知。81点调查水面低于地形格的兼容性问题只作交叉诊断，未删除。归档日期Jan7–8缺日内观测时刻，68.25h全事件最大值仅条件性对应；没有挑选时段。最大场零值意味着没有记录超过1mm，不证明绝对无水。

## 可行性、检查与后续
全部内部预定关口通过；净逐步质量残差{num['total_signed_mass_residual_m3']:.6g}m³，逐步绝对残差总和{num['sum_abs_mass_residual_m3']:.6g}m³。独立程序从二进制重新计算三阈值全部分类、263点索引和每个多边形诊断，与主评分一致。质量检查不等于现场可靠。

已实现：完整条件模拟与独立最大淹没正例比较。部分实现：高程和最大值时间归因。未验证：独立干区误报、连续实测涨退/主流向、道路与起降候选、人员搜索排序；全部定性目标仍未通过。未进行新的时间减半/网格回归，不称收敛。

下一步先检查全部未覆盖区域与81点高程旗标的空间关联；有原始地形/建筑或边界证据才设计单因素修复，不压低测点、不移点、不更换阈值。没有可支持改动时明确资料缺口，保留本结果用于受限研究，不升级默认。此轮计算结束，后续先查进程，不重复完整配置。
'''
report+='\n|指示深度m|已覆盖|未达指示|限定正例覆盖|\n|---|---:|---:|---:|\n'+''.join(f"|{x['threshold_m']}|{x['model_above_indicator']}|{x['model_below_or_equal_indicator']}|{x['positive_coverage_fraction']:.2%}|\n" for x in summary['rows'])
(R/'audit-round91-complete-20261031.md').write_text(report,encoding='utf8')
raw_manifest={str(p.relative_to(C)):sha(p) for p in C.rglob('*') if p.is_file()}
(R/'complete-raw-hashes.json').write_text(json.dumps(raw_manifest,indent=2))
with zipfile.ZipFile(R/'full-event-raw.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for rel in raw_manifest:z.write(C/rel,'full_event/'+rel)
shutil.copy2(__file__,R/Path(__file__).name)
names=['protocol.json','status.json','analyze_full91.py','numerical-verification.json','maximum-outline-comparison.json','all263-mark-comparison.json','all263-mark-comparison-with-source.json','full-classification.npz','independent-completion-verification.json','full-event-observation-comparison.png','audit-round91-complete-20261031.md','complete-raw-hashes.json','full-event-raw.zip',Path(__file__).name]
manifest={}
for name in names:
 shutil.copy2(R/name,A/name);manifest[name]=sha(R/name);assert sha(A/name)==manifest[name]
for root in [R,A]:(root/'completion-artifact-hashes.json').write_text(json.dumps(manifest,indent=2))
for root in [O,D]:shutil.copy2(R/'audit-round91-complete-20261031.md',root/'audit-round91-complete-20261031.md')
entry=f'''## 2026-10-31 第91轮完整事件已收尾

原245700s独立5m参考运行完成，内部关口全部通过，未升级生产/网页/PR。冻结217464实测湿区完整格中>0.1m为{hit}，未达指示{miss}，限定正例覆盖{frac:.2%}；不是整体准确率，未知干区/边界、时间范围与81点高程问题保留。全部263点、三阈值、原场、逐步水量、独立复算与地图已双份归档round91/docs round91-artifacts；详见audit-round91-complete-20261031.md。无旧版同案例基线，不称算法提升。完整计算已退出，不重跑。下一步先依据全部漏项和独立输入证据归因，不任意改地形、NS或验收阈值。全目标未通过。

'''
for root in [O,D]:
 for name in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
  p=root/name;p.write_text(entry+p.read_text(encoding='utf8'),encoding='utf8')
print(json.dumps({'complete':True,'main_result':main,'independent_checks_pass':True,'groups':groups,'map':str(R/'full-event-observation-comparison.png'),'dual_files':len(names)}))
