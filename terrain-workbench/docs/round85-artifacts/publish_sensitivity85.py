from pathlib import Path
import json, hashlib, shutil, zipfile
import numpy as np
S=Path(__file__).parent
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs'); R=O/'round85'
D=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029\terrain-workbench\docs')
C=R/'carlisle_2400_cfl_half'; B=O/'round83/carlisle_2400'
st=json.loads((R/'status.json').read_text()); assert st['state']!='running'
co=json.loads((R/'comparison.json').read_text()) if (R/'comparison.json').exists() else None
exe=O/'round83/build-rk-retry-cpu64/Release/lisflood.exe'
assert hashlib.sha256(exe.read_bytes()).hexdigest()=='039269203d57cd2a0f82c6f817fe9476e97bbb191eee6695f7e3ac8e9f5bc55a'
extra={'exe_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'field_passes_added':0}
extra['full_window_cost_planning']={'full_duration_s':164000,'short_duration_s':2400,'linear_wall_hours_nonforecast':st['wall_s']*164000/2400/3600,'caveat':'Linear throughput extrapolation only; inundated area, CFL and rejection frequency will change. Not a completion-time promise.','restart_supported':False,'threads_supported':1,'next_gate':'Decide bounded complete-window cost and recoverable state strategy before launching; no blind whole-run restart.'}
note='未通过内部关口，不评分不完整场；保存失败且不自动重启。'
detail=''
if co:
    a=np.loadtxt(B/'accepted_timesteps.txt',ndmin=2); b=np.loadtxt(C/'accepted_timesteps.txt',ndmin=2)
    assert np.all(b[:,1]>0) and abs(b[-1].sum()-2400)<1e-9
    extra.update(baseline_dt_quantiles=np.quantile(a[:,1],[0,.1,.5,.9,1]).tolist(),candidate_dt_quantiles=np.quantile(b[:,1],[0,.1,.5,.9,1]).tolist(),median_dt_ratio=float(np.median(b[:,1])/np.median(a[:,1])),step_count_ratio=len(b)/len(a))
    final=co['frames'][-1]; w=final['wet_mask_comparisons'][1]; first=co['first_saved_wet_frame_comparisons'][1]
    note=f"完成2400s：实际步数{len(a)}→{len(b)}，中位步长{np.median(a[:,1]):.8g}→{np.median(b[:,1]):.8g}s，比例{extra['median_dt_ratio']:.6f}；拒绝{st['rejections']}次。全部8帧有限非负且源/边界账目通过。最终h>1mm：共同湿{w['both_wet']}格，仅基线湿{w['baseline_only_wet']}格，仅细步湿{w['candidate_only_wet']}格；最大水深差{final['max_abs_depth_difference_m']:.8g}m。"
    rows=['|时间s|最大水深差m|h>1e-6变化格|h>1mm变化格|h>1cm变化格|','|---|---|---|---|---|']
    for f in co['frames']:
        counts=[x['baseline_only_wet']+x['candidate_only_wet'] for x in f['wet_mask_comparisons']]
        rows.append(f"|{f['time_s']}|{f['max_abs_depth_difference_m']:.9g}|{counts[0]}|{counts[1]}|{counts[2]}|")
    detail='\n'.join(rows)+f"\n\n内部h>1mm共同曾湿{first['both_ever_wet']}格，首个保存湿帧相同{first['same_first_saved_frame']}，细步较早{first['candidate_earlier_saved_frame']}、较晚{first['candidate_later_saved_frame']}，最大帧时差{first['max_abs_saved_frame_time_difference_s']}s。采样每300s，不能声称精确到达时差。三阈值全部掩膜与首湿索引保存在full-mask-comparisons.npz；不是现场湿润阈值或人员安全规则。"
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    h0=np.loadtxt(B/'results/check-0007.wd',skiprows=6); h1=np.loadtxt(C/'results/check-0007.wd',skiprows=6)
    colors=['#eeeeee','#d55e00','#009e73','#0072b2']; extent=(338490,343250,554700,557760)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    axes[0].imshow((h0>.001).astype(int)+2*(h1>.001).astype(int),extent=extent,origin='upper',cmap=ListedColormap(colors),vmin=0,vmax=3,interpolation='nearest')
    axes[0].set_title('2400 s numerical wet masks; no field truth')
    axes[0].legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,['Neither','CFL .33 only','CFL .165 only','Both'])],fontsize=8)
    d=h1-h0; bound=max(float(abs(d).max()),1e-12)
    im=axes[1].imshow(d,extent=extent,origin='upper',cmap='RdBu_r',vmin=-bound,vmax=bound)
    axes[1].set_title('Depth: CFL .165 minus .33 (m)'); fig.colorbar(im,ax=axes[1])
    for ax in axes:
        ax.set_xlabel('Easting (m, EPSG:27700)');ax.set_ylabel('Northing (m)');ax.ticklabel_format(style='plain',useOffset=False)
    fig.savefig(R/'actual-step-sensitivity.png',dpi=160);plt.close(fig)
(R/'independent-step-verification.json').write_text(json.dumps(extra,indent=2))
report=f'''# 第85轮：实际CFL步长细化对照

## 更改日志

2026-10-30。基线round83/carlisle_2400；同一个隔离CPU64可执行文件（SHA见independent-step-verification.json），不改求解源码、生产默认、用户网页或PR。run_sensitivity85只在case.par追加cfl .165，原默认.33；initial_tstep仍1s。固定2400s和300s输出，原地形、原边界、糙率.06、干初态、全部测点保持。7个非配置输入逐文件哈希一致。新增分析/发布脚本，不覆盖任何旧结果。回退为使用原round83配置；候选不升级默认。

## 更改原因与方法

round84仅把时间步上限1→.5，中位实际步仍约.476s，未完成实际步细化。此次只减半CFL系数，保存每个接受步及拒绝尝试，独立检查步长分布、连续时间、源分段线性积分、水量以及全部8帧。固定内部深度阈值1e-6/.001/.01m，保留全部差异而不选择较好阈值。拒绝操作的物理/数值机制沿用round83，代码不变。重负载只此一项，15分钟墙钟上限，失败不重启。

## 可行性与结果

{note}

{detail}

```json
{json.dumps(st,ensure_ascii=False,indent=2)}
```

已实现：限定正点源、无雨、无重启条件下的短时内部数值检查及真实接受步敏感性。部分实现：现场输入归档接入。未验证：完整洪水过程、真实空间淹没、道路和起降点影响。现场通过新增0。两个数值解的掩膜一致率不是现实正确率；两个步长也不构成收敛证明。不能按哪组更符合观测来挑时间步。

回归：同exe、输入哈希、时间积分、全8场有限非负和水量检查；不重复round83已完成的3项无触发回归（无代码改变），不重跑旧800s/墙位/HLLC。未做：完整164000s、多网格收敛、动量外部守恒总账、降雨/负点源/重启分支、现场定性验收。原多项不一致不删除。运行原始文件和拒绝日志全归档；拒绝前9场文件仅保留最新一次的机制限制同round83。

决定：内部数值关口通过，但不采用新参数为现实升级，也不启动完整长跑。最终h>1mm改变315格（31500m²），h>1cm仍改变175格，局部最大深差在2100s达约.639m；不能仅因总水量相同就认为搜救淹没判断稳定。差异在300s已出现36格，早于基线2179s首次拒绝，因此“全部差异仅来自后段RK回滚”已被排除；源总积分也一致，不能归因于简单的累计入水量差。尚不能据此断定哪组更接近实测或二维方程失效。

下一步结束盲扫时间步分支。先复用已存300/600s全场定位分流差异在源附近、河道还是薄层前缘；检查实际启用的摩阻分裂、限幅/干湿动量处理与点源逐步注入的耦合，提出能区分原因的一个局部复现实验，失败条件为不能复现早期差异或只能靠改变物理输入调分。不新跑相同2400s，不直接改阈值、流量、地形或换NS。输入证据核验独立继续。完整事件待此敏感性有可解释范围和成本方案后再决定，若启动仍须单独预登记为“2023归档输入下的条件数值试验”，保留起时/HVAR/初态等缺口，不能把2400s与最大淹没图对照。

长程成本审查：当前dg2.cpp入口明确限定OMP_NUM_THREADS=1且拒绝t不为0的重启；文件名含prestate不代表已有可恢复检查点。按本次短程吞吐线性折算164000s约{extra['full_window_cost_planning']['linear_wall_hours_nonforecast']:.2f}小时，仅作成本警示，后期湿区增大或步长变化会改变耗时，不能作为交付承诺。本轮不启动不可恢复的整夜长跑；下一单元须把全程成本、可恢复状态策略和原始结果留存闭合，避免失败后从零反复算。不得直接将OMP改多线程来绕过候选限定。

本轮独立资料审查读取作者论文 https://gmd.copernicus.org/articles/16/3291/2023/ 。它描述自由出口，并将183水位标记用于糙率校准；归档HVAR差异没有因此自动消失。文章的最近湿格采样不能照搬为移动观测坐标；归档模型/校准数据不能当独立留出真值。没有新增绝对时刻或高程基准闭合，不改变输入。
'''
(R/'audit-round85-20261030.md').write_text(report,encoding='utf-8')
for name in ['run_sensitivity85.py','analyze_sensitivity85.py','publish_sensitivity85.py']:shutil.copy2(S/name,R/name)
with zipfile.ZipFile(R/'raw-experiment.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
    for p in C.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(R))
manifest=[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(R.iterdir()) if p.is_file() and p.name!='manifest.json']
(R/'manifest.json').write_text(json.dumps(manifest,indent=2))
dest=D/'round85-artifacts';dest.mkdir(exist_ok=True)
for p in R.iterdir():
    if p.is_file():shutil.copy2(p,dest/p.name)
shutil.copy2(R/'audit-round85-20261030.md',D/'audit-round85-20261030.md')
update=f'''## 2026-10-30 第85轮：CFL .33→.165实际步长对照已收尾

{note} 状态{st['state']}，墙钟{st['wall_s']:.2f}s。独立积分与全帧分类保存round85及docs/round85-artifacts，报告audit-round85-20261030.md。软件对照不是现实正确率，现场通过仍未增加；不升级默认，不自动推送PR。

短时步长盲扫至此收尾。最终1mm分类315格不同、1cm仍175格，局部最大深差2100s约.639m，不能当定性稳定；300s已出现差异，早于基线2179s首次回滚，排除仅由后段拒绝造成差异。下一项复用300/600s场定位源/河道/薄层差异，审查摩阻分裂/限幅干湿动量/点源逐步注入耦合，再登记一个局部区分实验，不再直接跑整场或改物理输入调分。完整长跑暂不启动；当前单线程且不支持重启，短程线性折算全程约10.5小时也只是成本估算。起时、HVAR/free、初态与河宽证据仍未闭合。原网页8766服务不动，本实验已退出，不重复启动。

'''
for name in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
    for root in [O,D]:
        p=root/name
        if p.exists():p.write_text(update+p.read_text(encoding='utf-8-sig'),encoding='utf-8')
assert all(hashlib.sha256((dest/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in manifest)
print(json.dumps({'published':str(dest),'files':len(manifest),'hashes_match':True,'note':note},ensure_ascii=False))
