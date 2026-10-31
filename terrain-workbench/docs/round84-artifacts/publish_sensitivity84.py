from pathlib import Path
import json,hashlib,shutil,zipfile
import numpy as np
O=Path(r'C:\Users\Lenovo\Documents\ChatGPT\无人机集群搜救大创\outputs');R=O/'round84';S=Path(__file__).parent
REPO=Path(r'C:\Users\Lenovo\Desktop\无人机大创\HYRA-PR-20261029');D=REPO/'terrain-workbench/docs'
status=json.loads((R/'status.json').read_text());assert status['state']!='running'
comparison=json.loads((R/'comparison.json').read_text()) if (R/'comparison.json').exists() else None
note='未通过内部关口，保存结果并停止长跑路线。'
if comparison:
    last=comparison['frames'][-1];wet=last['wet_mask_comparisons'][1]
    note=f"两组均完成2400s并通过有限非负/水量关口。最终内部h>1mm比较：共同湿格{wet['both_wet']}，仅dtmax1湿{wet['baseline_only_wet']}，仅dtmax.5湿{wet['candidate_only_wet']}；最大水深差{last['max_abs_depth_difference_m']:.8g}m。步数仅5260→5387，实际步长中位约.476s未减半；此上限试验不构成真实步长减半或收敛证明。属于离散敏感性，不是现场准确率。"
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    a=np.loadtxt(O/'round83/carlisle_2400/results/check-0007.wd',skiprows=6)
    z=np.loadtxt(R/'carlisle_2400_dt_half/results/check-0007.wd',skiprows=6)
    cats=(a>.001).astype(int)+2*(z>.001).astype(int)
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    extent=(338490,343250,554700,557760)
    colors=['#eeeeee','#d55e00','#009e73','#0072b2']
    axs[0].imshow(cats,origin='upper',extent=extent,cmap=ListedColormap(colors),vmin=0,vmax=3,interpolation='nearest')
    axs[0].set_title('2400 s: numerical h > 1 mm (not observed flooding)')
    axs[0].legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,['Neither','dtmax 1 only','dtmax 0.5 only','Both'])],loc='upper left',fontsize=8)
    delta=z-a;bound=max(float(abs(delta).max()),1e-12)
    im=axs[1].imshow(delta,origin='upper',extent=extent,cmap='RdBu_r',vmin=-bound,vmax=bound)
    axs[1].set_title('Depth difference: dtmax 0.5 minus 1 (m)');fig.colorbar(im,ax=axs[1])
    for ax in axs:ax.set_xlabel('Easting (m, EPSG:27700)');ax.set_ylabel('Northing (m)');ax.ticklabel_format(style='plain',useOffset=False)
    fig.savefig(R/'time-sensitivity-2400s.png',dpi=160);plt.close(fig)
report=f'''# 第84轮：实际时间步上限敏感性

## 更改日志

2026-10-30。基线round83/carlisle_2400；同一个round83 CPU64 exe，源码及生产默认完全未改。新增run_sensitivity84/analyze_sensitivity84/publish_sensitivity84离线实验脚本，唯一配置差异initial_tstep由1改.5秒；仍2400秒、300秒输出、原10m地形/源/边界/糙率/干初态，所有输入逐文件哈希核对。回退为继续使用原配置，不覆盖旧结果，不自动更新PR #10。原网页8766服务保持。

## 原因与方法

第83轮只证明候选跨过旧负深失败位置，未证明结果对实际步长稳定。自适应CFL下参数上限减半不等于每个实际步减半，因此记录实际步数/步长分布/拒绝次数，并将共同300s时刻全部8场作对照。预先登记内部阈值1e-6、1e-3、1e-2m，分别保存共同湿、仅基线湿、仅候选湿及均非湿；不选其中最有利的阈值，不把这些阈值当人员安全或真实湿润界线。

比较全域水深绝对差、体积、全部分类掩膜及首个保存湿帧。首湿帧只有300s分辨率，不是精确到达时间。相同模拟之间的交并比仅描述离散敏感性，不是实测准确率；没有观测依据的误差容限时不虚报收敛或验收。

## 可行性、结果与剩余限制

{note}

全8帧最大差由.00201–.08475m不等；内部h>1mm的分类变化每帧3–14格。最后一帧13格=1300m²不能自动当无搜救影响。共同曾湿2342格中，2299格首个保存湿帧相同，43格不同，最多相差900s；这里只知道300s保存帧，不能作为现实到达时间误差。h>1cm最后5格不同，仍全部保留，不选择更好看的阈值宣布正确。

执行成本298.40→329.44s，拒绝次数9→5，源曲线独立积分一致。关键区分：实际中位步长.475757→.476229s，减半的是上限而非主体CFL控制步；不能将这个结果包装成已经完成了“实际步长减半敏感性”。

```json
{json.dumps(status,ensure_ascii=False,indent=2)}
```

完整全帧/全阈值对照见comparison.json和full-mask-comparisons.npz；全部原始输入/输出/尝试在raw-experiment.zip，基线保留round83。若图存在，它显示的是两个数值解的差别，不含现场验证图层。

本轮回归范围为相同真实事件输入的短时数值敏感性；不重跑已通过3项小回归，因为源码/可执行文件未变。旧Merewether40点和Okushiri墙位跳过，不把不同案例成绩移用到新候选。零新增现场通过；完整164000s、绝对时刻、HVAR基准/free区别、初态与河宽缺口仍存在。

下一步不得因两组内部关口通过就宣称定性正确。停止继续调整未起主导作用的dt上限；改做一个能真正改变内部CFL步的单因素对照：以round83为基线，dtmax仍1、原CFL=.33改.165，同2400s及8输出帧、同exe/输入，15分钟墙钟上限、失败不重启。先验证实际步分布确有细化，再比较固定全帧及3阈值；不能挑分高的参数或称两个/三个数值解相近即现场通过。这是为补本轮发现的离散证据缺口，并非再盲扫参数。完成后应结束短时步长分支并决定完整条件研究，保留输入未闭合声明和独立观测。不用无限小测试代替区域实测。

额外公开资料核验：本轮重新尝试Parkes学位论文原站点仍返回502；搜索摘要中的Jan7 00:45UTC没有获得新的正文核验，也没有证明与2023归档曲线同一时间轴，继续列未闭合，不向模型赋值、不对齐峰时。原论文URL：https://kclpure.kcl.ac.uk/portal/files/51218432/2015_Parkes_Brandon_0838465_ethesis.pdf
'''
(R/'audit-round84-20261030.md').write_text(report,encoding='utf-8')
for n in ['run_sensitivity84.py','analyze_sensitivity84.py','publish_sensitivity84.py']:shutil.copy2(S/n,R/n)
with zipfile.ZipFile(R/'raw-experiment.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as f:
    for p in (R/'carlisle_2400_dt_half').rglob('*'):
        if p.is_file():f.write(p,p.relative_to(R))
manifest=[{'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sorted(R.iterdir()) if p.is_file() and p.name!='manifest.json']
(R/'manifest.json').write_text(json.dumps(manifest,indent=2))
dest=D/'round84-artifacts';dest.mkdir(exist_ok=True)
for p in R.iterdir():
    if p.is_file():shutil.copy2(p,dest/p.name)
shutil.copy2(R/'audit-round84-20261030.md',D/'audit-round84-20261030.md')
update=f'''## 2026-10-30 第84轮：同2400s时间步上限1→.5敏感性已收尾

状态{status['state']}，退出{status['exit_code']}，到达{status.get('end_s')}s，墙钟{status['wall_s']:.2f}s。{note}

代码/exe不变，原始8文件只有case.par时间步上限改变；全8场、3阈值、实际步和源独立积分对照保存round84及docs/round84-artifacts。薄水首保存湿帧43格不同、最多差900s，不当现实到达误差。下一最小区分试验固定round83的dtmax1，仅CFL .33→.165，让内部接受步真正细化；同2400s/8帧、15分钟上限，失败不重启，停止继续扫描dt上限。之后结束短时步分支并决定完整条件研究。

未改生产或网页、未自动推送PR，现场新增通过0。报告audit-round84-20261030.md；不能把软件间吻合当现实准确率或重复同配置，时刻/HVAR/初态/河宽缺口保持。无本实验后台，server.py8766是用户网页服务。

'''
for n in ['HYRA自动续作进度.md','HYRA累计更改日志.md','HYRA现实贴合证据与改进工作表-20261029.md']:
    for root in [O,D]:
        p=root/n
        if p.exists():p.write_text(update+p.read_text(encoding='utf-8-sig'),encoding='utf-8')
assert all(hashlib.sha256((dest/x['file']).read_bytes()).hexdigest()==x['sha256'] for x in manifest)
print(json.dumps({'published':str(dest),'files':len(manifest),'all_hashes_match':True}))
