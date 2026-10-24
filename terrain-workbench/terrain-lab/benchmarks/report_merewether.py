"""Audit saved measurements/results and publish a compact, traceable report."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np

LAB=Path(__file__).resolve().parents[1];sys.path.insert(0,str(LAB))
import merewether


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=LAB/'.runtime/merewether/results')
    p.add_argument('--output',type=Path,default=LAB.parent/'docs/experiments/2026-10-20/merewether')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report=[];checks={};rows=[]
    for label in ['f4-n0.042','f8-n0.042','f8-n0.02']:
        source=args.results/label;r=json.loads((source/'result.json').read_text(encoding='utf-8'))
        cfg=json.loads((source/'config.json').read_text(encoding='utf-8'))
        expected=merewether.observations(LAB/'.runtime/merewether/source/experiment')
        checks[label+'_observations_match_source']=all(all(a[k]==b[k] for k in b) for a,b in zip(r['points'],expected)) and len(expected)==len(r['points'])==40
        error=np.array([x['model_max_speed_m_s']-x['speed_m_s'] for x in r['points']])
        checks[label+'_rmse']=bool(np.isclose(np.sqrt(np.mean(error**2)),r['metrics']['speed_rmse_m_s'],rtol=1e-12))
        checks[label+'_complete']=r['complete'] and r['duration_s']==800
        checks[label+'_mass']=abs(r['balance']['residual_m3'])<1e-6 and r['balance']['numerical_adjustment_m3']==0
        checks[label+'_forcing']=abs(r['balance']['external_inflow_m3']-19.6992*800)<1e-6
        with np.load(source/'final.npz',allow_pickle=False) as a:
            checks[label+'_finite_nonnegative']=all(np.isfinite(a[k]).all() for k in a.files) and bool((a['depth']>=0).all())
            checks[label+'_actual_roughness']=bool(np.all(a['n']==r['n']))
            checks[label+'_saved_volume']=abs(float(a['depth'].sum())*r['dx_m']**2-r['balance']['current_m3'])<1e-7
        small={k:r[k] for k in ['dx_m','shape','n','duration_s','complete','wall_s','quasi_steady','metrics','balance','water_level_validation']}
        report.append(small)
        target=args.output/label;target.mkdir(exist_ok=True)
        for filename in ['result.json','config.json','points.csv','comparison.png','water-levels.png']:
            shutil.copyfile(source/filename,target/filename)
        rows.append(f'| {r["dx_m"]:g} | {r["n"]:g} | {r["metrics"]["speed_rmse_m_s"]:.3f} | {r["metrics"]["speed_mae_m_s"]:.3f} | {r["metrics"]["direction_mae_deg"]:.1f} | {r["metrics"]["model_dry_points"]} | {r["wall_s"]:.1f} |')
    checks={k:bool(v) for k,v in checks.items()}
    write(args.output/'validation.json',checks)
    if not all(checks.values()):raise RuntimeError('Result audit failed: '+str(checks))
    write(args.output/'summary.json',report)
    shutil.copyfile(LAB/'.runtime/merewether/source/provenance.json',args.output/'source-provenance.json')
    if (args.results/'ui-replay.png').exists():shutil.copyfile(args.results/'ui-replay.png',args.output/'ui-replay.png')
    text='''# Merewether 物理实验：首轮导入与验证（2026-10-20）

已接入可回放的原型尺度实验地形，并完成三组800秒计算。**这轮没有通过准确性验收：流速误差仍大，未预先定义的通过阈值也不能事后补作合格依据。**

## 数据身份与糙率

使用 [Ayog 与 Sharifian 的2022年配套数据包](https://zenodo.org/records/6914888)，许可证标注 CC BY 4.0。导入其中 Test 4 的 DEM、BCI、参数、四个水位采样位置及40个实测最大流速／方向点。来源、文件哈希见 source-provenance.json。

与 [Ayog 等（2023）论文](https://doi.org/10.1061/JHEND8.HYENG-13244)的相同配置对接：原型尺度值已由数据作者换算，禁止再乘水平或垂直比例。主配置统一 Manning n=0.042，是文献在水位点4率定的值；本项目没有重新率定。早期 SEAMLESS-WAVE 教程采用0.02，作为敏感性对照，不将两者混成同一配置。

实验表面使用路基材料，并非有植被的城市土地覆盖图。建筑、实墙和细桥墩来自 DEM，本轮不叠加森林或建筑等额外糙率。桥墩约0.35～0.52米，0.7米与1.4米均不能完整解析其几何。

没有使用 ANUGA 同名目录的5个历史洪水观测点；其坐标、地图布置和事件基准不能直接与本次旋转后的实验 DEM 混用。40个速度点是缩尺实验换算到原型尺度的最大速度矢量，不是800秒逐时观测。

## 导入与边界

- 原始 DEM 为696×2197格，0.175米，实际覆盖121.8×384.475米；保留双精度浮点高程，无1米垂直量化。
- ESRI ASCII 行由北向南，原始坐标为局部旋转后的 x/y，不冒称地理 EPSG。求解器采用 x 向右、z 向下；比较时将 z 速度反号变回向北的 v。
- 粗格采用块内最高高程，优先保留桥墩／实墙阻塞，但可能扩大阻水占地、压缩开口。北侧补齐0.525米，两个格宽共同覆盖121.8×385米；不移动南侧入流口。
- 南侧 x∈[67.4375,105.4375] 米入流。BCI 的 QFIX=0.5184 是单宽流量 m²/s，总来水为19.6992 m³/s，不是总来水0.5184 m³/s；与论文四舍五入的19.7一致。
- 按格边实际覆盖比例分配流量，保留38米口宽；北侧自由出流，其他外边界封闭。初始干地，无降雨、入渗、基流、人员或新增地物。
- Rusanov、MUSCL二阶、SSP-RK2、CFL=0.4、low_froude=0；使用当前实际 HYRA 求解器，不用其他模型输出替代它。

## 结果

最大流速每0.5秒提取一次，选择其最大值发生时的同一对 u/v 分量；不将两个分量分别取最大后拼接。使用包含测点的格平均值，水深小于1厘米时不计速度；未进水点仍保留在40点速度误差中。方向比较排除观测或模型速度小于0.05 m/s的点，并分别报告样本数。

| 格宽(m) | Manning n | 流速RMSE(m/s) | 流速MAE(m/s) | 方向MAE(°) | 模型未有效进水点 | 计算耗时(s) |
|---|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

主配置0.7米共95,700格，方向评价37点。40点中2点在本模型中未达到1厘米水深，其中包含实测非零流速位置；没有把它们删除来改善RMSE。1.4米的方向评价36点。各配置水量残差小于1e-6 m³、数值补水为零；这些只表示数值收支一致，不能替代物理准确性。

0.7米主配置达到本轮设定的准稳态诊断（连续三个20秒窗口，湿区平均水深变化<1毫米且采样水位变化<1厘米），1.4米、n=0.042组在800秒末未满足同一诊断，n=0.02组满足。全域平均或少量采样稳定不保证所有局部流速已稳定。

0.7米先按600秒计算预算保存状态，再以240秒预算续算；保留已有 h、mx、mz、最大水深、水量收支、边界累计来水及测点速度最大值，累计主计算约785秒。config.json 中记录续算段及来源哈希。耗时不包括下载与归档，与同时运行的对照共享机器，不能作硬件无关性能比较。

与实测有明显偏差。0.7米相对1.4米的流速RMSE略降，但MAE反而更高，不能概括为全面改善。仅这两档不能确立网格收敛；保形合并、有限体积数值扩散、入流动量闭合及局部三维效应仍需分别检查，当前不能确定主因。不要用继续调 n 来代替几何与边界核验。

## 水位资料缺口

下载包提供4个水位采样位置，但没有对应的原始实测水位值，因此只输出计算曲线，不报告水位RMSE。计算曲线不能当作实验观测。

后续逐格诊断已确认：0.7米最高高程合并将前三处原始测点地面分别抬高约3.531、3.723、5.244米，导致这些位置被扩大的高障碍占用并保持干燥，不能据此与论文图中的相应水位曲线直接评分。当前不自动平移测点、改床面或剔除问题位置。详细证据与新增均值合并对照见 [逐测点诊断报告](diagnostics/REPORT.md)。补齐可追溯的原始水位资料与位置对应关系后再评分。

## 图与文件

![主配置流速对照](f4-n0.042/comparison.png)

各子目录 result.json 保存逐点结果、稳定性和收支；points.csv 可直接检查最差点，config.json 记录边界、参数、来源和代码哈希。water-levels.png 是计算曲线。validation.json 交叉检查40点身份、RMSE重算、800秒窗口、入流水量、实际 n、非负性和保存体积。

原始 DEM、环境、大型 NPZ 与回放帧保留在本地忽略目录，不复制进文档。真实完整数据许可与最初发布者声明随来源文件保留；本轮不发布更改后的原始 DEM 为自有数据。

复现与导入使用说明见 [Merewether 接入说明](../../../merewether.md)。
'''
    (args.output/'REPORT.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(checks_passed=len(checks),summary=report),ensure_ascii=True))


if __name__=='__main__':main()
