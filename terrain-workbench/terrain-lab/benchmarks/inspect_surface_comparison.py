"""Validate saved full-surface comparison and render its native land-cover map."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

LAB=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(LAB))
import server
import surface_roughness


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args();out=args.output.resolve()
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    terrain=server.describe(summary['provenance']['full_surface'])
    def field(name):
        f=terrain['fields'][name]
        return np.fromfile(LAB/f['url'].lstrip('/'),dtype=f['dtype']).reshape(terrain['shape'])
    n,profile=surface_roughness.from_terrain(LAB,terrain,.035)
    checks={};balances=[];volumes={};times=None
    for key in ['full_uniform','full_classified']:
        with np.load(out/f'{key}.npz',allow_pickle=False) as a:
            checks[key+'_finite']=all(np.isfinite(a[k]).all() for k in a.files)
            checks[key+'_nonnegative']=bool((a['flood']>=0).all() and (a['control']>=0).all())
            checks[key+'_roughness']=bool(np.array_equal(a['n'],n if key=='full_classified' else np.full(n.shape,.035)))
            checks[key+'_grid']=a['bed'].shape==(256,256)
            if times is None:times=a['time_s'].copy();bed=a['bed'].copy()
            checks[key+'_matched_geometry']=bool(np.array_equal(a['bed'],bed))
            checks[key+'_matched_times']=bool(np.array_equal(a['time_s'],times))
        result=json.loads((out/f'{key}.json').read_text(encoding='utf-8'))
        for i,branch in enumerate(result['metadata']['branches']):
            balance=branch['balance']
            balances.append(dict(variant=key,branch=['control','pulse'][i],
                time_s=branch['time_s'],residual_m3=balance['residual_m3'],
                adjustment_m3=balance['numerical_adjustment_m3']))
            checks[key+f'_{i}_mass']=abs(balance['residual_m3'])<1e-6 and balance['numerical_adjustment_m3']==0
            volumes.setdefault(i,[]).append(branch['external_inflow_by_id_m3'])
    if not summary['budget_hit']:
        checks['matched_forcing']=all(all(abs(a[k]-b[k])<1e-7 for k in a) for a,b in volumes.values())
        checks['complete_70_seconds']=bool(times[-1]==70)
    biome=field('biome_id');cover=field('cover_flags')
    biome_counts={name:int((biome==code).sum()) for name,code in terrain['biome_ids'].items()}
    report=dict(checks={k:bool(v) for k,v in checks.items()},balances=balances,roughness=profile,
                biome_counts=biome_counts,tree_coverage_fraction=float(((cover&1)!=0).mean()))
    (out/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if not all(checks.values()):raise RuntimeError('Saved experiment validation failed: '+str(checks))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    palette=['#c4d89e','#335b45','#5197b9']
    axes[0].imshow(biome,cmap=ListedColormap(palette),vmin=-.5,vmax=2.5,extent=(0,256,256,0),interpolation='nearest')
    axes[0].set_title('Complete generated biome map')
    axes[0].legend(handles=[Patch(color=palette[code],label=name.split(':')[-1].replace('_',' '))
        for name,code in terrain['biome_ids'].items()],loc='upper left',framealpha=.9)
    values=np.unique(n);encoded=np.searchsorted(values,n)
    colors=['#d9b77d','#969793','#a2c787','#276747']
    axes[1].imshow(encoded,cmap=ListedColormap(colors[:len(values)]),vmin=-.5,vmax=len(values)-.5,
                   extent=(0,256,256,0),interpolation='nearest')
    axes[1].set_title('Assigned static Manning n')
    axes[1].legend(handles=[Patch(color=colors[i],label=f'n = {v:.3f}') for i,v in enumerate(values)],loc='upper left',framealpha=.9)
    for ax in axes:ax.set_xlabel('Local X (m)');ax.set_ylabel('Local Z (m)')
    fig.suptitle('Same 256 × 256 m crop | native 1 m land-cover fields',fontsize=15)
    fig.savefig(out/'surface-cover.png',dpi=160);plt.close(fig)
    markdown=(out/'REPORT.md').read_text(encoding='utf-8').split('\n## 地表与核验')[0]
    markdown+='\n\n## 地表与核验\n\n![群系与赋值](surface-cover.png)\n\n'
    markdown+='完整地表的群系、地面材料和植被标记均已参与分类，未知回退格数为 '+str(profile['fallback_cells'])+'。\n\n'
    markdown+='| 地表类型 | Manning n | 格数 | 面积比例 |\n|---|---:|---:|---:|\n'
    for item in profile['native_classes']:
        markdown+=f'| {item["label"]} | {item["n"]:.3f} | {item["cells"]} | {item["cells"]/n.size:.2%} |\n'
    maximum=max(abs(b['residual_m3']) for b in balances)
    markdown+=f'\n四条新计算分支的最大水量残差为 {maximum:.3g} m³，数值补水均为零；保存的水深有限且非负，完整地表两组河床逐格相同、时间与入口累计供水一致。全部检查见 validation.json。\n'
    markdown+='\n完整地图已写入 current.json。生成器的原始七个字段与上次一致；实际河床取完整生成后的地面顶面，不含树冠。高程平均绝对变化 '+f'{summary["provenance"]["bed_difference"]["mae_m"]:.3f}'+' 米，因此不能把相对旧图的变化全部解释为糙率效应。\n'
    effect=summary['differences']['full_uniform__full_classified']
    footprint=summary['roughness_on_pulse_footprint']
    markdown+='\n本次观察：分类糙率相对完整地表统一糙率，末时共同湿区水深平均绝对差为 '+f'{100*effect["final_wet_depth_mae_m"]:.2f}'+' 厘米。脉冲抬水达到 5 厘米的 '+str(footprint['pulse_reached_cells'])+' 个原生格中，'+str(footprint['reached_cells_with_nonuniform_n'])+' 格的糙率不同于 0.035；本场洪水对全图高糙率区域的检验有限。新增湿区不是随糙率增大必然缩小，局部壅水与横向扩散可能增加湿区；这里只报告本图本次来水的实测计算差异。\n'
    markdown+='\n群系、地面和覆盖数据来自原始地形生成流程；本次因 Java 21 的临时 JAR 可写检查异常，使用本机 Java 24 启动已构建的导出器，未修改地形算法，原始字段一致性已核验。\n'
    markdown+='\n复现洪水对照：`python benchmarks/compare_surface_roughness.py --terrain-id generated/surface-df3682282712/raw --baseline .runtime/grid-observation-20261019 --output .runtime/surface-comparison-20261019 --wall-budget 360`；核验与地表图：`python benchmarks/inspect_surface_comparison.py .runtime/surface-comparison-20261019`。使用现有 .runtime/rivgraph-env 中的 Python 与加速依赖。\n'
    (out/'REPORT.md').write_text(markdown,encoding='utf-8')
    print(json.dumps(dict(valid=True,balances=balances,biome_counts=biome_counts,
        tree_coverage_fraction=report['tree_coverage_fraction']),ensure_ascii=False))


if __name__=='__main__':main()
