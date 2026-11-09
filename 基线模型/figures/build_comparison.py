"""Rebuild the published-model chart from the checked-in evidence (Python + matplotlib)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
rows = json.loads((HERE.parent / 'evidence/comparison.json').read_text(encoding='utf-8-sig'))
models = ['MAPPO', 'DQN', 'VDN', 'QMIX', 'QPLEX', 'MAT', 'V44 default', 'V44 pure']
versions = {'MAT': 'contact-ppo-1024', 'V44 default': 'official-exit-720s', 'V44 pure': 'official-exit-720s'}
fonts = {f.name for f in font_manager.fontManager.ttflist}
for name in ['Microsoft YaHei', 'Noto Sans CJK SC', 'SimHei', 'Arial Unicode MS']:
    if name in fonts:
        plt.rcParams['font.family'] = name
        break
plt.rcParams.update({'font.size': 11, 'axes.unicode_minus': False, 'svg.fonttype': 'path'})
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.3), sharey=True)
fig.patch.set_facecolor('#fbfcfe')
colors = ['#7196b4'] * 5 + ['#d48445', '#397a71', '#a4bcb7']
for ax, panel, title in zip(axes, ['main12', 'extra12'], ['主开发面板 · 12场', '额外开发面板 · 12场']):
    selected = [next(r for r in rows if r['model'] == m and r['version'] == versions.get(m, 'published') and r['panel'] == panel) for m in models]
    assert all(r['total'] == 768 and r['episodes'] == 12 for r in selected)
    values = [r['delivered'] / r['total'] * 100 for r in selected]
    ax.barh(range(len(models)), values, height=.62, color=colors)
    for y, (value, row) in enumerate(zip(values, selected)):
        ax.text(value + 1.1, y, f"{value:.2f}%  ({row['delivered']}/768)", va='center', fontsize=9.5, color='#263443')
    ax.set_xlim(0, 123)
    ax.set_xticks([0, 20, 40, 60, 80, 100], ['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.set_yticks(range(len(models)), ['MAPPO', 'DQN', 'VDN', 'QMIX', 'QPLEX', 'MAT · 近距PPO', 'V44 default', 'V44 pure'])
    ax.set_title(title, loc='left', fontsize=13, pad=16, fontweight='bold')
    ax.set_xlabel('人数送达率 RCR', labelpad=10)
    ax.set_facecolor('#fbfcfe')
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color='#e2e7ee', linewidth=.7)
    ax.tick_params(axis='both', length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
axes[0].invert_yaxis()
fig.suptitle('HYRA 已发布模型｜现行任务规则', x=.035, y=.98, ha='left', fontsize=20, fontweight='bold', color='#263443')
fig.text(.035, .905, '64人 / 场  ·  720秒个人窗口  ·  1米连续10秒投放  ·  人员出界失效', fontsize=11, color='#566678')
fig.text(.035, .05, '固定发布版本；五基线采用固定代表种子。开发面板并非独立确认集，不展示未经估计的误差条。', fontsize=10, color='#566678')
fig.text(.035, .02, '来源：evidence/comparison.json；训练预算、数据及设备不同。主面板与额外面板分别报告。', fontsize=10, color='#566678')
fig.subplots_adjust(left=.14, right=.99, bottom=.18, top=.82, wspace=.10)
for suffix in ['png', 'svg']:
    fig.savefig(HERE / f'model-comparison.{suffix}', dpi=200, facecolor=fig.get_facecolor())
print('Wrote model-comparison.png and model-comparison.svg')
