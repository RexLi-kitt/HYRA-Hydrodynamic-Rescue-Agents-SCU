"""Publish complete P3 results; never fit, select or alter a model."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hyra_uav.experiments import write_json


def publish(source,output):
    summary=json.loads((source/'summary.json').read_text(encoding='utf-8'))
    if summary['status']!='completed' or summary['metrics_rows']!=summary['expected_metrics_rows']:
        raise ValueError('只有完整结果可以发布')
    output.mkdir(parents=True,exist_ok=True)
    for name in ('plan.json','model-freeze.json','index.json','case-audits.json','selection.json','fit-index.json','summary.json'):
        shutil.copyfile(source/name,output/name)
    for name in ('plan-before-engineering-fix.json','engineering-amendment-validation.json','runtime-provenance.json','frozen-sources.zip'):
        if (source/name).exists():shutil.copyfile(source/name,output/name)
    for directory in ('input-audit','input-audit-final'):
        if (source/directory).exists():
            (output/directory).mkdir(exist_ok=True)
            for path in (source/directory).glob('*.json'):shutil.copyfile(path,output/directory/path.name)
    with (output/'event-metrics.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(summary['events'][0]));writer.writeheader();writer.writerows(summary['events'])
    fits=json.loads((source/'fit-index.json').read_text(encoding='utf-8'))
    selection=json.loads((source/'selection.json').read_text(encoding='utf-8'));budget=[]
    for group in ('B1','B2','B3','A1','A2'):
        rows=[r for r in fits if r['group']==group]
        budget.append(dict(group=group,models=len(rows),fit_rows=rows[0]['fit_rows'],positives=rows[0]['positives'],
                           feature_dimension=rows[0]['feature_dimension'],trees_each=rows[0]['trees'],
                           mean_nodes=float(np.mean([r['total_nodes'] for r in rows])),
                           mean_leaf_class_values=float(np.mean([r['leaf_parameters'] for r in rows])),
                           mean_split_thresholds=float(np.mean([r['total_nodes']-r['leaf_parameters']/2 for r in rows])),
                           compressed_models_bytes=sum((source/r['path']).stat().st_size for r in rows),
                           final_fit_s=sum(r['elapsed_s'] for r in rows),
                           selection_s=sum(r['elapsed_s'] for r in selection if r['group']==group),
                           capacity_definition='Tree nodes, split thresholds and stored binary leaf class values; not independent neural-network parameters'))
    write_json(output/'capacity-budget.json',budget)
    families=[]
    for family in sorted({r['family'] for r in summary['paired']}):
        rows=[r for r in summary['paired'] if r['family']==family]
        families.append(dict(family=family,events=len(rows),
            **{'delta_'+g:float(np.mean([r['delta_'+g] for r in rows])) for g in ('B1','B2','A1','A2')}))
    write_json(output/'family-differences.json',families)
    # Descriptive training-seed variation, never additional independent events.
    accum={}
    with (source/'metrics.jsonl').open(encoding='utf-8') as file:
        for line in file:
            record=json.loads(line)
            if record['band']!='all' or record['fraction']!=.1:continue
            key=(record['case'],record['group'],record['seed'],record['route'])
            totals=accum.setdefault(key,[0.,0])
            if record['precision'] is not None:totals[0]+=record['precision'];totals[1]+=1
    seed_rows=[]
    for case,group,seed in sorted({k[:3] for k in accum}):
        values=[v[0]/v[1] for k,v in accum.items() if k[:3]==(case,group,seed) and v[1]]
        seed_rows.append(dict(case=case,group=group,seed=seed,precision=float(np.mean(values)) if values else None))
    write_json(output/'seed-event-metrics.json',seed_rows)
    sys.path.append(str(ROOT/'.runtime/plot-packages'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'B1':'#60676d','B2':'#2886ae','B3':'#078573','A1':'#c78658','A2':'#906aaa'}
    fig,axes=plt.subplots(1,3,figsize=(12,4.4),sharey=True)
    for ax,mechanism in zip(axes,('slow','rapid','async')):
        for group,color in colors.items():
            values=[]
            for fraction in (.05,.1,.2):
                rows=[r for r in summary['events'] if r['mechanism']==mechanism and r['group']==group and r['fraction']==fraction and r['band']=='all']
                values.append(np.mean([r['precision'] for r in rows if r['precision'] is not None])*100)
            ax.plot([5,10,20],values,'o-',label=group,color=color)
        ax.set_title(mechanism);ax.set_xticks([5,10,20]);ax.set_xlabel('Selected old records (%)');ax.grid(alpha=.2)
    axes[0].set_ylabel('Review precision (%)');axes[-1].legend(fontsize=8)
    fig.suptitle('P3 nonlinear / equal event, route, time and seed weights');fig.tight_layout()
    fig.savefig(output/'ranking.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.7));paired=summary['paired'];x=np.arange(len(paired))
    for offset,group in ((-.18,'B1'),(.18,'B2')):
        ax.bar(x+offset,[r['delta_'+group]*100 for r in paired],width=.34,label='B3 - '+group,color=colors[group])
    ax.axhline(0,color='#777',linewidth=.8);ax.set_xticks(x,[r['case'].replace('r'+str(summary['round'])+'-holdout-','') for r in paired],rotation=35,ha='right',fontsize=8)
    ax.set_ylabel('Precision@10 difference (percentage points)');ax.legend();fig.tight_layout()
    fig.savefig(output/'paired-events.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(mean_deltas=summary['mean_deltas'],gate=summary['gate_passed'],budget=budget)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();publish(args.source,args.output)
