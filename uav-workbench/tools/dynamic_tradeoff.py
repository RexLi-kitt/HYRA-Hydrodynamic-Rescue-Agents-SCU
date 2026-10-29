"""Describe P2 revisit/change scales from the frozen base runs, no selection."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hyra_uav.experiments import write_json


def quantiles(values):
    return dict(zip(('p10_s','median_s','p90_s'),map(float,np.quantile(values,[.1,.5,.9])))) if values else None


def main():
    source=ROOT/'generated/p2-formal-20261025/runs';events=[];runs=[]
    for mechanism in ('slow','rapid','async'):
        sample=json.loads((source/f'{mechanism}-base-R0-s0/replay.json').read_text(encoding='utf-8'))
        complete=[r['segment_end_s']-r['changed_s'] for r in sample['changes'] if r['segment_end_s']<600]
        cutoff=[r['segment_end_s']-r['changed_s'] for r in sample['changes'] if r['segment_end_s']==600]
        events.append(dict(mechanism=mechanism,complete_change_segments=len(complete),complete_lifetimes=quantiles(complete),
                           cutoff_segments=len(cutoff),restricted_cutoff_windows=quantiles(cutoff)))
        for method in ('R0','R1','R2'):
            for start in range(3):
                run_id=f'{mechanism}-base-{method}-s{start}'
                replay=json.loads((source/run_id/'replay.json').read_text(encoding='utf-8'))
                gaps=[];last={};returned=set()
                for observation in replay['observations']:
                    now=observation['observed_s']
                    for cell in observation['cells']:
                        if cell in last and now-last[cell]>replay['config']['sample_s']+1e-9:
                            gaps.append(now-last[cell]);returned.add(cell)
                        last[cell]=now
                runs.append(dict(run_id=run_id,method=method,mechanism=mechanism,start=start,
                    revisit_gaps=quantiles(gaps),return_count=len(gaps),known_cells=len(last),returned_cells=len(returned),
                    known_never_returned_cells=len(last)-len(returned),**replay['summary']))
    output=ROOT/'docs/evidence/20261025-final-engineering'
    write_json(output/'dynamic-tradeoff.json',dict(events=events,runs=runs,
        limits='One physical process per mechanism; repeated routes are not new events. Last change segments are cutoff-censored. Complete segment lifetimes apply only to cells that change again.'))
    sys.path.append(str(ROOT/'.runtime/plot-packages'));import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,4.3),sharey=True)
    colors={'R0':'#777c81','R1':'#2384ad','R2':'#098575'}
    for ax,event in zip(axes,events):
        for i,method in enumerate(colors):
            rows=[r for r in runs if r['method']==method and r['mechanism']==event['mechanism']]
            ax.bar(i-.18,np.mean([r['revisit_gaps']['median_s'] for r in rows]),width=.34,color=colors[method],label=method+' revisit median')
            ax.bar(i+.18,np.mean([r['revisit_gaps']['p90_s'] for r in rows]),width=.34,color=colors[method],alpha=.35)
        if event['complete_lifetimes']:
            ax.axhline(event['complete_lifetimes']['median_s'],color='#bc7745',ls='--',label='Complete change lifetime median')
        ax.set_title(event['mechanism']);ax.set_xticks(range(3),['R0','R1','R2']);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Physical time (s)');axes[-1].legend(fontsize=7)
    fig.suptitle('Base scale / mean of three start summaries / pale bars: revisit p90')
    fig.tight_layout();fig.savefig(output/'change-vs-revisit.png',dpi=170);plt.close(fig)
    print(json.dumps(events))


if __name__=='__main__':main()
