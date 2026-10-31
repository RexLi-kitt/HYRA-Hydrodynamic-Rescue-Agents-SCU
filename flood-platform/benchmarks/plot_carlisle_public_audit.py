"""Scientific input audit figures, never model predictions or acceptance plots."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_carlisle_public_inputs import read_bdy


def plot(outputs):
    outputs = Path(outputs)
    r = outputs / 'round72'
    prepared = r / 'prepared'
    a = np.load(prepared / 'terrain-audit.npz')
    s = json.loads((prepared / 'public-input-audit.json').read_text())
    extent = s['terrain']['extent_m']
    ext = [extent[0], extent[2], extent[1], extent[3]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), constrained_layout=True)
    for ax, value, title in zip(axes, [a['terrain5'], a['terrain15'], a['terrain15'] - a['means5_to15']],
          ['Public repository: 5 m GeoTIFF', 'Zenodo example: 15 m ASCII', '15 m minus block-mean of 5 m']):
        difference = ax is axes[2]
        im = ax.imshow(value, extent=ext, origin='upper', cmap='RdBu_r' if difference else 'terrain',
                       vmin=-10 if difference else 5, vmax=10 if difference else 42)
        ax.set_title(title)
        ax.set_xlabel('Easting (m)');ax.set_ylabel('Northing (m)')
        ax.ticklabel_format(style='plain', useOffset=False)
        fig.colorbar(im, ax=ax, shrink=.72, label='Elevation difference (m)' if difference else 'Stored elevation (m)')
    for point in s['terrain']['source_point_samples']:
        axes[0].plot(point['x_m'], point['y_m'], 'rx', markersize=7)
        axes[0].annotate(point['name'], (point['x_m'], point['y_m']), xytext=(4, 6), textcoords='offset points', fontsize=8)
    fig.suptitle('Carlisle input geometry only — no flood simulation; red crosses are repository source points')
    fig.savefig(prepared / 'terrain-source-audit.png', dpi=160)
    plt.close(fig)
    old = json.loads((outputs / 'round71/carlisle/source-contract.json').read_text())['hydrographs']
    new = read_bdy(r / 'flooda5-inputs/carlisle.bdy')
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    for ax, item in zip(axes, s['source_hydrograph_comparisons']):
        h = next(h for h in old if h['name'] == item['old_name'])
        t = np.array(h['time_since_model_start_s']) / 3600
        ax.plot(t, h['physical_total_Q_m3_s'], label='T07: 900 s input spacing', color='#176c8c')
        n = new[item['new_name']]
        ax.plot(n['time_s'] / 3600, n['q_as_stored'], label='Public example: 1800 s input spacing', color='#d16b2e')
        ax.set_title(item['river']); ax.set_ylabel('Input discharge (m3/s)'); ax.set_xlabel('Hours since each file starts (different, unassigned epochs)')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Input hydrographs on their ORIGINAL time axes — no alignment to observations')
    fig.savefig(prepared / 'hydrograph-source-audit.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--outputs', required=True)
    plot(p.parse_args().outputs)
