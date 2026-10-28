"""Recheck archived actual-frame replays without modifying or regenerating them."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hyra_uav.core import Config,FloodProcess,Memory,Pose,footprint,move
from hyra_uav.experiments import write_json


def main(source=None,output=None,expected=81):
    formal=Path(source) if source else ROOT/'generated/p2-formal-20261025'
    processes={}
    for path in list((ROOT/'fixtures').glob('*/map-result.zip'))+list((formal/'scenes').glob('*/map-result.zip')):
        process=FloodProcess(path);processes[process.digest]=process
    results=[]
    replay_folder=formal/('runs' if (formal/'runs').exists() else 'replays')
    for path in sorted(replay_folder.glob('*/replay.json')):
        replay=json.loads(path.read_text(encoding='utf-8'));process=processes[replay['source_sha256']]
        config=Config(**replay['config']);memory=Memory(process.shape,config.history_length)
        previous=None;distance=0.;turn=0.;clipped=0.;observed_cells=0
        assert len(replay['snapshots'])==len(replay['observations'])
        for snapshot,observation,truth in zip(replay['snapshots'],replay['observations'],replay['truth']):
            now=snapshot['time_s'];index=snapshot['frame_index'];pose=Pose(**{k:snapshot[k] for k in ('x_m','z_m','heading_deg')})
            assert process.times[index]==now
            np.testing.assert_array_equal(np.asarray(truth).reshape(process.shape),process.labels[index])
            cells=np.flatnonzero(footprint(pose,config,process.shape,process.dx).ravel())
            np.testing.assert_array_equal(cells,observation['cells'])
            np.testing.assert_array_equal(process.labels[index].ravel()[cells],observation['labels'])
            assert observation['observed_s']==observation['arrived_s']==now
            memory.accept(observation,now);observed_cells+=len(cells)
            np.testing.assert_array_equal(memory.labels.ravel(),snapshot['labels'])
            np.testing.assert_array_equal(memory.observed_s.ravel(),snapshot['observed_s'])
            if previous:
                dt=now-previous['time_s'];before=Pose(**{k:previous[k] for k in ('x_m','z_m','heading_deg')})
                predicted,increment=move(before,snapshot['action'],dt,config,process.shape,process.dx)
                np.testing.assert_allclose([predicted.x_m,predicted.z_m,predicted.heading_deg],[pose.x_m,pose.z_m,pose.heading_deg],atol=1e-9)
                distance+=increment
                if snapshot['action'] in ('left','right'):turn+=dt
                if snapshot['action']=='forward':clipped+=max(0,dt-increment/config.speed_m_s)
            previous=snapshot
        np.testing.assert_allclose(distance,replay['summary']['distance_m'],atol=1e-8)
        np.testing.assert_allclose(turn,replay['summary']['turn_s'],atol=1e-8)
        assert len(replay['actions'])==60 and previous['time_s']==600
        results.append(dict(run_id=path.parent.name,method=replay['method'],scene=replay['scene']['id'],
                            checks_passed=True,samples=len(replay['snapshots']),observed_cells=observed_cells,
                            distance_m=distance,turn_s=turn,clipped_forward_s=clipped,
                            first_90_percent_coverage_s=replay['summary']['first_90_percent_coverage_s'],
                            final_first_coverage=replay['summary']['final']['first_coverage']))
    assert len(results)==expected
    groups=[]
    for method in ('R0','R1','R2'):
        rows=[r for r in results if r['method']==method]
        groups.append(dict(method=method,runs=len(rows),mean_distance_m=float(np.mean([r['distance_m'] for r in rows])),
                           mean_turn_s=float(np.mean([r['turn_s'] for r in rows])),
                           mean_clipped_forward_s=float(np.mean([r['clipped_forward_s'] for r in rows])),
                           reached_90=sum(r['first_90_percent_coverage_s'] is not None for r in rows)))
    output=Path(output) if output else ROOT/'docs/evidence/20261025-final-engineering'
    write_json(output/'replay-audit.json',dict(status='passed',runs=results,groups=groups,
        checks=['actual truth and observation values','frame timestamps','directional footprints','memory updates',
                'every intermediate pose','actual distance and paid rotations','complete action and sampling clocks']))
    print(json.dumps(groups,ensure_ascii=False))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--expected',type=int,default=81);args=parser.parse_args()
    main(args.source,args.output,args.expected)
