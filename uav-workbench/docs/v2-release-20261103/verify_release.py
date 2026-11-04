"""Read-only archive/weights/statistics verification; no training or hydraulics."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    manifest=read(HERE/'manifest.json')
    for item in manifest['included']:
        path=(REPO/item['path']).resolve()
        assert path.is_relative_to(REPO.resolve()),item['path']
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],item['path']
    sys.path.insert(0,str(REPO/'uav-workbench/experiments/flood-drop-frozen-validation-v1'))
    import validate as v
    archive=HERE/'evidence/uav-frozen-validation-20261103/run01'
    protocol=read(archive/'protocol.json');expected=read(archive/'summary.json')
    assert sha(HERE/'model/gru-32.pt')==protocol['checkpoint_sha']
    ck=torch.load(HERE/'model/gru-32.pt',map_location='cpu',weights_only=True)
    model=v.f.Model(ck['kind'],ck['mean']);model.load_state_dict(ck['state']);model.eval()
    assert np.isfinite(ck['mu']).all() and np.isfinite(ck['sd']).all() and (np.asarray(ck['sd'])>0).all()
    rows=[json.loads(line) for line in (archive/'results.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(rows)==180 and len({(r['case'],r['method']) for r in rows})==180
    for r in rows:
        assert len(r['persons'])==r['people']==40
        assert r['supplied']+r['expired']+r['outside']==40
        assert sum(p['status']==2 for p in r['persons'])==r['supplied']
        for person in r['persons']:
            if person['status']==2:assert person['fall']+10<=person['supplied_at']<=person['deadline']
    for case in {r['case'] for r in rows}:
        group=[r for r in rows if r['case']==case];assert len(group)==3
        assert all([(p['id'],p['fall'],p['deadline']) for p in r['persons']]==[(p['id'],p['fall'],p['deadline']) for p in group[0]['persons']] for r in group)
    for panel in ('main','rotate90','mirror_x','orientation_combined'):
        subset=[r for r in rows if r['panel']==panel] if panel!='orientation_combined' else [r for r in rows if r['panel']!='main']
        actual=v.statistics(subset);ref=expected['statistics'][panel]
        assert actual['results']==ref['results'] and actual['point_goal_met']==ref['point_goal_met']
        assert actual['bootstrap_goal_supported']==ref['bootstrap_goal_supported']
        for metric in ('people_rate','all_rate','p95_rate'):
            np.testing.assert_allclose(actual['gaps'][metric]['ci95'],ref['gaps'][metric]['ci95'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(actual['gaps'][metric]['estimate'],ref['gaps'][metric]['estimate'],rtol=0,atol=1e-12)
    print(json.dumps(dict(status='passed',sealed_files=len(manifest['included']),complete_rollouts=len(rows),scenarios=60,
        parameters=sum(p.numel() for p in model.parameters()),main_people_rate=expected['statistics']['main']['results'][v.METHODS[2]]['people_rate'],
        rotation_goal_met=expected['statistics']['rotate90']['point_goal_met'],refloat_events=sum(r['refloat_events'] for r in rows)),indent=2))

if __name__=='__main__':main()
