"""Validate training-only auxiliary labels before any auxiliary policy fitting."""
import os
os.environ['HYRA_FLOOD_BACKEND']='numpy'
from hashlib import sha256
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parent/'terrain-workbench/terrain-lab'))
from hyra_learning.runtime import enable_existing_packages
enable_existing_packages()
import numpy as np
import torch
from flood_solver import FloodSolver
from multilayer import BaseMap
from hyra_learning.physics import SWEOperator
from hyra_learning.sciml_data import read,write,digest,preparation_sources


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--include-holdout',action='store_true')
    args=parser.parse_args();root=ROOT/'generated/sciml-round1'
    plan=read(root/'data-plan.json');audit_plan=read(ROOT/'learning_configs/sciml_data_audit_20261025.json')
    assert plan['generation_sources']==preparation_sources()
    entries=read(root/'index.json')
    assert len(entries)==(21 if args.include_holdout else audit_plan['expected_preconfirmation_cases'])
    expected=[e for e in plan['cases'] if args.include_holdout or e['split']!='holdout']
    assert {e['id'] for e in entries}=={e['id'] for e in expected}
    if args.include_holdout:
        freeze=read(root/'model-freeze.json')
        assert len(freeze['selected_models'])==12
        for row in freeze['selected_models']:assert digest(root/row['path'])==row['sha256']
        assert digest(root/'training-index-frozen.json')==freeze['training_input_sha256']
    else:assert not any(e['split']=='holdout' for e in entries)
    records=[];parents={};control=None
    torch.set_num_threads(1)
    for entry in entries:
        path=Path(entry['path']);aux=Path(entry['auxiliary_path'])
        assert digest(path)==entry['source_sha256'] and digest(aux)==entry['auxiliary_sha256']
        with zipfile.ZipFile(path) as archive,np.load(aux,allow_pickle=False) as packed:
            data={key:packed[key] for key in packed.files}
            process,model=(json.loads(archive.read(name)) for name in ('process.json','map_model.json'))
            assert len(process['frames'])==121 and data['U'].shape==(121,3,32,32)
            assert data['rate'].shape==(60,3,32,32) and data['source'].shape==(60,32,32)
            assert np.isfinite(data['U']).all() and np.isfinite(data['rate']).all()
            assert np.all(data['U'][:,0]>=0) and np.all((data['dt']>0)&(data['dt']<=.01))
            bed_hash=sha256(data['bed'].tobytes()).hexdigest()
            assert entry['bed_sha256']==bed_hash
            if entry['family'] in parents:assert parents[entry['family']]==bed_hash
            parents[entry['family']]=bed_hash
            for i,frame in enumerate(process['frames']):
                assert frame['time_s']==5*i
                raw=np.frombuffer(archive.read(frame['url'].rsplit('/',1)[-1]),dtype='<f4').reshape(4,32,32)
                np.testing.assert_array_equal(raw[0],data['U'][i,0])
                np.testing.assert_allclose(raw[0]*raw[1],data['U'][i,1],rtol=1e-6,atol=1e-7)
                np.testing.assert_allclose(raw[0]*raw[2],data['U'][i,2],rtol=1e-6,atol=1e-7)
            base=BaseMap(data['bed'],5,dict(model['M_base'],hydrographs=model['source_hydrographs']),.035)
            reference=FloodSolver(base.z,np.zeros_like(base.z),5,boundary='closed')
            base.configure(reference)
            expected_sources=[]
            for now in range(0,600,10):expected_sources.append(reference._scheduled_rate(now))
            np.testing.assert_allclose(data['source'],np.stack(expected_sources),rtol=1e-6,atol=1e-8)
            errors=[]
            for t in range(60):
                current=torch.as_tensor(data['U'][2*t:2*t+1],dtype=torch.float64)
                evolved=SWEOperator().step(current,torch.as_tensor(data['bed']),float(data['dt'][t]),
                    source=torch.as_tensor(data['source'][t],dtype=torch.float64),
                    drains=torch.as_tensor(data['drains'],dtype=torch.float64),manning=torch.as_tensor(data['manning'],dtype=torch.float64),
                    solid=torch.as_tensor(data['solid']),wall_x=torch.as_tensor(data['wall_x']),wall_y=torch.as_tensor(data['wall_y']))
                rate=((evolved-current)/data['dt'][t])[0].numpy()
                errors.append(float(np.abs((rate-data['rate'][t])/np.array([.1,.05,.05])[:,None,None]).mean()))
            assert max(errors)<=audit_plan['normalized_rate_mean_absolute_error_max']
            wet=data['U'][:,0]>=.05
            speed=np.sqrt(data['U'][:,1]**2+data['U'][:,2]**2)/np.maximum(data['U'][:,0],1e-6)
            records.append(dict(id=entry['id'],split=entry['split'],frames_checked=121,rate_labels_checked=60,
                                 normalized_rate_mae_max=max(errors),max_depth_m=float(data['U'][:,0].max()),
                                 wet_max_speed_m_s=float(speed[wet].max()) if wet.any() else None,
                                 mass_residual_max_m3=entry['mass_residual_max_m3'],passed=True))
            if entry['id']==audit_plan['instrumentation_control']['case']:
                depths=[]
                for now in range(0,601,5):
                    reference.advance(now)
                    depths.append(reference.h.copy())
                depths=np.stack(depths)
                mae=float(np.abs(depths-data['U'][:,0]).mean(axis=(1,2)).max())
                disagreement=float(((depths>=.05)!=(data['U'][:,0]>=.05)).mean(axis=(1,2)).max())
                assert mae<=audit_plan['instrumentation_control']['max_frame_mean_depth_absolute_error_m']
                assert disagreement<=audit_plan['instrumentation_control']['max_frame_wet_label_disagreement']
                control=dict(case=entry['id'],max_frame_depth_mae_m=mae,max_frame_wet_disagreement=disagreement,passed=True)
        print('audited',entry['id'],flush=True)
    assert len(parents)==len(set(parents.values()))==(7 if args.include_holdout else 4) and control is not None
    for split in ('fit','selection','holdout'):
        families={e['family'] for e in entries if e['split']==split}
        assert families==set(plan['design']['terrain_families'][split]) if (split!='holdout' or args.include_holdout) else not families
    output=ROOT/'docs/evidence/20261025-sciml-development'
    write(output/('data-audit-final.json' if args.include_holdout else 'data-audit.json'),dict(status='passed',cases=len(entries),parents=len(parents),records=records,
        instrumentation_control=control,confirmation_water_generated=args.include_holdout,
        auditor_sha256=digest(Path(__file__)),audit_plan_sha256=digest(ROOT/'learning_configs/sciml_data_audit_20261025.json')))
    print('PASSED',len(entries),'events; confirmation included:',args.include_holdout)


if __name__=='__main__':main()
