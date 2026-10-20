from pathlib import Path
import argparse
import json
import os
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--audit-primary',action='store_true')
    p.add_argument('--workers',type=int,default=3)
    a=p.parse_args()
    from hyra_exp3.export import verify_bundle
    print(verify_bundle(ROOT),flush=True)
    os.environ['HYRA3_RELOCATE_FROM']=json.loads((ROOT/'BUNDLE_ORIGIN.json').read_text(encoding='utf-8'))['original_root']
    from hyra_exp3.freeze import verify_frozen
    frozen=ROOT/'configs/frozen_experiment3.v1.json'
    verify_frozen(ROOT,frozen)
    print('Frozen configuration, model weights and pre-test evidence verified',flush=True)
    if a.audit_primary:
        from hyra_exp3.formal_audit import audit_formal
        from hyra_exp3.protocol import load_protocol
        print(audit_formal(ROOT,load_protocol(ROOT/'registrations/formal_test.v1.json')['run'],frozen,a.workers),flush=True)
