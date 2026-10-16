"""Fresh metric-study families with independent nominal and reference branches."""
from pathlib import Path
import copy
import hashlib
import json
import numpy as np


def draw_metric_family(design,split,index):
    if split not in design['split_ids']:raise ValueError('Unknown split')
    seed=[design['master_seed'],design['seed_namespace'],design['split_ids'][split],index]
    rng=np.random.default_rng(np.random.SeedSequence(seed))
    params={key:float(rng.uniform(*limits)) for key,limits in design['hydraulic_parameters_uniform'].items()}
    factor=float(rng.uniform(*design['nominal_factor_uniform']))
    return dict(family_id=f'metric_v2_{split}_{index:03}',split=split,index=index,seed_identity=seed,
                params=params,nominal_specifications=[dict(mode='M0',factor=factor)])


def independent_restart(base_info,raw,state_time,source_kind):
    if source_kind=='same_mesh_independent_nominal_base' and base_info['kind']!='nominal_base':
        raise ValueError('Nominal prediction cannot initialize from reference')
    if source_kind=='reference_base_environment_only_not_forecast' and base_info['kind']!='truth_base':
        raise ValueError('Reference branch requires its own reference base')
    if source_kind not in ('same_mesh_independent_nominal_base','reference_base_environment_only_not_forecast'):
        raise ValueError('Unknown state provenance')
    ids=np.flatnonzero(raw['time_s']==state_time)
    if len(ids)!=1:raise ValueError('Missing exact saved state')
    i=int(ids[0])
    return dict(state={k:raw[k][i] for k in ('stage','xmomentum','ymomentum')},source_kind=source_kind,
                source_path=base_info['raw_path'],source_sha256=base_info['raw_sha256'],state_time_s=state_time)


def sha_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_manifest(path):
    path=Path(path);data=json.loads(path.read_text(encoding='utf-8'))
    for info in data['fields'].values():
        for p,h in [('raw_path','raw_sha256'),('projected_path','projected_sha256')]:
            if sha_file(info[p])!=info[h]:raise ValueError('Family field checksum mismatch')
    return data
