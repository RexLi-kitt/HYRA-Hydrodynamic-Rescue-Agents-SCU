"""Verify equal budgets, validation-only checkpoint choice and training-only links."""
from pathlib import Path
import numpy as np
import torch
from .artifacts import sha256,write_json
from .protocol import load_protocol
from .execution_plan import register
from .models import METHODS


def verify_model_budget(row,recipe):
    if not row['trained'] or row['test_accessed'] or row['distinct_training_families']!=24 or row['validation_families']!=12:
        raise AssertionError('Model training/split identity differs')
    if [r['step'] for r in row['training_history']]!=list(range(100,recipe['steps']+1,100)):
        raise AssertionError('Unequal or incomplete optimization budget')
    if [c['step'] for c in row['checkpoints']]!=recipe['checkpoint_steps']:
        raise AssertionError('Unequal validation checkpoint schedule')
    if not all(np.isfinite(c['validation_mse_m2']) for c in row['checkpoints']):raise AssertionError('Nonfinite validation loss')
    selected=min(row['checkpoints'],key=lambda c:(c['validation_mse_m2'],c['step']))
    if row['selected']!=selected:raise AssertionError('Checkpoint did not follow the frozen validation-only selection rule')
    if not all(np.isfinite(r['training_mse_m2']) for r in row['training_history']):raise AssertionError('Nonfinite training loss')


def audit_training(root,training_run):
    from .training import RECIPE
    root,training_run=Path(root),Path(training_run);plan=register(root)
    summary=load_protocol(training_run/'training_summary.json');recipe=load_protocol(training_run/'training_recipe.json')
    if summary['status']!='completed' or summary['test_accessed'] or len(summary['models'])!=15 or recipe!=RECIPE:
        raise ValueError('Complete prespecified equal-budget training matrix required')
    expected={(m,s) for m in METHODS for s in (101,202,303)};seen=set();shape_reference=None;files=0
    for model in summary['models']:
        identity=model['method'],model['seed']
        if identity in seen or identity not in expected:raise AssertionError('Duplicate or unregistered trained model')
        seen.add(identity);verify_model_budget(model,recipe)
        if load_protocol(training_run/'models'/f'{identity[0]}_seed{identity[1]}'/'training_result.json')!=model:
            raise AssertionError('Training model summary differs from its raw result')
        for info in model['checkpoints']:
            if sha256(info['path'])!=info['sha256']:raise AssertionError('A trained checkpoint changed')
            checkpoint=torch.load(info['path'],map_location='cpu',weights_only=True)
            if (checkpoint['method'],checkpoint['seed'],checkpoint['step'],checkpoint['hidden'])!=(*identity,info['step'],recipe['hidden']):
                raise AssertionError('Checkpoint identity differs from the training record')
            state=checkpoint['state_dict'];shapes={k:list(v.shape) for k,v in state.items()}
            if shape_reference is None:shape_reference=shapes
            if shapes!=shape_reference or sum(v.numel() for v in state.values())!=37070 or model['parameter_count']!=37070:
                raise AssertionError('Five-method model capacities differ')
            if not all(torch.isfinite(v).all().item() for v in state.values()):raise AssertionError('Nonfinite trained model weights')
            files+=1
    if seen!=expected:raise AssertionError('Training method/seed matrix incomplete')
    bank=load_protocol(training_run/'frozen_training_links.json')
    families={f['family_id']:f for f in plan['allocation'] if f['split']=='train'}
    fit=set(bank['fit_families']);held=set(bank['held_out_TRAIN_families'])
    if fit&held or fit|held!=set(families) or len(fit)!=18 or len(held)!=6 or bank['validation_or_test_used']:
        raise AssertionError('Link fitting/held-out training-family isolation differs')
    if bank['top_k']!=8 or bank['P_min_improving_held_out_train_families']!=4 or bank['lag_seconds']!=list(range(0,61,5)):
        raise AssertionError('Registered sparse-link lag/validation settings differ')
    cases=set()
    for record in bank['source_cases']:
        if sha256(record['path'])!=record['sha256']:raise AssertionError('Link-fitting source case changed')
        case=load_protocol(record['path']);identity=case['family']['family_id'],case['side']
        if identity in cases or identity[0] not in families or case['family']!=families[identity[0]]:
            raise AssertionError('A nontraining/duplicate family entered link fitting')
        cases.add(identity)
    if cases!={(f,s) for f in families for s in (32,50,70)}:raise AssertionError('Incomplete training-only link source matrix')
    for side in ('32','50','70'):
        arrays={k:np.asarray(v) for k,v in bank['banks'][side].items()}
        if any(v.shape!=(16,16) or not np.isfinite(v).all() for v in arrays.values()):raise AssertionError('Invalid link-score/lag matrix')
        lag=arrays['lag_steps'];counts=arrays['held_out_train_improving_families']
        if np.any((lag<0)|(lag>12)|(lag!=np.floor(lag))) or np.any((counts<0)|(counts>6)|(counts!=np.floor(counts))):
            raise AssertionError('Invalid link lag or held-out-family validation count')
    result=dict(status='passed',trained_models=15,parameter_count=37070,checkpoint_files=files,training_steps_per_model=recipe['steps'],
        identical_validation_checkpoint_schedule=True,validation_only_checkpoint_selection_recomputed=True,
        distinct_training_families=24,link_fit_families=18,link_validation_TRAIN_families=6,training_link_cases=72,
        test_accessed=False,summary_sha256=sha256(training_run/'training_summary.json'),
        recipe_sha256=sha256(training_run/'training_recipe.json'),link_bank_sha256=sha256(training_run/'frozen_training_links.json'))
    write_json(training_run/'training_integrity_audit.json',result)
    return result
