from collections import Counter
from pathlib import Path
import pytest
from hyra_exp3.execution_plan import register
from hyra_exp3.freeze import formal_matrix
from hyra_exp3 import formal
from hyra_exp3 import freeze
from hyra_exp3.artifacts import write_json,sha256


ROOT=Path(__file__).resolve().parents[1]


def test_original_formal_matrix_has_exact_paired_groups_without_development_phases():
    plan=register(ROOT);matrix=formal_matrix(plan)
    assert len(matrix)==len({r['id'] for r in matrix})==3024
    assert Counter((r['side'],r['delay_s']) for r in matrix)=={(32,0.):648,(50,0.):1080,(50,10.):648,(70,0.):648}
    for family in (f for f in plan['allocation'] if f['split']=='test'):
        rows=[r for r in matrix if r['family_id']==family['family_id']]
        assert len(rows)==84 and all(r['hydrology']==family['hydrology'] for r in rows)
    assert not any('phase' in r for r in matrix)


def test_test_registration_is_exclusive_and_only_exact_resume_is_accepted(tmp_path,monkeypatch):
    frozen=tmp_path/'frozen.json';write_json(frozen,{'frozen':True})
    run=tmp_path/'results/runs/registered';run.mkdir(parents=True)
    monkeypatch.setattr(formal,'verify_frozen',lambda *_: {'formal_matrix':[]})
    monkeypatch.setattr(formal,'new_run',lambda *_args,**_kwargs:run)
    original,_=formal.reserve_test(tmp_path,frozen)
    assert original==run
    with pytest.raises(ValueError,match='already registered'):
        formal.reserve_test(tmp_path,frozen)
    assert formal.reserve_test(tmp_path,frozen,run)[0]==run
    with pytest.raises(ValueError,match='originally registered'):
        formal.reserve_test(tmp_path,frozen,tmp_path/'results/runs/other')
    write_json(frozen,{'changed':True})
    with pytest.raises(ValueError,match='originally registered'):
        formal.reserve_test(tmp_path,frozen,run)


def test_numerical_test_fields_are_blocked_until_primary_completed(tmp_path,monkeypatch):
    frozen=tmp_path/'frozen.json';write_json(frozen,{'frozen':True})
    run=tmp_path/'results/runs/primary';run.mkdir(parents=True)
    write_json(tmp_path/'registrations/formal_test.v1.json',dict(run=str(run),frozen_sha256=sha256(frozen)))
    monkeypatch.setattr(formal,'verify_frozen',lambda *_: {})
    write_json(run/'formal_summary.json',dict(status='partial',episodes=3000))
    with pytest.raises(ValueError,match='only after'):
        formal._field_guard(tmp_path,tmp_path/'other',frozen,'numerical_replay')
    with pytest.raises(ValueError,match='single registered'):
        formal._field_guard(tmp_path,tmp_path/'other',frozen,'primary')
    write_json(run/'formal_summary.json',dict(status='completed',episodes=3024))
    formal._field_guard(tmp_path,tmp_path/'other',frozen,'numerical_replay')


def test_partial_test_summary_rejects_duplicate_and_unregistered_episodes():
    spec=dict(id='one',side=50,delay_s=0.)
    frozen=dict(formal_matrix=[spec],B_star='H')
    with pytest.raises(AssertionError,match='Duplicate or unregistered'):
        formal.summarize_primary([spec,spec],frozen)
    with pytest.raises(AssertionError,match='Duplicate or unregistered'):
        formal.summarize_primary([dict(spec,id='other')],frozen)


def test_freeze_seal_detects_file_changes_and_B_star_must_match_validation(tmp_path,monkeypatch):
    plan_path=tmp_path/'plan.json';write_json(plan_path,{})
    selection_path=tmp_path/'selection.json';selection=dict(B_star='H');write_json(selection_path,selection)
    matrix=[dict(id='fixture')]
    monkeypatch.setattr(freeze,'execution_hashes',lambda _: {})
    monkeypatch.setattr(freeze,'register',lambda _: {})
    monkeypatch.setattr(freeze,'formal_matrix',lambda _:matrix)
    config=dict(configuration_frozen=True,formal_test_enabled=True,test_results_used_for_freeze=False,
        plan_path=str(plan_path),plan_sha256=sha256(plan_path),source_hashes={},
        evidence=dict(B_star=dict(path=str(selection_path),sha256=sha256(selection_path))),checkpoints=[],
        B_star='H',B_star_selection=selection,formal_matrix=matrix,hydraulic_mesh_m=5.,hydraulic_output_interval_s=1.,
        mission_start_s=0.,mission_duration_s=1200.,noise_mode='counter_keyed')
    path=tmp_path/'frozen.json';write_json(path,config)
    seal=tmp_path/'registrations/frozen_configuration.v1.json'
    write_json(seal,dict(path=str(path),sha256=sha256(path),before_independent_test_registration=True))
    assert freeze.verify_frozen(tmp_path,path)['B_star']=='H'
    write_json(path,dict(config,B_star='E'))
    with pytest.raises(AssertionError,match='pre-test seal'):
        freeze.verify_frozen(tmp_path,path)
    write_json(seal,dict(path=str(path),sha256=sha256(path),before_independent_test_registration=True))
    with pytest.raises(AssertionError,match='single validation selection'):
        freeze.verify_frozen(tmp_path,path)
