from hyra_exp3.artifacts import new_run,finish_run,sha256
from hyra_exp3.protocol import load_protocol


def test_script_entrypoint_is_snapshotted_and_formal_run_state_is_preserved(tmp_path):
    script=tmp_path/'scripts/entry.py';script.parent.mkdir();script.write_text('print("entry")\n')
    run=new_run(tmp_path,'formal_test',{},formal_test=True)
    assert load_protocol(run/'source_sha256.json')['scripts/entry.py']==sha256(script)
    finish_run(tmp_path,run,formal_test=True)
    state=load_protocol(run/'state.json')
    assert state['formal_test'] and state['phase']=='formal_test' and 'started_at' in state
