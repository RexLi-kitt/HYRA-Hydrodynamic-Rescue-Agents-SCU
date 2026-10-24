import pytest
from hyra_exp3.artifacts import sha256,write_json
from hyra_exp3.export import verify_bundle,verify_record_references


def test_portable_manifest_rejects_modified_raw_evidence_and_escape_paths(tmp_path):
    raw=tmp_path/'raw.json';raw.write_text('{"services": 2}',encoding='utf-8')
    entry=dict(sha256=sha256(raw),bytes=raw.stat().st_size)
    write_json(tmp_path/'INTEGRITY.json',dict(files={'raw.json':entry}))
    assert verify_bundle(tmp_path)['status']=='passed'
    raw.write_text('{"services": 3}',encoding='utf-8')
    with pytest.raises(AssertionError,match='integrity failed'):
        verify_bundle(tmp_path)
    write_json(tmp_path/'INTEGRITY.json',dict(files={'../outside.json':entry}))
    with pytest.raises(AssertionError,match='escapes package'):
        verify_bundle(tmp_path)


def test_export_checks_original_recorded_hashes_before_signing_a_new_copy(tmp_path):
    raw=tmp_path/'depth.npz';raw.write_bytes(b'original raw field')
    proof=tmp_path/'case.json'
    write_json(proof,dict(manifest=dict(field_path=str(raw),field_sha256=sha256(raw))))
    assert verify_record_references(tmp_path,{proof,raw})['unique_files']==1
    raw.write_bytes(b'changed raw field')
    with pytest.raises(AssertionError,match='before export'):
        verify_record_references(tmp_path,{proof,raw})
