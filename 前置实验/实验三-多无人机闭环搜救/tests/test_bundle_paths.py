from pathlib import Path
import os
import shutil
import subprocess
import sys
import json
from hyra_exp3.protocol import relocate_paths


def test_exported_evidence_paths_relocate_without_confusing_a_sibling_folder():
    old=r'D:\study';new=Path('/new/project')
    value=dict(path=r'D:\study\results\one.json',other=r'D:\study-other\two.json',list=[r'd:\study\model.pt',3])
    result=relocate_paths(value,old,new)
    assert result['path']==str(new/'results/one.json')
    assert result['other']==value['other']
    assert result['list']==[str(new/'model.pt'),3]
    assert value['path']==r'D:\study\results\one.json'


def test_all_bundle_entrypoints_activate_relocation_before_reading_records(tmp_path):
    shutil.copy2(Path(__file__).resolve().parents[1]/'run.py',tmp_path/'run.py')
    origin=r'D:\original-experiment'
    (tmp_path/'BUNDLE_ORIGIN.json').write_text(json.dumps(dict(original_root=origin)),encoding='utf-8')
    environment=dict(os.environ);environment.pop('HYRA3_RELOCATE_FROM',None)
    result=subprocess.run([sys.executable,'-c','import run,os; print(os.environ["HYRA3_RELOCATE_FROM"])'],
        cwd=tmp_path,env=environment,capture_output=True,text=True,check=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert result.stdout.strip()==origin
