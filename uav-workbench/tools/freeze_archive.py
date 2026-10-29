"""Preserve exact executable study sources and local runtime provenance."""
import argparse
from hashlib import sha256
import json
import platform
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from hyra_uav.experiments import write_json
from hyra_uav.strong_study import dependencies,ensure_sources


def main(root):
    plan=json.loads((root/'plan.json').read_text(encoding='utf-8'));ensure_sources(plan)
    archive=root/'frozen-sources.zip';manifest=root/'runtime-provenance.json'
    if archive.exists() or manifest.exists():raise ValueError('保留已有源码归档，不能覆盖')
    sklearn,joblib,_=dependencies();import scipy,numpy
    files={('uav-workbench/'+name.replace('\\','/')):ROOT/name for name in plan['version']['source_files_sha256']}
    for name in ('flood_solver.py','flood_run.py','multilayer.py','map_sessions.py'):
        files['terrain-workbench/terrain-lab/'+name]=ROOT.parent/'terrain-workbench/terrain-lab'/name
    for name in ('requirements.txt','research-requirements.txt','report-requirements.txt'):
        files['uav-workbench/'+name]=ROOT/name
    hashes={name:sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as output:
        for name,path in files.items():output.writestr(name,path.read_bytes())
    write_json(manifest,dict(schema='hyra-study-provenance-v1',plan_sha256=sha256((root/'plan.json').read_bytes()).hexdigest(),
        source_zip_sha256=sha256(archive.read_bytes()).hexdigest(),files_sha256=hashes,
        platform=platform.platform(),python=sys.version,packages=dict(numpy=numpy.__version__,scipy=scipy.__version__,
        sklearn=sklearn.__version__,joblib=joblib.__version__),
        limitation='hydraulic dependency hashes recorded during fit before holdout generation; original baseline Git files are unchanged'))
    print('archived',len(files),'files')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('study',type=Path);main(parser.parse_args().study)
