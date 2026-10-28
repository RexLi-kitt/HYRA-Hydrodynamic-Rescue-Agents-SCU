"""Archive the local research delivery after actual evidence passes.

No Git commit/push, no source deletion, no copying/recompressing all large
recovery files. Every retained experiment file gets a streamed hash manifest.
"""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent
AUDITS=['training','transition','trajectory','forecast','data','statistics','targets']


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def digest(path):
    h=sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*2**20),b''):h.update(chunk)
    return h.hexdigest()
def git(*args):return subprocess.check_output(['git',*args],cwd=REPO)
def identity(path):
    assert path.resolve().is_relative_to(REPO.resolve())
    before=path.stat();h=digest(path);after=path.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),path
    return dict(path=str(path.relative_to(REPO)).replace('\\','/'),bytes=after.st_size,sha256=h)


def zip_paths(target,paths):
    if target.exists():raise ValueError('Preserve delivery archive: '+str(target))
    rows=[]
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as archive:
        for path in sorted(set(paths)):
            if not path.is_file():continue
            row=identity(path);archive.write(path,row['path']);rows.append(row)
        archive.writestr('ARCHIVE_FILES.json',json.dumps(rows,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for row in rows:assert sha256(archive.read(row['path'])).hexdigest()==row['sha256']
    return dict(path=target.name,bytes=target.stat().st_size,sha256=digest(target),files=len(rows),content_sha256_verified=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'generated/delivery-20261025')
    args=parser.parse_args();out=args.output.resolve();assert out.is_relative_to(ROOT/'generated')
    if out.exists():raise ValueError('Choose a new delivery directory; never overwrite')
    evidence=ROOT/'docs/evidence/20261025-sciml-development'
    for kind in AUDITS:assert read(evidence/(kind+'-audit-final.json'))['status']=='passed',kind
    assert read(ROOT/'docs/evidence/20261025-sciml-final/visual-qa.json')['status']=='passed'
    assert read(ROOT/'docs/evidence/20261025-team-viewer/system-qa.json')['status']=='passed'
    assert read(ROOT/'docs/evidence/20261025-p3-closeout/audit.json')['status']=='passed'
    plan=read(ROOT/'generated/sciml-round1/formal-plan.json')
    for name,h in plan['sources'].items():assert digest(Path(name))==h,name
    registration=read(ROOT/'generated/sciml-round1/evaluation-plan.json')
    assert digest(ROOT/'tools/evaluate_sciml.py')==registration['evaluator_sha256']
    out.mkdir()
    baseline=git('rev-parse','HEAD').decode().strip()
    subprocess.run(['git','bundle','create',str(out/'baseline.bundle'),'HEAD'],cwd=REPO,check=True)
    subprocess.run(['git','bundle','verify',str(out/'baseline.bundle')],cwd=REPO,check=True)
    (out/'tracked-changes.patch').write_bytes(git('diff','--binary','HEAD'))
    tracked=[REPO/name.decode('utf-8') for name in git('ls-files','-z').split(b'\0') if name]
    untracked=[REPO/name.decode('utf-8') for name in git('ls-files','--others','--exclude-standard','-z').split(b'\0') if name]
    sources=[p for p in tracked+untracked if p.is_file() and not any(x in p.parts for x in ('.git','.runtime','__pycache__'))]
    print('Packaging complete current source/evidence tree:',len(sources),flush=True)
    bundles=[zip_paths(out/'source-and-evidence.zip',sources)]
    study=ROOT/'generated/sciml-round1';freeze=read(study/'model-freeze.json')
    selected=[study/r['path'] for r in freeze['selected_models']]
    selected+=list((study/'deployment').rglob('*'))
    selected+=[p for p in study.glob('*.json') if not any(s in p.name for s in ('state','lock'))]
    selected+=list(study.glob('*sources.zip'))+[study/'evaluation-frozen-tools.zip',study/'independent-audit-tools-20261025.zip']
    selected+=list((study/'audit-amendments').glob('*'))+[ROOT/'research/rounds.json',ROOT/'research/sciml_rounds.json']
    bundles.append(zip_paths(out/'selected-models-and-protocols.zip',selected))
    # All experiment and recovery artifacts remain in their original folders.
    # Include superseded runs, imports, failures and intermediate data too.
    artifacts=[]
    folders=[p for p in (ROOT/'generated').iterdir() if p.is_dir() and p!=out and not p.name.startswith(('delivery','test','tmp','.'))]
    for folder in sorted(folders):
        paths=[p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.lock')]
        for path in paths:artifacts.append(identity(path))
        print('Manifested',folder.name,len(paths),'files; total',len(artifacts),flush=True)
    for path in sorted((ROOT/'generated').glob('*')):
        if path.is_file() and path.suffix in ('.log','.err','.json','.zip') and not path.name.startswith(('delivery','service-','team-viewer','CONTINUATION')):artifacts.append(identity(path))
    artifact_manifest=out/'artifact-manifest.json'
    write(artifact_manifest,dict(schema='hyra-local-delivery-manifest-v1',baseline_git_head=baseline,
        created_utc=datetime.now(timezone.utc).isoformat(),files=artifacts,total_files=len(artifacts),
        total_bytes=sum(r['bytes'] for r in artifacts),scope='All retained experiment folders,including superseded history,raw events,episodes,recoveries,selected checkpoints,evaluations,audits and failures; excludes transient test/tmp folders and this delivery folder'))
    packages=bundles+[dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p)) for p in (out/'baseline.bundle',out/'tracked-changes.patch',artifact_manifest)]
    write(out/'package-audit.json',dict(status='passed',baseline_git_head=baseline,packages=packages,
        raw_artifact_files=len(artifacts),raw_artifact_bytes=sum(r['bytes'] for r in artifacts),
        checks=['Git baseline bundle verified','Every source/evidence and selected-model ZIP entry independently hashed after compression','Every retained original experiment file streamed and stable during hashing','All7full SciML audits,original P3closeout,plot and systemQA passed','Frozen scientific and evaluation source hashes unchanged'],
        limits='Local delivery only; large original recoveries are indexed at original paths,not duplicated. No push/deploy/paid resources. Restoring archives to another path requires fresh executable manifests or an explicitly separate reproduction checkout.'))
    (out/'README.md').write_text('# HyRA 本地交付\n\n科学结论与逐项验收见源码包中的 uav-workbench/docs/SCIML_FINAL_20261025.md、CONTINUOUS_ACCEPTANCE.md。\n\n- baseline.bundle：完整Git基线对象；可git clone到独立复现目录，再覆盖source-and-evidence.zip。\n- tracked-changes.patch：基线上的已跟踪文件差异；未跟踪的新功能已完整包含在源码包。\n- source-and-evidence.zip：当前源码、工作台、测试、配置、三个原始fixture及轻量证据。\n- selected-models-and-protocols.zip：12个选定完整检查点、12个clean actor、冻结计划/源码及失败修正说明。含运行登记，仅供归档；新复现副本不带入旧运行登记。\n- artifact-manifest.json：所有保留实验文件的实际路径、大小和SHA256，包括全部训练分片、恢复版本、原始洪水和失败记录。大文件保留原目录，不重复复制。\n- package-audit.json：各包SHA256及完整性核查。\n\n运行与从头复现使用 uav-workbench/docs/REPRODUCE.md。单机网页8767、洪水8766、双机回放8768均只监听127.0.0.1。两个工作台目录保持并列。\n\n本轮物理辅助闭环结果为负，不能以预测误差下降替代决策收益；仅三个独立确认母地形，10/12学习任务后段漂移，未证明收敛。原P3两轮与新机制一轮均按登记收束。\n',encoding='utf-8')
    print('PASSED local delivery packages and',len(artifacts),'original artifact hashes',flush=True)


if __name__=='__main__':main()
