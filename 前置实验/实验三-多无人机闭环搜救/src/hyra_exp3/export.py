"""Byte-preserving portable experiment export with a complete file hash manifest."""
from pathlib import Path
import json
import shutil
from datetime import datetime
from .artifacts import sha256,write_json,TZ
from .protocol import load_protocol


def verify_record_references(root,files):
    """Check the original recorded hashes, before creating a new export manifest."""
    root=Path(root).resolve();digests={};checked=0;snapshots=0
    def check(path,expected):
        nonlocal checked
        path=Path(path)
        if not path.is_absolute():path=root/path
        path=path.resolve()
        if not path.is_relative_to(root):raise AssertionError('Recorded evidence path outside experiment export')
        if path not in digests:digests[path]=sha256(path)
        if digests[path]!=expected:raise AssertionError(f'Recorded evidence hash differs before export: {path}')
        checked+=1
    def walk(value):
        if isinstance(value,dict):
            paths=[k for k in value if (k=='path' or k.endswith('_path')) and isinstance(value[k],str)]
            for key in paths:
                checksum='sha256' if key=='path' else key[:-5]+'_sha256'
                if checksum not in value and len(paths)==1:checksum='sha256'
                if isinstance(value.get(checksum),str) and len(value[checksum])==64:check(value[key],value[checksum])
            for child in value.values():walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    for path in files:
        if path.suffix!='.json':continue
        record=load_protocol(path)
        if path.name=='source_sha256.json' and (path.parent/'source_snapshot').exists():
            for relative,expected in record.items():check(path.parent/'source_snapshot'/relative,expected);snapshots+=1
        walk(record)
    return dict(status='passed',recorded_reference_checks=checked,unique_files=len(digests),source_snapshot_files_checked=snapshots)


def export_bundle(root,run_paths,report_directory):
    root=Path(root).resolve();files=set()
    for name in ('run.py','pyproject.toml','requirements.txt','README.md','COMPLETION.json'):
        files.add(root/name)
    if (root/'README.development.md').exists():files.add(root/'README.development.md')
    for folder in ('src','scripts','tests','configs','manifests','registrations','docs'):
        files.update(p for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for directory in list(run_paths)+[report_directory]:
        directory=Path(directory).resolve()
        if not directory.is_relative_to(root):raise ValueError('Export source outside experiment')
        files.update(p for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.tmp')
    # Follow file references only; references to historical source directories do
    # not silently pull every unrelated development run into the final package.
    queue=[p for p in files if p.suffix=='.json'];scanned=set()
    expanded_sources=set()
    prefix=str(root).replace('\\','/').lower()+'/'
    def references(value):
        if isinstance(value,dict):
            for key in ('source_case','source'):
                item=value.get(key)
                if not isinstance(item,str) or not item.replace('\\','/').lower().startswith(prefix):continue
                directory=Path(item)
                identity=(key,directory)
                if identity in expanded_sources or not directory.is_dir():continue
                expanded_sources.add(identity)
                if key=='source_case':
                    for leaf in ('phase600/phase_case.json','phase_case.json','case.json'):
                        if (directory/leaf).is_file():yield directory/leaf
                else:
                    yield from directory.glob('*_summary.json')
            for item in value.values():yield from references(item)
        elif isinstance(value,list):
            for item in value:yield from references(item)
        elif isinstance(value,str) and value.replace('\\','/').lower().startswith(prefix):
            path=Path(value)
            if path.is_file():yield path
    while queue:
        path=queue.pop()
        if path in scanned:continue
        scanned.add(path)
        for dependency in references(json.loads(path.read_text(encoding='utf-8'))):
            if dependency not in files:
                files.add(dependency)
                if dependency.suffix=='.json':queue.append(dependency)
    references_audit=verify_record_references(root,files)
    directory=root/'results/packages'/datetime.now(TZ).strftime('%Y%m%d_%H%M%S_experiment3')
    directory.mkdir(parents=True,exist_ok=False)
    entries={};total=0
    for index,source in enumerate(sorted(files),1):
        if not source.is_file():raise FileNotFoundError(source)
        relative=source.relative_to(root);target=directory/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        digest=sha256(source)
        if sha256(target)!=digest:raise AssertionError('Export copy differs from source')
        size=target.stat().st_size;total+=size
        entries[relative.as_posix()]=dict(sha256=digest,bytes=size)
        if index%1000==0:print(f'Portable export {index}/{len(files)} files',flush=True)
    write_json(directory/'BUNDLE_ORIGIN.json',dict(original_root=str(root),byte_preserving=True,
        relocation='scripts/verify_bundle.py sets HYRA3_RELOCATE_FROM; raw JSON bytes and recorded hashes are preserved'))
    entries['BUNDLE_ORIGIN.json']=dict(sha256=sha256(directory/'BUNDLE_ORIGIN.json'),bytes=(directory/'BUNDLE_ORIGIN.json').stat().st_size)
    write_json(directory/'INTEGRITY.json',dict(schema='hyra3-portable-export-v1',files=entries,bytes=sum(v['bytes'] for v in entries.values()),
        raw_evidence_modified=False,absolute_references_relocated_only_while_reading=True,
        original_recorded_reference_audit=references_audit,
        verification_command='python scripts/verify_bundle.py --audit-primary'))
    result=verify_bundle(directory)
    write_json(directory/'PACKAGE_VERIFICATION.json',result)
    return directory


def verify_bundle(directory):
    directory=Path(directory).resolve();manifest=json.loads((directory/'INTEGRITY.json').read_text(encoding='utf-8'))
    total=0
    for relative,record in manifest['files'].items():
        path=(directory/relative).resolve()
        if not path.is_relative_to(directory):raise AssertionError('Export manifest path escapes package')
        if not path.is_file() or path.stat().st_size!=record['bytes'] or sha256(path)!=record['sha256']:
            raise AssertionError(f'Export integrity failed: {relative}')
        total+=record['bytes']
    return dict(status='passed',files=len(manifest['files']),bytes_verified=total,all_manifest_hashes_checked=True)
