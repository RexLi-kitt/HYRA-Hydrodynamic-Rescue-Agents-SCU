"""Download and verify published training assets; no network access on import."""
import argparse,hashlib,json,os,shutil,urllib.request,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parent

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',choices=('MAPPO','DQN','VDN','QMIX','QPLEX'));ap.add_argument('--seed',type=int);ap.add_argument('--inputs',action='store_true');ap.add_argument('--all',action='store_true');a=ap.parse_args()
    if not (a.all or a.inputs or (a.model and a.seed)):ap.error('Select --all, --inputs, or --model NAME --seed N')
    manifest=json.loads((HERE/'assets.json').read_text(encoding='utf8'));chosen=[]
    for item in manifest['assets']:
        is_inputs=item.get('kind')=='inputs' or item['name']=='training-inputs-20261108-v1.zip'
        if a.all or (a.inputs and is_inputs) or (a.model and a.seed and item['name'].startswith(f'{a.model}-') and f'-seed{a.seed}-' in item['name']):chosen.append(item)
    if not chosen:raise ValueError('No matching asset')
    for item in chosen:
        is_inputs=item.get('kind')=='inputs' or item['name']=='training-inputs-20261108-v1.zip'
        target=(PACKAGE/item['path']).resolve()
        if not target.is_relative_to(PACKAGE.resolve()):raise ValueError('Unsafe asset path')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if target.stat().st_size!=item['bytes'] or sha(target)!=item['sha256']:raise ValueError('Existing asset changed: '+str(target))
        else:
            temp=target.with_name(target.name+'.download');
            if temp.exists():raise FileExistsError(temp)
            try:
                with urllib.request.urlopen(item['url']) as src,temp.open('xb') as dst:shutil.copyfileobj(src,dst)
                if temp.stat().st_size!=item['bytes'] or sha(temp)!=item['sha256']:raise ValueError('Download checksum failed')
                os.replace(temp,target)
            finally:
                if temp.exists():temp.unlink()
        if item.get('kind')=='checkpoint_zip':
            checkpoint=(PACKAGE/item['extracted_path']).resolve()
            if not checkpoint.is_relative_to(PACKAGE.resolve()):raise ValueError('Unsafe checkpoint path')
            checkpoint.parent.mkdir(parents=True,exist_ok=True)
            if checkpoint.exists():
                if checkpoint.stat().st_size!=item['extracted_bytes'] or sha(checkpoint)!=item['extracted_sha256']:raise ValueError('Existing checkpoint mismatch')
            else:
                temporary=checkpoint.with_name(checkpoint.name+'.extracting')
                if temporary.exists():raise FileExistsError(temporary)
                try:
                    with zipfile.ZipFile(target) as archive:
                        names=archive.namelist()
                        if names!=[item['member']] or Path(item['member']).name!=item['member']:raise ValueError('Unexpected checkpoint archive member')
                        with archive.open(item['member']) as src,temporary.open('xb') as dst:shutil.copyfileobj(src,dst)
                    if temporary.stat().st_size!=item['extracted_bytes'] or sha(temporary)!=item['extracted_sha256']:raise ValueError('Extracted checkpoint mismatch')
                    os.replace(temporary,checkpoint)
                finally:
                    if temporary.exists():temporary.unlink()
        if is_inputs:
            original=HERE/'original'
            if original.exists():
                inventory=json.loads((HERE/'original_manifest.json').read_text(encoding='utf8'))
                for rel,digest in inventory['all_files_sha256'].items():
                    f=(original/rel).resolve()
                    if not f.is_relative_to(original.resolve()) or not f.is_file() or sha(f)!=digest:raise ValueError('Existing input mismatch: '+rel)
                print('Existing input directory verified');continue
            stage=HERE/'extracting_inputs'
            if stage.exists():raise FileExistsError(stage)
            stage.mkdir()
            try:
                with zipfile.ZipFile(target) as archive:
                    for member in archive.infolist():
                        dest=(stage/member.filename).resolve()
                        if not dest.is_relative_to(stage.resolve()):raise ValueError('Unsafe archive path')
                    archive.extractall(stage)
                inventory=json.loads((HERE/'original_manifest.json').read_text(encoding='utf8'))
                for rel,digest in inventory['all_files_sha256'].items():
                    f=stage/'original'/rel
                    if not f.is_file() or sha(f)!=digest:raise ValueError('Extracted input mismatch: '+rel)
                os.replace(stage/'original',original)
            finally:
                resolved=stage.resolve()
                if resolved.parent!=HERE.resolve() or resolved.name!='extracting_inputs' or stage.is_symlink():raise ValueError('Unsafe extraction cleanup path')
                shutil.rmtree(stage)
        print('Verified',item['name'])
if __name__=='__main__':main()





