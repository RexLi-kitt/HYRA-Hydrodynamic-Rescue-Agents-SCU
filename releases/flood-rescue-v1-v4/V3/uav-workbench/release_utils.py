"""Release helpers that remove unrelated experiment imports."""
import csv
import hashlib
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent.parent

def atomic_torch(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    torch.save(obj,tmp)
    tmp.replace(path)

def write_rows(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0])
        writer.writeheader();writer.writerows(rows)

def model_file(directory,name):
    candidate=directory/name
    return candidate if candidate.exists() else ROOT/'models'/directory.name/name

def fingerprints():
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'models').rglob('*.pt'))}
