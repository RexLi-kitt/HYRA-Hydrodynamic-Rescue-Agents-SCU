"""Lightweight gzip log IO for the standalone distribution."""
import gzip
import json


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with gzip.open(tmp, 'wt', encoding='utf8', compresslevel=1) as stream:
        json.dump(obj, stream, separators=(',', ':'), allow_nan=False)
    tmp.replace(path)


def read(path):
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        return json.load(stream)
