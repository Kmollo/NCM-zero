"""Canonical hashes and crash-safe local artifacts; never downloads anything."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [*sorted((ROOT/'ncm_code').glob('*.py')), ROOT/'upstream.lock.json', ROOT/'pyproject.toml',
             ROOT/'zero_baseline/grading.py', ROOT/'zero_baseline/sandbox.py',
             ROOT/'zero_baseline/runtime-manifest.json']
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
