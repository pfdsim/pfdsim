"""Local study directories and portable, manifest-relative fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent
ROOT = STUDY_DIR.parent
DATA_DIR = STUDY_DIR / 'data'
RESULTS_DIR = STUDY_DIR / 'results'


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative_path(path, manifest):
    return Path(os.path.relpath(Path(path).resolve(), Path(manifest).resolve().parent)).as_posix()


def write_manifest(path, arguments, files, **metadata):
    """Store paths relative to the manifest, independent of the checkout location."""
    path = Path(path)
    files = dict.fromkeys([*files, Path(__file__)])
    manifest = dict(
        metadata,
        path_base='manifest',
        arguments={key: relative_path(value, path) if isinstance(value, Path) else value
                   for key, value in arguments.items()},
        sha256={relative_path(source, path): file_sha256(source) for source in files},
    )
    with path.open('x') as stream:
        stream.write(json.dumps(manifest, indent=2) + '\n')


def verify_manifest(path):
    """Validate every available fingerprint, refusing incomplete historical runs."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get('path_base') != 'manifest':
        raise ValueError(f'Manifest must use manifest-relative paths: {path}')
    if manifest.get('unavailable_historical_code_sha256'):
        raise ValueError(f'Historical source snapshots are unavailable for {path}; '
                         'see its NOTE.txt and use the final assessment instead.')
    for name, expected in manifest['sha256'].items():
        source = Path(name)
        if source.is_absolute():
            raise ValueError(f'Absolute fingerprint path in {path}: {name}')
        source = path.parent / source
        if file_sha256(source) != expected:
            raise ValueError(f'Source benchmark fingerprint changed: {source}')
    return manifest
