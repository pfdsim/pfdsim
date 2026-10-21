"""Fetch only this study's shared pfdsim inputs from raw.githubusercontent.com.

Uses only the Python standard library. Existing files with different contents
are refused; downloads are staged before any dependency files are installed.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen

from solubility_dependencies import SHARED_FILES
from solubility_paths import ROOT, file_sha256

DEFAULT_REPOSITORY = 'pfdsim/pfdsim'


def raw_url(repository, ref, filename):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository):
        raise ValueError('Repository must have the form OWNER/REPOSITORY')
    if not ref.strip():
        raise ValueError('Ref must be a nonempty branch, tag, or commit SHA')
    return (f'https://raw.githubusercontent.com/{repository}/'
            f'{quote(ref, safe="")}/{quote(filename, safe="/")}')


def download_dependencies(destination, repository=DEFAULT_REPOSITORY, ref='main', timeout=30):
    """Install the 15 shared inputs without replacing existing files."""
    urls = {filename: raw_url(repository, ref, filename) for filename in SHARED_FILES}
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.solubility-download-', dir=destination) as temporary:
        staging = Path(temporary)
        conflicts = []
        for filename, url in urls.items():
            staged = staging / filename
            staged.parent.mkdir(parents=True, exist_ok=True)
            request = Request(url, headers={'User-Agent': 'pfdsim-solubility-dependencies'})
            with urlopen(request, timeout=timeout) as response, staged.open('wb') as stream:
                shutil.copyfileobj(response, stream)
                length = response.headers.get('Content-Length')
                if length is not None and stream.tell() != int(length):
                    raise OSError(f'Incomplete download: {filename}')
            target = destination / filename
            if target.exists() or target.is_symlink():
                if not target.is_file() or file_sha256(target) != file_sha256(staged):
                    conflicts.append(filename)
        if conflicts:
            raise FileExistsError('Existing dependencies differ from the download: '
                                  + ', '.join(conflicts)
                                  + '. Choose a fresh --destination; existing files were preserved.')
        installed = []
        for filename in urls:
            target = destination / filename
            if target.exists() or target.is_symlink():
                if target.is_file() and file_sha256(target) == file_sha256(staging / filename):
                    continue
                raise FileExistsError(f'Dependency changed during download: {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            # Staging is on the destination filesystem. Linking publishes each
            # complete file atomically and refuses a concurrently created target.
            os.link(staging / filename, target)
            installed.append(filename)
    return installed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', default=DEFAULT_REPOSITORY, help='GitHub OWNER/REPOSITORY')
    parser.add_argument('--ref', default='main', help='Branch, tag, or full commit SHA (recommended)')
    parser.add_argument('--destination', type=Path, default=ROOT,
                        help='Repository root to populate; defaults to the parent of solubility/')
    parser.add_argument('--timeout', type=float, default=30, help='Per-request timeout in seconds')
    parser.add_argument('--list', action='store_true', help='Print the 15 raw URLs without downloading')
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be finite and positive')
    try:
        if args.list:
            for filename in SHARED_FILES:
                print(raw_url(args.repository, args.ref, filename))
            return
        installed = download_dependencies(args.destination, args.repository, args.ref, args.timeout)
    except (OSError, ValueError) as error:
        parser.exit(1, f'{error}\n')
    print(f'Downloaded {len(SHARED_FILES)} dependency files from {args.repository}@{args.ref}.')
    print(f'Installed {len(installed)}; {len(SHARED_FILES)-len(installed)} existing files matched.')
    print(f'Destination: {args.destination.resolve()}')


if __name__ == '__main__':
    main()
