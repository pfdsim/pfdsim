"""Install PFDSIM's live systemd/nginx files, preserving replaced targets.

Run as root from the checkout. The bootstrap phase serves only HTTP redirects
and ACME challenges; the https phase activates the maintained nginx file after
certificate issuance. Machine-specific certificate/webroot paths are generated
in nginx snippets outside the repository.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import pwd
import subprocess
import tempfile

DEPLOY = Path(__file__).resolve().parent
PROJECT = DEPLOY.parent
SYSTEM_ROOT = Path(PROJECT.anchor)
ETC = SYSTEM_ROOT / 'etc'
WEBROOT = SYSTEM_ROOT / 'var/www/html'


def run(*command):
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('bootstrap', 'https'), required=True)
    args = parser.parse_args()
    if os.geteuid() != 0:
        raise SystemExit('Run this installer with sudo.')
    account = pwd.getpwuid(PROJECT.stat().st_uid)
    if PROJECT != Path(account.pw_dir) / 'pfdsim':
        raise SystemExit('This service expects the pfdsim checkout in its owner’s home directory.')
    if not (PROJECT / '.venv/bin/gunicorn').is_file():
        raise SystemExit('Install the web extra in the project virtualenv first.')
    certificate = ETC / 'letsencrypt/live/chemicalprocess.org'
    if args.phase == 'https' and not all((certificate / name).is_file() for name in ('fullchain.pem', 'privkey.pem')):
        raise SystemExit('Issue the chemicalprocess.org/www certificate before activating HTTPS.')

    backup_root = ETC / 'pfdsim-deploy-backups'
    backup_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    backup = Path(tempfile.mkdtemp(prefix='deployment-', dir=backup_root))
    print('Recoverable deployment backups:', backup, flush=True)
    replacements = []

    def preserve(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or path.exists():
            saved = backup / path.relative_to(SYSTEM_ROOT)
            saved.parent.mkdir(parents=True, exist_ok=True)
            path.rename(saved)
            replacements.append((path, saved))
        else:
            replacements.append((path, None))

    def write(path, text):
        preserve(path)
        path.write_text(text)
        path.chmod(0o644)

    def link(path, target):
        if path.is_symlink() and path.resolve() == target.resolve():
            return
        preserve(path)
        path.symlink_to(target)

    nginx = ETC / 'nginx'
    site = nginx / 'sites-available/pfdsim'
    service = ETC / 'systemd/system/pfdsim@.service'
    try:
        WEBROOT.mkdir(parents=True, exist_ok=True)
        write(nginx / 'snippets/pfdsim-acme.conf',
              f'root {WEBROOT};\ndefault_type text/plain;\ntry_files $uri =404;\n')
        if args.phase == 'bootstrap':
            existing = site.read_text() if site.is_file() else ''
            if 'listen 443' in existing:
                raise SystemExit('HTTPS is already configured; use the https phase to update it.')
            write(site, (DEPLOY / 'chemicalprocess.org').read_text().split('# HTTPS begins')[0])
        else:
            write(nginx / 'snippets/pfdsim-tls.conf',
                  f'ssl_certificate {certificate / "fullchain.pem"};\n'
                  f'ssl_certificate_key {certificate / "privkey.pem"};\n'
                  'ssl_protocols TLSv1.2 TLSv1.3;\n'
                  'ssl_session_cache shared:PFDSIM_TLS:10m;\n'
                  'ssl_session_timeout 1d;\n')
            link(site, DEPLOY / 'chemicalprocess.org')
            hooks = ETC / 'letsencrypt/renewal-hooks/deploy'
            link(hooks / 'pfdsim-nginx', DEPLOY / 'renew-nginx.sh')
        # Appending this name preserves the existing default nginx virtual host.
        link(nginx / 'sites-enabled/zz-pfdsim', site)
        link(service, DEPLOY / 'pfdsim@.service')
        run('nginx', '-t')
        run('systemd-analyze', 'verify', f'pfdsim@{account.pw_name}.service')
    except BaseException:
        # Restore files without hard deletion; failed candidates remain in the
        # backup directory for inspection.
        for index, (path, saved) in enumerate(reversed(replacements)):
            if path.is_symlink() or path.exists():
                path.rename(backup / f'failed-candidate-{index}')
            if saved is not None:
                saved.rename(path)
        raise

    run('systemctl', 'daemon-reload')
    run('systemctl', 'enable', '--now', f'pfdsim@{account.pw_name}.service')
    run('systemctl', 'reload', 'nginx')
    timestamp = datetime.now(timezone.utc).isoformat()
    print(f'{args.phase} deployment completed at {timestamp}', flush=True)


if __name__ == '__main__':
    main()
