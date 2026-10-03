"""Exercise deployment generation without touching live system configuration."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_webroot_https_install_does_not_require_certbot_nginx_plugin(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "deploy"
    spec = importlib.util.spec_from_file_location("deploy_install", source / "install.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    project = tmp_path / "pfdsim"
    deploy = project / "deploy"
    deploy.mkdir(parents=True)
    for name in ("chemicalprocess.org", "pfdsim@.service", "renew-nginx.sh"):
        (deploy / name).write_text((source / name).read_text())
    binary = project / ".venv/bin/gunicorn"
    binary.parent.mkdir(parents=True)
    binary.touch()
    etc = tmp_path / "etc"
    certificate = etc / "letsencrypt/live/chemicalprocess.org"
    certificate.mkdir(parents=True)
    for name in ("fullchain.pem", "privkey.pem"):
        (certificate / name).touch()
    monkeypatch.setattr(installer, "PROJECT", project)
    monkeypatch.setattr(installer, "DEPLOY", deploy)
    monkeypatch.setattr(installer, "SYSTEM_ROOT", tmp_path)
    monkeypatch.setattr(installer, "ETC", etc)
    monkeypatch.setattr(installer, "WEBROOT", tmp_path / "var/www/html")
    monkeypatch.setattr(installer.os, "geteuid", lambda: 0)
    monkeypatch.setattr(installer.pwd, "getpwuid", lambda uid: SimpleNamespace(
        pw_dir=str(tmp_path), pw_name="laboratory-owner",
    ))
    monkeypatch.setattr("sys.argv", ["install.py", "--phase", "https"])
    commands = []
    monkeypatch.setattr(installer, "run", lambda *args: commands.append(args))
    installer.main()
    snippet = (etc / "nginx/snippets/pfdsim-tls.conf").read_text()
    assert str(certificate / "fullchain.pem") in snippet
    assert str(certificate / "privkey.pem") in snippet
    assert "ssl_protocols TLSv1.2 TLSv1.3;" in snippet
    assert "include" not in snippet and "ssl_dhparam" not in snippet
    assert (etc / "nginx/sites-available/pfdsim").resolve() == deploy / "chemicalprocess.org"
    assert commands.index(("nginx", "-t")) < commands.index(("systemctl", "reload", "nginx"))
    assert ("systemctl", "enable", "--now", "pfdsim@laboratory-owner.service") in commands
