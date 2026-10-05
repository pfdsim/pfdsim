"""Shared task-start warnings and CPU allowance display in the actual JS module."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for the JS UI check")
def test_compute_budget_display_and_task_warning_cooldown(tmp_path):
    log_path = tmp_path / "account-ui.log"
    with log_path.open("w") as log:
        result = subprocess.run(
            ["node", "--experimental-vm-modules", str(Path(__file__).with_name("web_account_ui.mjs"))],
            stdout=log, stderr=log, timeout=10,
        )
    assert result.returncode == 0, log_path.read_text()
