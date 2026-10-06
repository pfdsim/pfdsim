"""Review information, full-fit loading and preservation of unsaved work."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required for the JS UI check")
def test_review_publication_and_full_fit_loading(tmp_path):
    log_path = tmp_path / "fit-review-ui.log"
    with log_path.open("w") as log:
        result = subprocess.run(
            ["node", "--experimental-vm-modules", str(Path(__file__).with_name("activity_fit_review_ui.mjs"))],
            stdout=log, stderr=log, timeout=10,
        )
    assert result.returncode == 0, log_path.read_text()
