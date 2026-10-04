"""Run the default suite with pytest workers, including pytest function tests."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKER_COUNT = 6


def run_parallel_default() -> int:
    # Used by explicit unittest discovery. Bare pytest configures xdist in
    # conftest.py and does not need an additional coordinator process.
    return subprocess.call(
        [sys.executable, '-m', 'pytest', '-n', str(WORKER_COUNT), 'tests'],
        cwd=ROOT,
    )


if __name__ == '__main__':
    raise SystemExit(run_parallel_default())
