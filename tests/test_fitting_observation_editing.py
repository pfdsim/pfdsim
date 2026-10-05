"""Browser observation identities stay distinct from cleared import history."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_appending_after_removal_does_not_reuse_imported_observation_ids(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to check the browser observation data model")
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(
        root / "static/js/fitting-observations.js", tmp_path / "observations.mjs"
    )
    script = """import assert from 'node:assert/strict';
import {appendObservations} from './observations.mjs';
const history = ['1','2','3','paper-point'];
const surviving = [{id:'1',kind:'VLE',sigma:{log_fugacity:0.02}}];
const incoming = [{id:'3',kind:'HE',HE_J_mol:20},{id:'paper-point',kind:'HE',HE_J_mol:30}];
const added = appendObservations(surviving,incoming,{},history);
assert.deepEqual(added.added,['4','5']);
assert.deepEqual(added.rows[0],surviving[0]);
assert.deepEqual(surviving,[{id:'1',kind:'VLE',sigma:{log_fugacity:0.02}}]);
assert.deepEqual(incoming.map(row=>row.id),['3','paper-point']);
assert.deepEqual(history,['1','2','3','paper-point']);
const empty = appendObservations([],incoming,{},history);
assert.deepEqual(empty.added,['4','5']);
const cleared = appendObservations([],incoming);
assert.deepEqual(cleared.added,['1','paper-point']);
"""
    result = subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
