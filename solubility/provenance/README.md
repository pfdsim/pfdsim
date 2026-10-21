# Original study source

`original-scripts/*.py.txt` are immutable snapshots of the five study scripts
before they were moved out of `scripts/activity_fitting/`. They are historical
source records, not maintained entry points. Run the scripts at the top of
`solubility/`.

The existing numerical outputs were preserved byte for byte. Their manifests
now use relative paths, and references to original script hashes resolve to
these snapshots. Dependent manifest hashes were updated to reflect the path
changes. Each migrated manifest records the SHA-256 of its original manifest
under `relocation.original_manifest_sha256`.

Three preliminary runs already referenced source versions that were no longer
available before the move: `unifac-transfer-pilot`, `unifac-transfer-20261014`,
and `solvent-bias-results`. Their original missing-source hashes are retained
under `unavailable_historical_code_sha256`; validation rejects those runs as
upstream inputs. Their results and explanatory notes remain available.
All four final assessment manifests validate against the preserved originals.

The saved base manifests were subsequently extended to fingerprint the imported
`compiled_cache.py` and `physical_constants.py` files. These hashes were recorded
after the original runs, not measured retroactively at execution time.
`fingerprint_extension` records the preceding manifest hash and added paths;
dependent manifest hashes were updated to preserve the existing result chains.
No numerical outputs, original source snapshots, or existing source/parameter
hashes were replaced by this extension.
