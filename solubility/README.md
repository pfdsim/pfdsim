# Solubility assessments

The study scripts, source datasets, and saved results live together here.
There is no dependency on a separate `TESTS/solubility` directory. The scripts
use pfdsim's shared UNIFAC implementation and parameter tables in the parent
repository; those are not copied into this study.

## Layout

```text
solubility/
  benchmark_solubility_transfer.py
  analyze_solubility_transfer.py
  benchmark_solvent_transfer_multipliers.py
  benchmark_infinite_dilution_solubility.py
  analyze_solvent_bias.py
  benchmark_bias_corrected_transfer.py
  solubility_paths.py
  solubility_dependencies.py
  pull_pfdsim_dependencies.py
  data/
    BigSolDBv2.0.csv
    BigSolDBv2.0_densities.csv
    curated-solubility-dataset.csv
  results/
    unifac-transfer-results/
    solvent-multiplier-results/
    infinite-dilution-assessment/
    solvent-bias-assessment/
    bias-corrected-transfer-results/
    ... historical and diagnostic runs ...
  provenance/original-scripts/
  tests/
```

The existing datasets total about 21 MiB; saved runs total about 3 GiB.
Generated result files up to 50 MB (50,000,000 bytes) are included in Git;
the five existing files above that limit are explicitly ignored and stay local.
These are the final and preliminary full-run `metric_records.npy` arrays and
all three `predictions.csv` files. Git ignore rules do not support size limits,
so check sizes and update the explicit list when adding future results.
All historical runs and notes remain present locally. This directory is a
repository study, not a separately installed pfdsim package.

## Dependencies

Use the pfdsim Python environment. The study needs NumPy, Numba, RDKit, SciPy,
pandas, openpyxl, and Matplotlib. The first six are pfdsim dependencies;
Matplotlib is additionally needed by `analyze_solubility_transfer.py`.

The complete shared dependency set is **six Python files and nine parameter
files** outside this directory. `solubility_dependencies.py` owns the file list
used both by the benchmark fingerprints and the dependency downloader.

| Python file, relative to this directory | Reached through | Purpose |
| --- | --- | --- |
| `../unifac.py` | Direct benchmark import | Parameter loading, group resolution, and reference activity coefficients. |
| `../compiled_unifac.py` | Direct benchmark import | Numeric model arrays and compiled activity-coefficient kernels. |
| `../unifac_fragmenter.py` | `unifac.py` | Variant-specific fragmentation of molecular SMILES. |
| `../lyngby_parameters.py` | `unifac.py` | Translation of classic groups into Lyngby groups. |
| `../compiled_cache.py` | `compiled_unifac.py` | Numba cache identities and decorator. |
| `../physical_constants.py` | `compiled_unifac.py` | Shared physical constants imported by that backend. |

The nine parameter files are:

```text
../data/unifac_params.json
../data/unifac_dmd.txt
../data/nist_modified_unifac_params.json
../data/lyngby_unifac.json
../data/unifac2/UNIFAC2rq.csv
../data/unifac2/UNIFAC2int.csv
../data/unifac2/UNIFM2rq.xlsx
../data/unifac2/UNIFM2intA.xlsx
../data/unifac2/UNIFM2intB.xlsx
```

Lyngby's group translation also reads `unifac_params.json`, already included
above. All six variants are exercised by the finite-concentration and
infinite-dilution numerical checks. The multiplier and bias analyses reuse
saved predictions and validate their upstream source fingerprints. The report
renderer imports the benchmark module but does not load the parameter tables
when rendering existing numerical results.

These workflows supply SMILES directly, so the conditional identifier resolver
in `unifac.py` does not import `chemical_properties.py` or its dependencies.
They do not use the Lyngby gas extension (`mhv2_unifac.json`), the EOS models,
or the process simulator.

## Fetching only the shared pfdsim files

`pull_pfdsim_dependencies.py` downloads exactly the 15 files above through
`https://raw.githubusercontent.com/`, without cloning a repository or using the
GitHub API. The downloader itself needs only the Python standard library and
the two local study helpers; it does not install Python packages.

```sh
python solubility/pull_pfdsim_dependencies.py --list
python solubility/pull_pfdsim_dependencies.py --ref main
```

The default repository is `pfdsim/pfdsim`, and the default destination is the
parent of `solubility/`. For a reproducible snapshot, pass a full GitHub commit
SHA to `--ref` instead of a moving branch. `--repository OWNER/REPOSITORY` and
`--destination DIRECTORY` select another source or destination. An alternate
destination becomes the shared-code root only when the study's `solubility/`
directory is also placed inside it; existing scripts retain their location-based
root. This allows the study to run without the rest of a full pfdsim checkout.

Downloads are staged before installation. Identical existing files are left
untouched, and differing existing files cause an error without overwriting them.
Choose a fresh destination when fetching a different revision. Files are
installed atomically, and failed downloads do not install a partial file set.
Existing result manifests still require their recorded source and parameter
hashes to match; downloading a newer revision does not bypass that validation.

## Running the study

Run the scripts directly from the repository root. Their default input and
output directories are resolved from the scripts' location, independently of
the working directory:

```sh
python solubility/benchmark_solubility_transfer.py
python solubility/analyze_solubility_transfer.py
python solubility/benchmark_solvent_transfer_multipliers.py
python solubility/benchmark_infinite_dilution_solubility.py
python solubility/analyze_solvent_bias.py
python solubility/benchmark_bias_corrected_transfer.py
```

These commands describe the dependency order. The completed results already
exist, so benchmarks refuse to overwrite their default output directories.
The renderer likewise refuses to overwrite its report, comparison JSON, or
figures. Inspect the existing `REPORT.md` files instead of rerunning a finished
assessment.

For a fresh assessment, select new output directories and pass them to the
downstream stages:

```sh
python solubility/benchmark_solubility_transfer.py --output-dir solubility/results/new-transfer
python solubility/analyze_solubility_transfer.py solubility/results/new-transfer
python solubility/benchmark_solvent_transfer_multipliers.py --results-dir solubility/results/new-transfer --output-dir solubility/results/new-multipliers
python solubility/benchmark_infinite_dilution_solubility.py --results-dir solubility/results/new-transfer --baseline-dir solubility/results/new-multipliers --output-dir solubility/results/new-infinite
python solubility/analyze_solvent_bias.py --results-dir solubility/results/new-transfer --infinite-dir solubility/results/new-infinite --output-dir solubility/results/new-bias
```

The benchmark and downstream analyses accept `--data-dir` for alternate source
CSVs. It defaults to `solubility/data` and does not depend on where the results
are written. User-supplied relative CLI paths are relative to the working
directory. Use `--help` for worker, bootstrap, and numerical-check options.

Long benchmark commands should persist both output streams, for example:

```sh
set -o pipefail
python solubility/benchmark_solubility_transfer.py --output-dir solubility/results/new-transfer 2>&1 | tee /tmp/solubility-transfer.log
```

## Saved results and provenance

The four final assessment directories are `unifac-transfer-results`,
`solvent-multiplier-results`, `infinite-dilution-assessment`, and
`solvent-bias-assessment`. Historical directories retain their `NOTE.txt`
explanations, including limitations of preliminary and pilot runs.

Manifest fingerprint paths and path-valued arguments are relative to the
containing `manifest.json`. Moving the whole checkout therefore preserves
validation. Moving individual result folders requires keeping their relative
relationship to the referenced data, source files, and upstream results.
Upstream hashes are validated before their predictions are reused.

A fresh checkout includes the reports, figures, and smaller result files, but
omits the five large files listed in `.gitignore`. Reanalyzing a full run needs
its local `metric_records.npy`, or a new base benchmark run to regenerate that
array. The manifests retain hashes of omitted arrays so local copies can still
be validated; the missing files are not silently substituted or skipped.

Existing manifests were relocated without changing numerical outputs. They
reference immutable original source snapshots under `provenance/`; new runs
fingerprint the maintained scripts. See `provenance/README.md` for details.

The shared fingerprints also cover `compiled_cache.py` and
`physical_constants.py`. Their previously omitted hashes were added to the
saved base manifests after the original assessments, with a
`fingerprint_extension` provenance record. Downstream manifest references were
updated to those augmented manifests. Numerical outputs, original script
snapshots, and all pre-existing source/parameter fingerprints were preserved.
Previously incomplete historical source records remain marked incomplete.

## Held-out solvent-bias correction

`benchmark_bias_corrected_transfer.py` reuses the saved finite and infinite-
dilution predictions and the multiplier benchmark's molecule-level folds.
It estimates solvent corrections using training molecules only. Its main
correction subtracts a target-minus-reference log bias; a target-only mean
correction is also reported as a sensitivity check.

For a controlled comparison, the script fits a solvent-only transfer baseline
on exactly the same training cases, chemical coverage, and equal-molecule
weights as the relative correction. The original broad-population baseline
is retained separately. It does not select offsets using the earlier
full-dataset significance analysis or refit any UNIFAC parameters.

The default evaluates five scenarios (finite concentration without a dilute
cutoff, and finite/infinite dilution with both observed mole fractions below
0.05 or 0.01), using the existing five-fold splits repeated ten times.
The principal uncertainty intervals account for 60 corrected-versus-matched-
baseline comparisons: six variants, five scenarios, and two correction forms.
Outputs include every held-out error, fitted fold coefficients, and a report.
The script refuses to overwrite an existing results directory.

```sh
set -o pipefail
python solubility/benchmark_bias_corrected_transfer.py --output-dir solubility/results/new-bias-correction 2>&1 | tee /tmp/solubility-bias-correction.log
```

Run the focused path/provenance tests with:

```sh
python -m pytest solubility/tests/
```
