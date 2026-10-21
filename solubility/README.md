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
  solubility_paths.py
  data/
    BigSolDBv2.0.csv
    BigSolDBv2.0_densities.csv
    curated-solubility-dataset.csv
  results/
    unifac-transfer-results/
    solvent-multiplier-results/
    infinite-dilution-assessment/
    solvent-bias-assessment/
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

Shared numerical code comes from `../unifac.py`, `../compiled_unifac.py`,
`../unifac_fragmenter.py`, and `../lyngby_parameters.py`, with their supporting
modules. The parameter tables are:

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

Run the focused path/provenance tests with:

```sh
python -m pytest solubility/tests/test_solubility_paths.py
```
