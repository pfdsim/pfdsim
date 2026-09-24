"""Compare sparse assembly with an optional saved pre-change module.

Run from the repository root, for example:
    python scripts/benchmark_sparse_jacobian.py --baseline /tmp/sparse_before.py

Measures assembly and conversion only, excluding thermodynamics and solves.
Every timed result is checked against the same reference matrix.
"""

import argparse
import importlib.util
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--stages", type=int, nargs="+", default=[50, 500, 1000])
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 1 or any(stages < 1 for stages in args.stages):
        parser.error("stages and repeats must be positive")
    current = load_module(
        Path(__file__).resolve().parents[1] / "sparse_jacobian.py",
        "sparse_current",
    )
    modules = {"current": current}
    if args.baseline:
        modules["baseline"] = load_module(args.baseline, "sparse_baseline")
    rng = np.random.default_rng(20260924)
    print("Median milliseconds; width=10 variables/stage, 50% structural zeros")
    for stages in args.stages:
        width = 10
        n = stages * width
        structural = current.SparsePatternBuilder((n, n))
        for row in range(n):
            stage = row // width
            structural.mark_range(
                row, max(0, stage - 1) * width, min(stages, stage + 2) * width
            )
        pattern = structural.tocsr()
        values = rng.normal(size=pattern.nnz)
        values[rng.random(pattern.nnz) < 0.5] = 0.0
        reference = csr_matrix(
            (values.copy(), pattern.indices.copy(), pattern.indptr.copy()),
            shape=pattern.shape,
        )
        reference.eliminate_zeros()
        rows = np.repeat(np.arange(n), np.diff(pattern.indptr))
        entries = list(zip(rows.tolist(), pattern.indices.tolist(), values.tolist()))
        csc = reference.tocsc()
        # Include explicit zero entries in column assignment workloads.
        full_csc = csr_matrix(
            (values.copy(), pattern.indices.copy(), pattern.indptr.copy()),
            shape=pattern.shape,
        ).tocsc()
        columns = [
            (full_csc.indices[full_csc.indptr[col]:full_csc.indptr[col + 1]],
             col, full_csc.data[full_csc.indptr[col]:full_csc.indptr[col + 1]])
            for col in range(n)
        ]
        fixed = {name: module.FixedPatternCSR(pattern) for name, module in modules.items()}
        print(f"stages={stages} shape={n}x{n} structural_nnz={pattern.nnz} "
              f"numeric_nnz={csc.nnz} extra_copy_MiB={values.nbytes / 2**20:.3f}")
        for mode in ("add", "set_column", "conversion"):
            timings = {name: [] for name in modules}
            for repeat in range(args.repeats + 1):
                names = list(modules)
                if repeat % 2:
                    names.reverse()
                for name in names:
                    builder = fixed[name].empty()
                    if mode == "conversion":
                        builder.data[:] = values
                    start = perf_counter()
                    if mode == "add":
                        for row, col, value in entries:
                            builder.add(row, col, value)
                    elif mode == "set_column":
                        for column_rows, col, column_values in columns:
                            builder.set_column(column_rows, col, column_values)
                    result = builder.tocsr()
                    elapsed = perf_counter() - start
                    difference = result - reference
                    if difference.nnz:
                        raise AssertionError(f"{name}/{mode}: incorrect assembled matrix")
                    if repeat:
                        timings[name].append(elapsed * 1000)
            medians = {name: median(samples) for name, samples in timings.items()}
            report = " ".join(f"{name}={value:.3f}" for name, value in medians.items())
            if "baseline" in medians:
                report += f" ratio={medians['current'] / medians['baseline']:.3f}"
            print(f"  {mode}: {report}; matrix_error=0", flush=True)


if __name__ == "__main__":
    main()
