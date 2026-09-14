#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
PYTHON_VERSION="3.14.4"

mkdir -p "${RESULTS_DIR}"
cd "${REPO_ROOT}"

run_and_save() {
    local name="$1"
    shift
    uv run --python "${PYTHON_VERSION}" --frozen python "$@" \
        2>&1 | tee "${RESULTS_DIR}/${name}.txt"
}

run_and_save geometry_dependent_benchmark \
    scripts/thermal_conductivity/benchmark_gas_viscosity_cv_relation.py \
    --method geometry_dependent \
    --output-json scripts/thermal_conductivity/results/geometry_dependent_benchmark.json

run_and_save acentric_factor_benchmark \
    scripts/thermal_conductivity/benchmark_gas_viscosity_cv_relation.py \
    --method acentric_factor \
    --output-json scripts/thermal_conductivity/results/acentric_factor_benchmark.json

run_and_save acentric_correction_comparison \
    scripts/thermal_conductivity/experiment_functional_group_corrections.py \
    --baseline acentric_factor \
    --model log_constant_presence \
    --model log_linear_Tr_presence \
    --model log_quadratic_Tr_presence \
    --output-json scripts/thermal_conductivity/results/acentric_correction_comparison.json

run_and_save geometry_correction_comparison \
    scripts/thermal_conductivity/experiment_functional_group_corrections.py \
    --baseline geometry_dependent \
    --model log_constant_presence \
    --model log_linear_Tr_presence \
    --model log_inverse_Tr_presence \
    --model log_inverse_T_presence \
    --model log_constant_acid_detailed_presence \
    --model log_linear_Tr_acid_detailed_presence \
    --model log_inverse_Tr_acid_detailed_presence \
    --model log_inverse_T_acid_detailed_presence \
    --model log_inverse_T_acid_detailed_pruned_presence \
    --model log_inverse_T_acid_detailed_pruned_hc_split_presence \
    --selected-model log_inverse_T_acid_detailed_pruned_presence \
    --output-json scripts/thermal_conductivity/results/geometry_correction_comparison.json
