"""Test generic vapor-dimerization corrections for estimated acid Hvap.

The dataset is the carboxylic-acid subset excluded from the maintained Perry
Hvap comparisons. Perry Table 2-150 supplies calorimetric reference Hvap and
Table 2-8 supplies the incipient-vapor saturation pressure. Each predictive
arm uses a source-backed Tb, Nannoolal-estimated Tc/Pc, and Lee-Kesler omega.

The requested correction uses the production generic monoacid parameters and
the ideal physical-fugacity VDM equilibrium state::

    corrected Hvap = base Hvap - extent * delta_H_dimer

Because production ``delta_H_dimer`` is negative, literal signed subtraction
raises the estimated Hvap. Subtracting its positive dissociation magnitude is
therefore reported separately as ``H + extent * delta_H_dimer``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import warnings

from chemicals.acentric import LK_omega
from rdkit import Chem


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import nannoolal_method as nm  # noqa: E402
from vapor_dimerization import (  # noqa: E402
    GENERIC_MONOCARBOXYLIC_DELTA_H_J_MOL,
    GENERIC_MONOCARBOXYLIC_DELTA_S_J_MOL_K,
    VaporDimerizationModel,
)

from compare_estimated_deltaz import peng_robinson_dz, p95_absolute  # noqa: E402
from compare_hvap_formula import (  # noqa: E402
    P_ATM_KPA,
    bisect_temperature,
    corresponding_states_hvap,
    smiles_for,
    statistics,
    trouton_hvap_tb,
    watson_hvap,
)


TR_TARGETS = (0.5, 0.6, 0.7, 0.8)
COOH = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
OUTPUT_PATH = REPO_ROOT / "outputs/nannoolal_acid_hvap_dimer_correction.txt"

BASE_ARMS = (
    "Nannoolal + Peng-Robinson",
    "Corresponding states",
    "Trouton-Watson",
)
VARIANTS = (
    "uncorrected",
    "literal signed subtraction: H - extent*dHdim",
    "subtract |dHdim|: H + extent*dHdim",
)


def format_statistics(values: list[float], coverage: int) -> str:
    stat = statistics(values)
    if stat is None:
        return f"unavailable (0/{coverage})"
    return (
        f"{stat['mape']:.2f} | {stat['median']:.2f} | "
        f"{p95_absolute(values):.2f} | {stat['bias']:+.2f} "
        f"({stat['n']}/{coverage})"
    )


def main() -> int:
    perry = json.loads(
        (REPO_ROOT / "data/perry_properties.json").read_text()
    )["chemicals"]
    errors = {
        variant: {
            arm: {target: [] for target in TR_TARGETS}
            for arm in BASE_ARMS
        }
        for variant in VARIANTS
    }
    eligible = {target: 0 for target in TR_TARGETS}
    extents = {target: [] for target in TR_TARGETS}
    curve_errors = {
        variant: {arm: {} for arm in BASE_ARMS}
        for variant in VARIANTS[1:]
    }
    subset_errors = {
        variant: {
            arm: {"monoacid": [], "polyacid": []}
            for arm in BASE_ARMS
        }
        for variant in VARIANTS
    }
    acid_candidates = 0
    candidate_names: dict[str, str] = {}
    skipped: dict[str, str] = {}
    included_acids: set[str] = set()
    monocarboxylic_acids: set[str] = set()
    polycarboxylic_acids: set[str] = set()

    dimer_model = VaporDimerizationModel(
        "acid",
        "acid_dimer",
        delta_H=GENERIC_MONOCARBOXYLIC_DELTA_H_J_MOL,
        delta_S=GENERIC_MONOCARBOXYLIC_DELTA_S_J_MOL_K,
    )

    for cas, row in perry.items():
        if cas == "302-17-0":
            continue
        critical = row.get("critical_constants") or {}
        hvap_rows = row.get("heat_of_vaporization") or []
        psat_rows = row.get("vapor_pressure") or []
        if (
            not hvap_rows
            or not psat_rows
            or not all(
                critical.get(key)
                for key in ("Tc_K", "Pc_MPa", "Vc_m3_per_kmol")
            )
            or critical.get("omega") is None
        ):
            continue
        smiles = smiles_for(cas)
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        if molecule is None or COOH is None:
            continue
        cooh_count = len(molecule.GetSubstructMatches(COOH))
        if cooh_count == 0:
            continue
        acid_candidates += 1
        acid_name = str(row.get("name") or cas)
        candidate_names[cas] = acid_name
        acid_kind = "monoacid" if cooh_count == 1 else "polyacid"

        real_tc = float(critical["Tc_K"])
        hvap_row = hvap_rows[0]
        psat_row = psat_rows[0]
        hvap_coefficients = (
            list(hvap_row["coefficients"]) + [0.0] * 4
        )[:4]
        hvap_low = float(hvap_row["T_min_K"])
        hvap_high = float(hvap_row["T_max_K"])
        psat_coefficients = (
            list(psat_row["coefficients"]) + [0.0] * 5
        )[:5]
        if psat_coefficients[4] == 0.0:
            psat_coefficients[4] = 1.0
        psat_low = float(psat_row["T_min_K"])
        psat_high = float(psat_row["T_max_K"])

        def reference_hvap(temperature: float) -> float:
            tr = temperature / real_tc
            a, b, c, d = hvap_coefficients
            return a * (1.0 - tr) ** (b + c * tr + d * tr**2) / 1000.0

        def reference_psat(temperature: float) -> float:
            a, b, c, d, exponent = psat_coefficients
            return math.exp(
                a
                + b / temperature
                + c * math.log(temperature)
                + d * temperature**exponent
            ) / 1000.0

        tb = bisect_temperature(
            reference_psat,
            P_ATM_KPA,
            psat_low,
            psat_high,
        )
        if tb is None:
            skipped[cas] = "normal-boiling root unavailable"
            continue
        try:
            curve = nm.estimate_psat(smiles, tb=tb)
            estimated = nm.estimate(smiles, tb=tb)
            if (
                curve.db is None
                or estimated.tc_K is None
                or estimated.pc_kPa is None
            ):
                missing = []
                if curve.db is None:
                    missing.append("Psat fragmentation")
                if estimated.tc_K is None:
                    missing.append("Tc")
                if estimated.pc_kPa is None:
                    missing.append("Pc")
                skipped[cas] = (
                    "Nannoolal setup unavailable: " + ", ".join(missing)
                )
                continue
            omega = LK_omega(
                tb,
                estimated.tc_K,
                estimated.pc_kPa * 1000.0,
            )
            if omega is None or not math.isfinite(omega):
                continue
        except Exception as exc:
            skipped[cas] = f"Nannoolal/LK setup failed: {type(exc).__name__}"
            continue

        contributed = False
        for target_tr in TR_TARGETS:
            temperature = target_tr * real_tc
            if not (
                hvap_low + 1.0 < temperature < hvap_high - 1.0
                and psat_low + 1.0 < temperature < psat_high - 1.0
            ):
                continue
            reference = reference_hvap(temperature)
            reference_pressure_kpa = reference_psat(temperature)
            if reference <= 0.0 or reference_pressure_kpa <= 0.0:
                continue

            nannoolal_slope_hvap = curve.dhvap_J_mol(temperature)
            nannoolal_pressure_kpa = curve.psat_kPa(temperature)
            nannoolal_delta_z = (
                None
                if nannoolal_pressure_kpa is None
                else peng_robinson_dz(
                    temperature,
                    nannoolal_pressure_kpa,
                    estimated.tc_K,
                    estimated.pc_kPa,
                    omega,
                )
            )
            nannoolal_hvap = (
                None
                if nannoolal_slope_hvap is None or nannoolal_delta_z is None
                else nannoolal_slope_hvap * nannoolal_delta_z
            )
            predictions = {
                BASE_ARMS[0]: nannoolal_hvap,
                BASE_ARMS[1]: corresponding_states_hvap(
                    temperature,
                    estimated.tc_K,
                    omega,
                ),
                BASE_ARMS[2]: watson_hvap(
                    temperature,
                    trouton_hvap_tb(tb),
                    tb,
                    estimated.tc_K,
                ),
            }

            state = dimer_model.association_state(
                temperature,
                reference_pressure_kpa / 100.0,
                {"acid": 1.0},
            )
            extent = float(
                state.get("extents", {}).get(("acid", "acid"), 0.0)
            )
            association_enthalpy = (
                extent * GENERIC_MONOCARBOXYLIC_DELTA_H_J_MOL
            )
            eligible[target_tr] += 1
            extents[target_tr].append(extent)
            contributed = True

            for arm, base_hvap in predictions.items():
                if (
                    base_hvap is None
                    or not math.isfinite(base_hvap)
                    or base_hvap <= 0.0
                ):
                    continue
                values = {
                    VARIANTS[0]: base_hvap,
                    VARIANTS[1]: base_hvap - association_enthalpy,
                    VARIANTS[2]: base_hvap + association_enthalpy,
                }
                for variant, prediction in values.items():
                    error = 100.0 * (prediction / reference - 1.0)
                    errors[variant][arm][target_tr].append(error)
                    subset_errors[variant][arm][acid_kind].append(error)
                    if variant != VARIANTS[0]:
                        curve_errors[variant][arm].setdefault(
                            (cas, acid_name),
                            [],
                        ).append(abs(error))

        if contributed:
            included_acids.add(cas)
            target_set = (
                monocarboxylic_acids
                if cooh_count == 1
                else polycarboxylic_acids
            )
            target_set.add(cas)
        else:
            skipped[cas] = "no common Perry Psat/Hvap state at selected Tr"

    lines = [
        "Generic vapor-dimer correction for estimated carboxylic-acid Hvap",
        f"Acid candidates matching the earlier exclusion: {acid_candidates}",
        f"Acids contributing at least one state: {len(included_acids)}",
        (
            f"  monocarboxylic={len(monocarboxylic_acids)}; "
            f"polycarboxylic={len(polycarboxylic_acids)}"
        ),
        (
            "Inputs: source-backed Tb; Nannoolal-estimated Tc/Pc; "
            "Lee-Kesler omega; Perry incipient-vapor Psat."
        ),
        "Nannoolal PR delta Z is evaluated at Nannoolal-predicted Psat.",
        (
            "Generic VDM: deltaH=-60.5 kJ/mol-dimer; "
            "deltaS=-144 J/mol/K."
        ),
        (
            "Errors are MAPE | median APE | p95 APE | bias percent "
            "(coverage/eligible)."
        ),
        "",
        "Reaction extent (mol dimer formed per nominal mol acid):",
    ]
    for target_tr in TR_TARGETS:
        values = sorted(extents[target_tr])
        if not values:
            lines.append(f"Tr={target_tr:g}: unavailable")
            continue
        lines.append(
            f"Tr={target_tr:g}: median={values[len(values) // 2]:.5f}; "
            f"range={values[0]:.5f}-{values[-1]:.5f}; n={len(values)}"
        )

    lines.extend(("", "Excluded-set accounting:"))
    for cas, name in sorted(candidate_names.items(), key=lambda item: item[1]):
        status = "included" if cas in included_acids else skipped.get(cas, "skipped")
        lines.append(f"  {name} ({cas}): {status}")

    for variant in VARIANTS:
        lines.extend(("", variant))
        lines.append(
            f"{'arm':36s}"
            + "".join(f" {'Tr=' + str(tr):>38s}" for tr in TR_TARGETS)
        )
        for arm in BASE_ARMS:
            cells = [
                format_statistics(
                    errors[variant][arm][target_tr],
                    eligible[target_tr],
                )
                for target_tr in TR_TARGETS
            ]
            lines.append(
                f"{arm:36s}"
                + "".join(f" {cell:>38s}" for cell in cells)
            )
        lines.append("Aggregate over Tr=0.5-0.8:")
        for arm in BASE_ARMS:
            aggregate = [
                error
                for target_tr in TR_TARGETS
                for error in errors[variant][arm][target_tr]
            ]
            lines.append(
                f"{arm:36s} "
                f"{format_statistics(aggregate, sum(eligible.values()))}"
            )
        lines.append("Aggregate by COOH count:")
        for arm in BASE_ARMS:
            mono = subset_errors[variant][arm]["monoacid"]
            poly = subset_errors[variant][arm]["polyacid"]
            lines.append(
                f"{arm:36s} mono {format_statistics(mono, len(mono))}; "
                f"poly {format_statistics(poly, len(poly))}"
            )

    for variant in VARIANTS[1:]:
        lines.extend(("", f"{variant}; per-acid curve MAPE:"))
        for arm in BASE_ARMS:
            summaries = sorted(
                (
                    sum(values) / len(values),
                    name,
                    cas,
                    len(values),
                )
                for (cas, name), values in curve_errors[variant][arm].items()
                if values
            )
            curve_mapes = [item[0] for item in summaries]
            lines.append(f"{arm}:")
            if not summaries:
                lines.append("  unavailable")
                continue
            lines.append(
                f"  median={sorted(curve_mapes)[len(curve_mapes) // 2]:.2f}%; "
                f"p95={p95_absolute(curve_mapes):.2f}%; curves={len(summaries)}"
            )
            for mape, name, cas, count in reversed(summaries[-5:]):
                lines.append(
                    f"  worst: {name} ({cas}) MAPE={mape:.2f}% n={count}"
                )

    report = "\n".join(lines) + "\n"
    print(report, end="")
    OUTPUT_PATH.write_text(report)
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raise SystemExit(main())
