"""Compare delta-Z treatments for Nannoolal Hvap with estimated criticals.

The Nannoolal vapor-pressure slope is anchored at a source-backed normal
boiling point.  Nannoolal estimates Tc, Pc, and Vc, and Lee-Kesler estimates
omega from that internally consistent Tb/Tc/Pc tier.  The final Hvap estimate
uses virial/Rackett, Peng-Robinson, or Patel-Teja-Valderrama delta Z.

Perry 9th Table 2-150 correlations are the reference. Carboxylic acids are
excluded because a Psat derivative describes their apparent
liquid-to-associated-vapor enthalpy rather than calorimetric monomer Hvap.
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
from cubic_eos import CubicEOS  # noqa: E402

from compare_hvap_formula import (  # noqa: E402
    bisect_temperature,
    ptv_dz,
    smiles_for,
    statistics,
    virial_rackett_dz,
)


P_ATM_KPA = 101.325
TR_TARGETS = (0.5, 0.6, 0.7, 0.75, 0.8)
COOH = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
OUTPUT_PATH = REPO_ROOT / "outputs/nannoolal_estimated_deltaz_comparison.txt"

ARMS = (
    "Nannoolal dZ=1",
    "Nannoolal + virial/Rackett",
    "Nannoolal + Peng-Robinson",
    "Nannoolal + PTV",
)


def peng_robinson_dz(temperature, pressure_kpa, tc, pc_kpa, omega):
    reduced_temperature = temperature / tc
    reduced_pressure = pressure_kpa / pc_kpa
    if not (
        0.0 < reduced_temperature < 1.0
        and math.isfinite(reduced_pressure)
        and reduced_pressure > 0.0
    ):
        return None
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (
        1.0 + kappa * (1.0 - math.sqrt(reduced_temperature))
    ) ** 2
    a = (
        0.45724 * alpha * reduced_pressure / reduced_temperature**2
    )
    b = 0.07780 * reduced_pressure / reduced_temperature
    roots = sorted(
        root
        for root in CubicEOS._solve_monic_cubic(
            -(1.0 - b),
            a - 3.0 * b**2 - 2.0 * b,
            -(a * b - b**2 - b**3),
        )
        if root > b + 1.0e-12
    )
    if len(roots) < 2:
        return None
    delta_z = roots[-1] - roots[0]
    return delta_z if math.isfinite(delta_z) and delta_z > 0.05 else None


def p95_absolute(errors):
    absolute = sorted(abs(value) for value in errors)
    if not absolute:
        return None
    return absolute[min(len(absolute) - 1, math.ceil(0.95 * len(absolute)) - 1)]


def main() -> int:
    perry = json.loads(
        (REPO_ROOT / "data/perry_properties.json").read_text()
    )["chemicals"]
    errors = {arm: {tr: [] for tr in TR_TARGETS} for arm in ARMS}
    eligible = {tr: 0 for tr in TR_TARGETS}
    estimated_supercritical = {tr: 0 for tr in TR_TARGETS}
    estimated_tr_values = {tr: [] for tr in TR_TARGETS}
    included = excluded_acids = 0

    for cas, row in perry.items():
        if cas == "302-17-0":
            continue
        critical = row.get("critical_constants") or {}
        hvap_rows = row.get("heat_of_vaporization") or []
        psat_rows = row.get("vapor_pressure") or []
        if (
            not hvap_rows
            or not psat_rows
            or not critical.get("Tc_K")
        ):
            continue
        smiles = smiles_for(cas)
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        if molecule is None:
            continue
        if molecule.HasSubstructMatch(COOH):
            excluded_acids += 1
            continue

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

        def reference_hvap(temperature):
            tr = temperature / real_tc
            a, b, c, d = hvap_coefficients
            return a * (1.0 - tr) ** (b + c * tr + d * tr**2) / 1000.0

        def reference_psat(temperature):
            a, b, c, d, exponent = psat_coefficients
            return math.exp(
                a + b / temperature + c * math.log(temperature)
                + d * temperature**exponent
            ) / 1000.0

        tb = bisect_temperature(
            reference_psat,
            P_ATM_KPA,
            psat_low,
            psat_high,
        )
        if tb is None:
            continue
        try:
            curve = nm.estimate_psat(smiles, tb=tb)
            estimated = nm.estimate(smiles, tb=tb)
            if (
                curve.db is None
                or estimated.tc_K is None
                or estimated.pc_kPa is None
                or estimated.vc_cm3_mol is None
            ):
                continue
            omega = LK_omega(
                tb,
                estimated.tc_K,
                estimated.pc_kPa * 1000.0,
            )
            if omega is None or not math.isfinite(omega):
                continue
        except Exception:
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
            base_hvap = curve.dhvap_J_mol(temperature)
            pressure = curve.psat_kPa(temperature)
            if (
                reference <= 0.0
                or base_hvap is None
                or pressure is None
                or pressure <= 0.0
            ):
                continue
            eligible[target_tr] += 1
            estimated_tr = temperature / estimated.tc_K
            estimated_tr_values[target_tr].append(estimated_tr)
            if estimated_tr >= 1.0:
                estimated_supercritical[target_tr] += 1

            delta_z = {
                ARMS[0]: 1.0,
                ARMS[1]: virial_rackett_dz(
                    temperature,
                    pressure,
                    estimated.tc_K,
                    estimated.pc_kPa,
                    omega,
                ),
                ARMS[2]: peng_robinson_dz(
                    temperature,
                    pressure,
                    estimated.tc_K,
                    estimated.pc_kPa,
                    omega,
                ),
                ARMS[3]: ptv_dz(
                    temperature,
                    pressure,
                    estimated.tc_K,
                    estimated.pc_kPa,
                    estimated.vc_cm3_mol / 1000.0,
                    omega,
                ),
            }
            for arm, dz in delta_z.items():
                if dz is None:
                    continue
                prediction = base_hvap * dz
                errors[arm][target_tr].append(
                    100.0 * (prediction / reference - 1.0)
                )
            contributed = True
        included += int(contributed)

    lines = [
        "Estimated-input delta-Z comparison for Nannoolal Hvap",
        f"Perry compounds: {included}; carboxylic acids excluded: {excluded_acids}",
        "Tb is source-backed; Nannoolal estimates Tc/Pc/Vc; LK estimates omega.",
        "Errors are MAPE | median APE | p95 APE | bias percent "
        "(coverage/eligible).",
        "",
    ]
    lines.append(
        f"{'arm':40s}"
        + "".join(f" {'Tr=' + str(tr):>31s}" for tr in TR_TARGETS)
    )
    for arm in ARMS:
        cells = []
        for target_tr in TR_TARGETS:
            values = errors[arm][target_tr]
            stat = statistics(values)
            cell = "-" if stat is None else (
                f"{stat['mape']:.2f} | {stat['median']:.2f} | "
                f"{p95_absolute(values):.2f} | "
                f"{stat['bias']:+.2f} ({stat['n']}/{eligible[target_tr]})"
            )
            cells.append(cell)
        lines.append(
            f"{arm:40s}" + "".join(f" {cell:>31s}" for cell in cells)
        )

    lines.extend(("", "Aggregate over Tr=0.5-0.8:"))
    for arm in ARMS:
        values = [
            error
            for target_tr in TR_TARGETS
            for error in errors[arm][target_tr]
        ]
        stat = statistics(values)
        lines.append(
            f"{arm:40s} MAPE {stat['mape']:.2f}; "
            f"median {stat['median']:.2f}; p95 {p95_absolute(values):.2f}; "
            f"bias {stat['bias']:+.2f}; n={stat['n']}"
        )

    lines.extend(("", "Estimated reduced-temperature coordinate:"))
    for target_tr in TR_TARGETS:
        values = sorted(estimated_tr_values[target_tr])
        middle = values[len(values) // 2]
        lines.append(
            f"real Tr={target_tr:.2f}: median estimated Tr={middle:.4f}; "
            f"estimated Tr>=1 for {estimated_supercritical[target_tr]}/"
            f"{eligible[target_tr]} states"
        )

    report = "\n".join(lines) + "\n"
    print(report, end="")
    OUTPUT_PATH.write_text(report)
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raise SystemExit(main())
