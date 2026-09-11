"""Compare Nannoolal Hvap with corresponding-states and Watson estimates.

The comparison uses Perry 9th Table 2-150 correlations and independent CRC
points.  Carboxylic acids are excluded because Psat slopes measure their
apparent liquid-to-associated-vapor enthalpy rather than the calorimetric
monomer vaporization enthalpy.

The corresponding-states relation is

    Hvap/(R*Tc) = 7.08*(1-Tr)**0.354 + 10.95*omega*(1-Tr)**0.456.

Three information tiers are retained for that relation: real Tc/omega, real
Tc with omega predicted from a fully estimated Nannoolal Tb/Tc/Pc chain and
Lee-Kesler, and fully predicted Tc/omega.  Nannoolal is evaluated with both a
real and internally estimated Tb, with dZ=1, the previously selected
Tsonopoulos-virial/Rackett correction, and PTV to expose its mean-field
near-critical closure.  Watson is tested with a perfect Perry-curve Hvap(Tb)
anchor and with independent CRC Hvap(Tb), using real and estimated Tc.
"""

from pathlib import Path
import json
import math
import sys
import warnings

from chemicals.acentric import LK_omega
from chemicals.identifiers import search_chemical
from chemicals.phase_change import Hvap_data_CRC
import numpy as np
from rdkit import Chem


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import nannoolal_method as nm
from physical_constants import R_J_MOL_K as R


TR_TARGETS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99)
P_ATM_KPA = 101.325
COOH = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
OUTPUT_PATH = REPO_ROOT / "outputs/nannoolal_vs_hvap_formula.txt"

ARMS = (
    "NN real Tb, dZ=1",
    "NN estimated Tb, dZ=1",
    "NN real Tb, virial/Rackett real TcPcOmega",
    "NN real Tb, virial/Rackett estimated TcPcOmega",
    "NN estimated chain, virial/Rackett",
    "NN real Tb, PTV real TcPcVcOmega",
    "NN estimated chain, PTV estimated TcPcVcOmega",
    "Formula real Tc/omega",
    "Formula real Tc, predicted omega",
    "Formula predicted Tc/omega",
    "Watson Perry HvapTb, real Tc",
    "Watson Perry HvapTb, estimated Tc",
    "Watson CRC HvapTb, real Tc",
    "Watson CRC HvapTb, estimated Tc",
    "Current Trouton constant, real Tb",
    "Watson-scaled Trouton, real Tb/Tc",
    "Watson-scaled Trouton, estimated Tb/Tc",
)


def smiles_for(cas):
    try:
        return search_chemical(cas).smiles
    except Exception:
        return None


def bisect_temperature(function, target, low, high):
    f_low, f_high = function(low), function(high)
    if not f_low <= target <= f_high:
        return None
    for _ in range(80):
        middle = 0.5 * (low + high)
        if function(middle) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def corresponding_states_hvap(temperature, tc, omega):
    tr = temperature / tc
    if not 0.0 < tr < 1.0:
        return None
    tau = 1.0 - tr
    return R * tc * (
        7.08 * tau**0.354 + 10.95 * omega * tau**0.456
    )


def watson_hvap(temperature, reference_hvap, reference_temperature, tc):
    if (
        reference_hvap is None
        or reference_temperature is None
        or reference_hvap <= 0.0
        or not 0.0 < reference_temperature < tc
    ):
        return None
    if temperature >= tc:
        return 0.0
    if temperature <= 0.0:
        return None
    return reference_hvap * (
        (1.0 - temperature / tc) / (1.0 - reference_temperature / tc)
    ) ** 0.38


def trouton_hvap_tb(tb):
    coefficient = 0.075 if tb < 250.0 else 0.095 if tb > 400.0 else 0.088
    return 1000.0 * coefficient * tb


def virial_rackett_dz(temperature, pressure_kpa, tc, pc_kpa, omega):
    tr = temperature / tc
    if not 0.0 < tr < 0.995 or pressure_kpa <= 0.0:
        return None
    b0 = (
        0.1445 - 0.330 / tr - 0.1385 / tr**2
        - 0.0121 / tr**3 - 0.000607 / tr**8
    )
    b1 = 0.0637 + 0.331 / tr**2 - 0.423 / tr**3 - 0.008 / tr**8
    pressure_pa = pressure_kpa * 1.0e3
    pc_pa = pc_kpa * 1.0e3
    second_virial = R * tc / pc_pa * (b0 + omega * b1)
    z_vapor = 1.0 + second_virial * pressure_pa / (R * temperature)
    z_ra = 0.29056 - 0.08775 * omega
    if z_ra <= 0.0:
        return None
    liquid_volume = (
        R * tc / pc_pa
        * z_ra ** (1.0 + (1.0 - tr) ** (2.0 / 7.0))
    )
    dz = z_vapor - pressure_pa * liquid_volume / (R * temperature)
    return dz if dz > 0.05 else None


def ptv_dz(temperature, pressure_kpa, tc, pc_kpa, vc_m3_kmol, omega):
    tr = temperature / tc
    if not 0.0 < tr < 1.0 or pressure_kpa <= 0.0:
        return None
    pressure = pressure_kpa * 1.0e3
    pc = pc_kpa * 1.0e3
    zc = pc * (vc_m3_kmol / 1000.0) / (R * tc)
    omega_a = 0.66121 - 0.76105 * zc
    omega_b = 0.02207 + 0.20868 * zc
    omega_c = 0.57765 - 1.87080 * zc
    alpha_coefficient = (
        0.46283 + 3.58230 * omega * zc + 8.19417 * (omega * zc) ** 2
    )
    alpha = (1.0 + alpha_coefficient * (1.0 - math.sqrt(tr))) ** 2
    a = omega_a * R**2 * tc**2 / pc * alpha
    b = omega_b * R * tc / pc
    c = omega_c * R * tc / pc
    roots = np.roots([
        pressure,
        pressure * c - R * temperature,
        a - pressure * (b**2 + 2.0 * b * c) - R * temperature * (b + c),
        pressure * b**2 * c + R * temperature * b * c - a * b,
    ])
    volumes = sorted(
        root.real
        for root in roots
        if abs(root.imag) < 1.0e-9 and root.real > b
    )
    if len(volumes) < 2:
        return None
    dz = pressure * (volumes[-1] - volumes[0]) / (R * temperature)
    return dz if dz > 0.05 else None


def statistics(errors):
    if not errors:
        return None
    absolute = sorted(abs(value) for value in errors)
    return {
        "mape": sum(absolute) / len(absolute),
        "median": absolute[len(absolute) // 2],
        "bias": sum(errors) / len(errors),
        "n": len(errors),
    }


def main():
    perry = json.loads(
        (REPO_ROOT / "data/perry_properties.json").read_text()
    )["chemicals"]
    crc = {cas: row for cas, row in Hvap_data_CRC.iterrows()}
    bins = {arm: {tr: [] for tr in TR_TARGETS} for arm in ARMS}
    crc_bins = {arm: {"Tb": [], "298 K": []} for arm in ARMS}
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
            or not all(
                critical.get(key)
                for key in ("Tc_K", "Pc_MPa", "Vc_m3_per_kmol")
            )
            or critical.get("omega") is None
        ):
            continue
        smiles = smiles_for(cas)
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        if molecule is None:
            continue
        if molecule.HasSubstructMatch(COOH):
            excluded_acids += 1
            continue

        tc = critical["Tc_K"]
        pc_kpa = critical["Pc_MPa"] * 1.0e3
        omega = critical["omega"]
        hvap_row = hvap_rows[0]
        psat_row = psat_rows[0]
        hvap_coefficients = (
            list(hvap_row["coefficients"]) + [0.0] * 4
        )[:4]
        hvap_low, hvap_high = hvap_row["T_min_K"], hvap_row["T_max_K"]
        psat_coefficients = (
            list(psat_row["coefficients"]) + [0.0] * 5
        )[:5]
        if psat_coefficients[4] == 0.0:
            psat_coefficients[4] = 1.0
        psat_low, psat_high = psat_row["T_min_K"], psat_row["T_max_K"]

        def reference_hvap(temperature):
            tr = temperature / tc
            a, b, c, d = hvap_coefficients
            return a * (1.0 - tr) ** (b + c * tr + d * tr * tr) / 1000.0

        def reference_psat(temperature):
            a, b, c, d, exponent = psat_coefficients
            return math.exp(
                a + b / temperature + c * math.log(temperature)
                + d * temperature**exponent
            ) / 1000.0

        tb = bisect_temperature(
            reference_psat, P_ATM_KPA, psat_low, psat_high
        )
        if tb is None:
            continue
        perry_hvap_tb = (
            reference_hvap(tb)
            if hvap_low <= tb <= hvap_high
            else None
        )
        crc_row = crc.get(cas)
        crc_tb = None if crc_row is None else crc_row.get("Tb")
        crc_hvap_tb = None if crc_row is None else crc_row.get("HvapTb")
        if crc_tb is not None and not math.isfinite(float(crc_tb)):
            crc_tb = None
        if crc_hvap_tb is not None and not math.isfinite(float(crc_hvap_tb)):
            crc_hvap_tb = None
        try:
            nn_real_tb = nm.estimate_psat(smiles, tb=tb)
            nn_estimated_tb = nm.estimate_psat(smiles)
            estimated_from_real_tb = nm.estimate(smiles, tb=tb)
            estimated = nm.estimate(smiles)
            if (
                nn_real_tb.db is None
                or nn_estimated_tb.db is None
                or estimated_from_real_tb.tc_K is None
                or estimated_from_real_tb.pc_kPa is None
                or estimated.tb_K is None
                or estimated.tc_K is None
                or estimated.pc_kPa is None
            ):
                continue
            trouton_real_tb = trouton_hvap_tb(tb)
            trouton_estimated_tb = trouton_hvap_tb(estimated.tb_K)
            real_tb_predicted_omega = LK_omega(
                tb,
                estimated_from_real_tb.tc_K,
                estimated_from_real_tb.pc_kPa * 1.0e3,
            )
            predicted_omega = LK_omega(
                estimated.tb_K,
                estimated.tc_K,
                estimated.pc_kPa * 1.0e3,
            )
            if (
                real_tb_predicted_omega is None
                or not math.isfinite(real_tb_predicted_omega)
                or predicted_omega is None
                or not math.isfinite(predicted_omega)
            ):
                continue
        except Exception:
            continue

        def estimates_at(temperature):
            nn_real = nn_real_tb.dhvap_J_mol(temperature)
            nn_estimated = nn_estimated_tb.dhvap_J_mol(temperature)
            pressure_real = nn_real_tb.psat_kPa(temperature)
            pressure_estimated = nn_estimated_tb.psat_kPa(temperature)
            if None in (nn_real, nn_estimated, pressure_real, pressure_estimated):
                return None
            dz_real = virial_rackett_dz(
                temperature, pressure_real, tc, pc_kpa, omega
            )
            dz_real_tb_estimated_criticals = virial_rackett_dz(
                temperature,
                pressure_real,
                estimated_from_real_tb.tc_K,
                estimated_from_real_tb.pc_kPa,
                real_tb_predicted_omega,
            )
            dz_estimated = virial_rackett_dz(
                temperature,
                pressure_estimated,
                estimated.tc_K,
                estimated.pc_kPa,
                predicted_omega,
            )
            dz_ptv_real = ptv_dz(
                temperature,
                pressure_real,
                tc,
                pc_kpa,
                critical["Vc_m3_per_kmol"],
                omega,
            )
            dz_ptv_estimated = (
                None
                if estimated.vc_cm3_mol is None
                else ptv_dz(
                    temperature,
                    pressure_estimated,
                    estimated.tc_K,
                    estimated.pc_kPa,
                    estimated.vc_cm3_mol / 1000.0,
                    predicted_omega,
                )
            )
            return {
                ARMS[0]: nn_real,
                ARMS[1]: nn_estimated,
                ARMS[2]: nn_real * dz_real if dz_real is not None else None,
                ARMS[3]: (
                    nn_real * dz_real_tb_estimated_criticals
                    if dz_real_tb_estimated_criticals is not None else None
                ),
                ARMS[4]: (
                    nn_estimated * dz_estimated
                    if dz_estimated is not None else None
                ),
                ARMS[5]: (
                    nn_real * dz_ptv_real
                    if dz_ptv_real is not None else None
                ),
                ARMS[6]: (
                    nn_estimated * dz_ptv_estimated
                    if dz_ptv_estimated is not None else None
                ),
                ARMS[7]: corresponding_states_hvap(temperature, tc, omega),
                ARMS[8]: corresponding_states_hvap(
                    temperature, tc, predicted_omega
                ),
                ARMS[9]: corresponding_states_hvap(
                    temperature, estimated.tc_K, predicted_omega
                ),
                ARMS[10]: watson_hvap(
                    temperature, perry_hvap_tb, tb, tc
                ),
                ARMS[11]: watson_hvap(
                    temperature,
                    perry_hvap_tb,
                    tb,
                    estimated_from_real_tb.tc_K,
                ),
                ARMS[12]: watson_hvap(
                    temperature, crc_hvap_tb, crc_tb, tc
                ),
                ARMS[13]: watson_hvap(
                    temperature,
                    crc_hvap_tb,
                    crc_tb,
                    estimated_from_real_tb.tc_K,
                ),
                ARMS[14]: trouton_real_tb,
                ARMS[15]: watson_hvap(
                    temperature, trouton_real_tb, tb, tc
                ),
                ARMS[16]: watson_hvap(
                    temperature,
                    trouton_estimated_tb,
                    estimated.tb_K,
                    estimated.tc_K,
                ),
            }

        contributed = False
        for target_tr in TR_TARGETS:
            temperature = target_tr * tc
            if not (
                hvap_low + 1.0 < temperature < hvap_high - 1.0
                and psat_low + 1.0 < temperature < psat_high - 1.0
            ):
                continue
            reference = reference_hvap(temperature)
            estimates = estimates_at(temperature)
            if reference <= 0.0 or estimates is None:
                continue
            for arm, value in estimates.items():
                if value is not None and math.isfinite(value) and value >= 0.0:
                    bins[arm][target_tr].append(100.0 * (value / reference - 1.0))
            contributed = True
        included += int(contributed)

        if crc_row is None:
            continue
        for label, temperature, field in (
            ("Tb", crc_row.get("Tb"), "HvapTb"),
            ("298 K", 298.15, "Hvap298"),
        ):
            reference = crc_row.get(field)
            if not (
                temperature
                and reference
                and reference == reference
                and hvap_low < temperature < hvap_high
                and psat_low + 1.0 < temperature < psat_high - 1.0
            ):
                continue
            estimates = estimates_at(temperature)
            if estimates is None:
                continue
            for arm, value in estimates.items():
                if value is not None and math.isfinite(value) and value >= 0.0:
                    crc_bins[arm][label].append(
                        100.0 * (value / reference - 1.0)
                    )

    lines = [
        f"Nannoolal vs corresponding-states and Watson Hvap: "
        f"{included} Perry compounds",
        f"Carboxylic acids excluded: {excluded_acids}",
        "Errors are MAPE | median APE | bias percent (n).",
        "",
        "Against Perry 2-150 correlations:",
    ]
    lines.append(
        f"{'arm':48s}"
        + "".join(f" {'Tr=' + str(tr):>30s}" for tr in TR_TARGETS)
    )
    for arm in ARMS:
        cells = []
        for target_tr in TR_TARGETS:
            stat = statistics(bins[arm][target_tr])
            cells.append(
                "-" if stat is None else (
                    f"{stat['mape']:.2f} | {stat['median']:.2f} | "
                    f"{stat['bias']:+.2f} ({stat['n']})"
                )
            )
        lines.append(
            f"{arm:48s}" + "".join(f" {cell:>30s}" for cell in cells)
        )

    for aggregate_label, aggregate_targets in (
        ("Perry aggregate over Tr=0.5-0.8:", TR_TARGETS[:4]),
        ("Perry near-critical aggregate over Tr=0.8-0.99:", TR_TARGETS[3:]),
    ):
        lines.extend(("", aggregate_label))
        for arm in ARMS:
            aggregate = [
                error
                for target_tr in aggregate_targets
                for error in bins[arm][target_tr]
            ]
            stat = statistics(aggregate)
            if stat is None:
                lines.append(f"{arm:52s} unavailable (n=0)")
            else:
                lines.append(
                    f"{arm:52s} MAPE {stat['mape']:.2f}  "
                    f"median {stat['median']:.2f}  bias {stat['bias']:+.2f} "
                    f"(n={stat['n']})"
                )

    lines.extend(("", "Against CRC experimental points:"))
    lines.append(f"{'arm':48s} {'at Tb':>30s} {'at 298 K':>30s}")
    for arm in ARMS:
        cells = []
        for label in ("Tb", "298 K"):
            stat = statistics(crc_bins[arm][label])
            cells.append(
                "-" if stat is None else (
                    f"{stat['mape']:.2f} | {stat['median']:.2f} | "
                    f"{stat['bias']:+.2f} ({stat['n']})"
                )
            )
        lines.append(f"{arm:48s} {cells[0]:>30s} {cells[1]:>30s}")

    report = "\n".join(lines) + "\n"
    print(report, end="")
    OUTPUT_PATH.write_text(report)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
