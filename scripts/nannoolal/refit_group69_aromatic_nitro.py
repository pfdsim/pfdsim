#!/usr/bin/env python3
"""Diagnose missing Nannoolal group-69 Tc/Pc/Vc contributions.

This script does not modify :mod:`nannoolal_method`.  It inverts the
published Part-2 equations one compound at a time to find the first-order
aryl-NO2 (group 69, interaction class Q) contribution implied by each
critical-property observation.  The individual implied values show whether
a common contribution is defensible.  The adopted local extension is
Tc=85.41 (*1e-3), Pc=2.621 (*1e-4), Vc=48.527 cm3/mol; this script remains
its reproducible source-level diagnostic and never writes those parameters.

Data contract
-------------
* Perry Table 2-106 physical criticals: 1,3,5-trinitrobenzene and TNT.
* EOS-effective values are admitted only per-property at quality > 0.90:
  2,4-dinitrotoluene has Tc/Pc/Vc q=0.94; 3,5-dinitrotoluene has Tc/Pc
  q=0.94, while its Vc q=0.70 is deliberately omitted.
* PFDSim's experimental online compilation supplies the remaining Tc/Pc
  observations and their suggested uncertainties.
* Tc inversion is reported twice because the Nannoolal Tc equation requires
  Tb: once with a source-backed normal boiling point and once with the
  internally predicted Nannoolal Tb.  Pc/Vc do not depend on Tb.

Units follow the tables in nannoolal_method.py: returned group contributions
are Tc table units (*1e-3), Pc table units (*1e-4), and Vc cm3/mol.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import statistics
import sys
from typing import Iterable, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nannoolal_method as nm


GROUP_ID = 69


@dataclass(frozen=True)
class CriticalRecord:
    name: str
    cas: str
    smiles: str
    source: str
    source_kind: str
    quality: Optional[float]
    tb_K: float
    tb_source: str
    tb_alternatives: tuple[tuple[float, str], ...] = ()
    tc_K: Optional[float] = None
    tc_error_K: Optional[float] = None
    pc_MPa: Optional[float] = None
    pc_error_MPa: Optional[float] = None
    vc_cm3_mol: Optional[float] = None


RECORDS = (
    CriticalRecord(
        "1,3,5-trinitrobenzene", "99-35-4",
        "C1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        "Perry 9th ed. Table 2-106", "physical", 0.96,
        629.9014598557144, "Perry vapor-pressure normal point",
        tb_alternatives=((588.15, "PubChem: 315 degC at 760 mmHg"),),
        tc_K=846.0, pc_MPa=3.39, vc_cm3_mol=479.0,
    ),
    CriticalRecord(
        "2,4,6-trinitrotoluene", "118-96-7",
        "CC1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
        "Perry 9th ed. Table 2-106", "physical", 0.96,
        624.6678653454317, "Perry vapor-pressure normal point",
        tc_K=828.0, pc_MPa=3.04, vc_cm3_mol=572.0,
    ),
    CriticalRecord(
        "2,4-dinitrotoluene", "121-14-2",
        "CC1=C(C=C(C=C1)[N+](=O)[O-])[N+](=O)[O-]",
        "effective_criticals.sqlite", "EOS-effective", 0.94,
        573.15, "Common Chemistry experimental Tb",
        tb_alternatives=((590.0, "Yaws Tb"),),
        tc_K=816.0, pc_MPa=3.35, vc_cm3_mol=439.945554371955,
    ),
    CriticalRecord(
        "3,5-dinitrotoluene", "618-85-9",
        "CC1=CC(=CC(=C1)[N+](=O)[O-])[N+](=O)[O-]",
        "effective_criticals.sqlite; Vc q=0.70 excluded", "EOS-effective", 0.94,
        588.15, "PubChem: 315 degC, extrapolated from vapor-pressure data",
        tb_alternatives=((598.15, "conflicting CRC Organic entry"),),
        tc_K=798.0, pc_MPa=2.20,
    ),
    CriticalRecord(
        "nitrobenzene", "98-95-3", "C1=CC=C(C=C1)[N+](=O)[O-]",
        "PFDSim online experimental compilation", "experimental-online", None,
        483.85, "CRC Organic experimental Tb",
        tc_K=720.0, tc_error_K=8.0, pc_MPa=4.82, pc_error_MPa=0.50,
    ),
    CriticalRecord(
        "3-nitrotoluene", "99-08-1", "CC1=CC(=CC=C1)[N+](=O)[O-]",
        "PFDSim online experimental compilation", "experimental-online", None,
        505.25, "CRC Organic experimental Tb",
        tc_K=731.0, tc_error_K=8.0, pc_MPa=3.05, pc_error_MPa=0.30,
    ),
    CriticalRecord(
        "4-nitrotoluene", "99-99-0", "CC1=CC=C(C=C1)[N+](=O)[O-]",
        "PFDSim online experimental compilation", "experimental-online", None,
        511.81, "CRC Organic experimental Tb",
        tc_K=745.0, tc_error_K=4.0, pc_MPa=3.21, pc_error_MPa=0.20,
    ),
    CriticalRecord(
        "3-chloronitrobenzene", "121-73-3",
        "C1=CC(=CC(=C1)Cl)[N+](=O)[O-]",
        "PFDSim online experimental compilation", "experimental-online", None,
        509.65, "CRC Organic experimental Tb",
        tc_K=744.0, tc_error_K=4.0, pc_MPa=3.98, pc_error_MPa=0.30,
    ),
    CriticalRecord(
        "2-nitroaniline", "88-74-4", "C1=CC=C(C(=C1)N)[N+](=O)[O-]",
        "PFDSim online experimental compilation", "experimental-online", None,
        558.15, "CRC Organic experimental Tb",
        tc_K=800.0, tc_error_K=3.0,
    ),
    CriticalRecord(
        "3,4-dichloronitrobenzene", "99-54-7",
        "C1=CC(=C(C=C1[N+](=O)[O-])Cl)Cl",
        "PFDSim online experimental compilation", "experimental-online", None,
        528.65, "CRC Organic experimental Tb",
        tc_K=758.0, tc_error_K=20.0, pc_MPa=3.60, pc_error_MPa=0.50,
    ),
    CriticalRecord(
        "2-chloronitrobenzene", "88-73-3",
        "C1=CC=C(C(=C1)[N+](=O)[O-])Cl",
        "PFDSim online experimental compilation", "experimental-online", None,
        519.35, "CRC Organic experimental Tb",
        tc_K=757.0, tc_error_K=15.0, pc_MPa=3.98, pc_error_MPa=0.50,
    ),
    CriticalRecord(
        "4-chloronitrobenzene", "100-00-5",
        "C1=CC(=CC=C1[N+](=O)[O-])Cl",
        "PFDSim online experimental compilation", "experimental-online", None,
        515.3722222222223, "PubChem reported Tb selected by PFDsim",
        tc_K=751.0, tc_error_K=15.0, pc_MPa=3.98, pc_error_MPa=0.50,
    ),
)


def _fragment(record: CriticalRecord) -> nm.Fragmentation:
    mol = Chem.MolFromSmiles(record.smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES for {record.name}")
    frag = nm.fragment(mol)
    if frag.groups.get(GROUP_ID, 0) < 1:
        raise ValueError(f"{record.name} does not contain Nannoolal group 69")
    return frag


def _known_sum(frag: nm.Fragmentation, prop: str) -> float:
    """Return S without group 69; retain every other published term."""
    idx, scale = nm._PROP_INDEX[prop], nm._PROP_SCALE[prop]
    total = 0.0
    for gid, frequency in frag.groups.items():
        if gid == GROUP_ID:
            continue
        value = nm.GROUP_CONTRIBUTIONS[gid][idx]
        if prop in nm.LOCAL_REFITS.get(gid, {}):
            value = nm.LOCAL_REFITS[gid][prop]
        if value is None:
            raise ValueError(f"{prop}: group {gid} is also unavailable")
        total += frequency * value * scale
    for silicon in frag.si_atoms:
        gid = nm._si_group_id(silicon, prop)
        value = nm.GROUP_CONTRIBUTIONS[gid][idx]
        if value is None:
            raise ValueError(f"{prop}: silicon group {gid} is unavailable")
        total += value * scale
    for cid, frequency in frag.corrections.items():
        if prop == "tb" and cid == 217:
            continue
        value = nm.CORRECTION_CONTRIBUTIONS[cid][idx]
        if value is not None:
            total += frequency * value * scale
    # Missing Q interactions contribute zero under the published Part-2
    # convention; any available non-Q interactions remain included.
    total += nm._interaction_sum(frag, prop, local_refits=True)
    return total


def implied_tc(frag: nm.Fragmentation, tb_K: float, tc_K: float) -> float:
    ratio_term = tc_K / tb_K - nm.TC_B
    if ratio_term <= 0.0:
        raise ValueError("Tc/Tb is outside the invertible model range")
    powered = 1.0 / ratio_term - nm.TC_A
    if powered <= 0.0:
        raise ValueError("Tc observation implies a non-positive group sum")
    target_sum = powered ** (1.0 / nm.TC_C)
    frequency = frag.groups[GROUP_ID]
    return (target_sum - _known_sum(frag, "tc")) / (frequency * nm._TC_SCALE)


def implied_pc(frag: nm.Fragmentation, pc_MPa: float) -> float:
    molar_mass = Descriptors.MolWt(frag.mol)
    pc_kPa = pc_MPa * 1000.0
    target_sum = math.sqrt(molar_mass ** nm.PC_B / pc_kPa) - nm.PC_A
    frequency = frag.groups[GROUP_ID]
    return (target_sum - _known_sum(frag, "pc")) / (frequency * nm._PC_SCALE)


def implied_vc(frag: nm.Fragmentation, vc_cm3_mol: float) -> float:
    target_sum = (vc_cm3_mol - nm.VC_B) / frag.n_atoms ** (-nm.VC_A)
    frequency = frag.groups[GROUP_ID]
    return (target_sum - _known_sum(frag, "vc")) / frequency


def predicted_tc(frag: nm.Fragmentation, tb_K: float, contribution: float) -> float:
    total = (
        _known_sum(frag, "tc")
        + frag.groups[GROUP_ID] * contribution * nm._TC_SCALE
    )
    return tb_K * (nm.TC_B + 1.0 / (nm.TC_A + total ** nm.TC_C))


def predicted_pc(frag: nm.Fragmentation, contribution: float) -> float:
    total = (
        _known_sum(frag, "pc")
        + frag.groups[GROUP_ID] * contribution * nm._PC_SCALE
    )
    return Descriptors.MolWt(frag.mol) ** nm.PC_B / (nm.PC_A + total) ** 2 / 1000.0


def predicted_vc(frag: nm.Fragmentation, contribution: float) -> float:
    total = _known_sum(frag, "vc") + frag.groups[GROUP_ID] * contribution
    return total * frag.n_atoms ** (-nm.VC_A) + nm.VC_B


def _range(func, value: float, error: Optional[float]) -> str:
    if error is None:
        return ""
    values = sorted((func(value - error), func(value + error)))
    return f" [{values[0]:.2f}, {values[1]:.2f}]"


def _summary(label: str, values: Iterable[float]) -> None:
    values = list(values)
    mean = statistics.fmean(values)
    median = statistics.median(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    mad = statistics.median(abs(value - median) for value in values)
    print(
        f"{label}: n={len(values)}, mean={mean:.3f}, median={median:.3f}, "
        f"SD={stdev:.3f}, MAD={mad:.3f}, min={min(values):.3f}, "
        f"max={max(values):.3f}"
    )


def main() -> int:
    anchored_tc_values: list[float] = []
    predictive_tc_values: list[float] = []
    pc_values: list[float] = []
    vc_values: list[float] = []
    plain_mono_tc_values: list[float] = []
    plain_mono_pc_values: list[float] = []

    print("Group 69 individual implied contributions")
    print("Tc units=*1e-3; Pc units=*1e-4; Vc units=cm3/mol")
    print("EOS-effective records are explicitly labeled with per-record quality.\n")
    header = (
        f"{'compound':27s} {'n69':>3s} {'source':22s} "
        f"{'Tc69 anchored':>22s} {'Tc69 predictive':>17s} "
        f"{'Pc69':>22s} {'Vc69':>10s}"
    )
    print(header)
    print("-" * len(header))

    for record in RECORDS:
        frag = _fragment(record)
        internal_tb = nm.estimate(record.smiles).tb_K
        source = record.source_kind
        if record.quality is not None:
            source += f" q={record.quality:.2f}"

        tc_anchor_text = tc_predictive_text = pc_text = vc_text = "—"
        if record.tc_K is not None:
            tc_anchor = implied_tc(frag, record.tb_K, record.tc_K)
            tc_predictive = implied_tc(frag, internal_tb, record.tc_K)
            assert math.isclose(
                predicted_tc(frag, record.tb_K, tc_anchor),
                record.tc_K,
                rel_tol=1.0e-12,
            )
            anchored_tc_values.append(tc_anchor)
            predictive_tc_values.append(tc_predictive)
            if frag.groups[GROUP_ID] == 1 and frag.interaction_classes == ["Q"]:
                plain_mono_tc_values.append(tc_anchor)
            tc_anchor_text = (
                f"{tc_anchor:8.2f}"
                + _range(
                    lambda value: implied_tc(frag, record.tb_K, value),
                    record.tc_K,
                    record.tc_error_K,
                )
            )
            tc_predictive_text = f"{tc_predictive:8.2f}"
        if record.pc_MPa is not None:
            pc_value = implied_pc(frag, record.pc_MPa)
            assert math.isclose(
                predicted_pc(frag, pc_value),
                record.pc_MPa,
                rel_tol=1.0e-12,
            )
            pc_values.append(pc_value)
            if frag.groups[GROUP_ID] == 1 and frag.interaction_classes == ["Q"]:
                plain_mono_pc_values.append(pc_value)
            pc_text = (
                f"{pc_value:8.2f}"
                + _range(
                    lambda value: implied_pc(frag, value),
                    record.pc_MPa,
                    record.pc_error_MPa,
                )
            )
        if record.vc_cm3_mol is not None:
            vc_value = implied_vc(frag, record.vc_cm3_mol)
            assert math.isclose(
                predicted_vc(frag, vc_value),
                record.vc_cm3_mol,
                rel_tol=1.0e-12,
            )
            vc_values.append(vc_value)
            vc_text = f"{vc_value:8.2f}"

        print(
            f"{record.name:27.27s} {frag.groups[GROUP_ID]:3d} "
            f"{source:22.22s} {tc_anchor_text:>22s} "
            f"{tc_predictive_text:>17s} {pc_text:>22s} {vc_text:>10s}"
        )

    print("\nAcross-compound dispersion")
    _summary("Tc69 anchored", anchored_tc_values)
    _summary("Tc69 predictive", predictive_tc_values)
    _summary("Pc69", pc_values)
    _summary("Vc69", vc_values)
    print("\nClean first-order diagnostic: one Q group, no second interaction class")
    _summary("Tc69 plain mononitro", plain_mono_tc_values)
    _summary("Pc69 plain mononitro", plain_mono_pc_values)
    print(
        "\nAdopted local extension in nannoolal_method.py: "
        f"{nm.LOCAL_GROUP_EXTENSIONS.get(GROUP_ID)}"
    )

    print("\nTb conditioning used for anchored Tc inversion")
    for record in RECORDS:
        internal_tb = nm.estimate(record.smiles).tb_K
        print(
            f"{record.cas:10s} {record.name:27s} anchored={record.tb_K:8.3f} K "
            f"internal={internal_tb:8.3f} K; {record.tb_source}"
        )
        if record.tc_K is not None:
            frag = _fragment(record)
            for alternate_tb, source in record.tb_alternatives:
                contribution = implied_tc(frag, alternate_tb, record.tc_K)
                print(
                    f"{'':39s} alternate={alternate_tb:8.3f} K "
                    f"-> Tc69={contribution:8.3f}; {source}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
