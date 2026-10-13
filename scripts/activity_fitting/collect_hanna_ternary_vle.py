"""Extract measured ternary VLE/VLLE from local ThermoML archives.

Solve composition constraints in their declared mole/mass basis, including
mixed bases. Preserve each observed liquid phase separately. Salt/solvent-basis
datasets and boundary binary states are explicitly excluded. No fitting.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from chemicals.identifiers import search_chemical
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def composition(constraints, molecular_weights):
    a, b = [np.ones(3)], [1.0]
    for kind, index, value in constraints:
        vector = np.zeros(3)
        if kind == "Mole fraction":
            vector[index] = 1
            target = value
        elif kind == "Mass fraction":
            vector = -value*molecular_weights
            vector[index] += molecular_weights[index]
            target = 0
        else:
            raise ValueError(f"unsupported composition basis: {kind}")
        a.append(vector)
        b.append(target)
    a, b = np.array(a), np.array(b)
    x, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
    if rank < 3 or np.max(np.abs(a@x-b)) > 1e-7 or np.min(x) < -1e-8:
        raise ValueError("composition constraints incomplete or inconsistent")
    x = np.maximum(x, 0)
    return (x/x.sum()).tolist()


def extract(path, audit):
    payload = json.loads(path.read_text())
    doi = payload["Citation"]["sDOI"]
    identities = {}
    for c in payload["Compound"]:
        try:
            key = c["sStandardInChIKey"]
            chemical = search_chemical("InChIKey="+key)
            if chemical.InChI_key != key:
                raise ValueError("InChIKey identity mismatch")
            identities[c["RegNum"]["nOrgNum"]] = chemical
        except (KeyError, ValueError) as error:
            audit.append({"doi": doi, "scope": "identity", "reason": str(error)})
    rows = []
    for dataset in payload["PureOrMixtureData"]:
        if len(dataset.get("Component", [])) != 3:
            continue
        tag = {"doi": doi, "dataset": dataset["nPureOrMixtureDataNumber"]}
        phases = {p["ePhase"] for p in dataset.get("PhaseID", [])}
        if "Gas" not in phases:
            audit.append({**tag, "scope": "dataset", "reason": "no gas phase"})
            continue
        try:
            orgs = sorted((c["RegNum"]["nOrgNum"] for c in dataset["Component"]), key=lambda i: identities[i].CASs)
            species = [identities[i] for i in orgs]
            if any(c.CASs == "7647-14-5" for c in species):
                raise ValueError("electrolyte/solvent-basis dataset outside shared molecular vapor-pressure benchmark")
        except (KeyError, ValueError) as error:
            audit.append({**tag, "scope": "dataset", "reason": str(error)})
            continue
        positions = {org: i for i, org in enumerate(orgs)}
        weights = np.array([c.MW for c in species])
        variables = {v["nVarNumber"]: v for v in dataset.get("Variable", [])}
        properties = {p["nPropNumber"]: p for p in dataset["Property"]}
        for number, observation in enumerate(dataset["NumValues"]):
            scalar, fractions = {}, {}

            def add(definition, kind, value, phase):
                if "eTemperature" in kind:
                    scalar["T_K"] = value
                elif "ePressure" in kind:
                    if kind["ePressure"] != "Pressure, kPa":
                        raise ValueError("unrecognized pressure units")
                    scalar["P_Pa"] = 1000*value
                elif "eComponentComposition" in kind:
                    index = positions[definition["RegNum"]["nOrgNum"]]
                    fractions.setdefault(phase, []).append((kind["eComponentComposition"], index, value))
                elif "eSolventComposition" in kind:
                    raise ValueError("solvent-relative composition needs electrolyte-specific interpretation")

            try:
                for c in dataset.get("Constraint", []):
                    definition = c["ConstraintID"]
                    add(definition, definition["ConstraintType"], c["nConstraintValue"], c.get("ConstraintPhaseID", {}).get("eConstraintPhase", "Liquid"))
                for v in observation.get("VariableValue", []):
                    spec = variables[v["nVarNumber"]]
                    definition = spec["VariableID"]
                    add(definition, definition["VariableType"], v["nVarValue"], spec.get("VarPhaseID", {}).get("eVarPhase", "Liquid"))
                for v in observation["PropertyValue"]:
                    spec = properties[v["nPropNumber"]]
                    definition = spec["Property-MethodID"]
                    pg = definition["PropertyGroup"]
                    group = next(value for key, value in pg.items() if key != "tml_elements")
                    name = group["ePropName"]
                    phase = spec["PropPhaseID"]["ePropPhase"]
                    # Lladosa Table 11 explicitly reports x2 and y2 as mole
                    # fractions. ThermoML dataset 18 mislabels these two
                    # property columns as mass fractions; variable x1 and
                    # property y1 are correctly labeled. Honor the original
                    # measured table, not its erroneous archive metadata.
                    if doi == "10.1016/j.fluid.2007.09.014" and dataset["nPureOrMixtureDataNumber"] == 18 and name == "Mass fraction":
                        name = "Mole fraction"
                    if name in ("Mole fraction", "Mass fraction"):
                        if phase == "Gas" and "CHROM" not in group.get("sMethodName", "").upper() and group.get("eMethodName") != "Chromatography":
                            raise ValueError("vapor composition measurement method not established")
                        add(definition, {"eComponentComposition": name}, v["nPropValue"], phase)
                    elif name == "Boiling temperature at pressure P, K":
                        scalar["T_K"] = v["nPropValue"]
                    elif name == "Vapor or sublimation pressure, kPa":
                        scalar["P_Pa"] = 1000*v["nPropValue"]
                if not all(key in scalar for key in ("T_K", "P_Pa")):
                    raise ValueError("missing measured T or P")
                y = composition(fractions["Gas"], weights) if "Gas" in fractions else None
                for phase, constraints in fractions.items():
                    if not phase.startswith("Liquid"):
                        continue
                    x = composition(constraints, weights)
                    if min(x) <= 1e-9:
                        audit.append({**tag, "row": number, "phase": phase, "reason": "binary boundary of ternary system"})
                        continue
                    rows.append({**tag, "row": number, **scalar, "components": [c.CASs for c in species],
                                 "names": [c.common_name for c in species], "x": x, "y": y, "liquid_phase": phase,
                                 "composition_metadata_correction": "Lladosa Table 11: x2/y2 are mole fractions, despite archive mass-fraction labels" if doi == "10.1016/j.fluid.2007.09.014" else "",
                                 "source_composition_constraints": constraints})
            except (KeyError, ValueError, np.linalg.LinAlgError) as error:
                audit.append({**tag, "row": number, "scope": "point", "reason": str(error)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.archive_dir.glob("*.json"))+sorted((ROOT/"data/reference/vapor-liquid-equilibria").glob("*thermoml.json"))
    rows, audit, evidence, seen = [], [], [], set()
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except ValueError:
            continue
        doi = payload["Citation"]["sDOI"]
        if doi in seen:
            continue
        seen.add(doi)
        extracted = extract(path, audit)
        rows.extend(extracted)
        if extracted:
            evidence.append({"doi": doi, "filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pdf = ROOT/"data/reference/vapor-liquid-equilibria/2007-lladosa-dipe-isopropanol-azeotrope-pressure-effect.pdf"
    evidence.append({"filename": str(pdf.relative_to(ROOT)), "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                     "note": "Table 11 and experimental-method mole-fraction statement verify correction of dataset 18 mass-fraction metadata"})
    result = {"rows": rows, "audit": audit, "source_evidence": evidence,
              "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with args.output.open("x") as handle:
        handle.write(json.dumps(result, indent=2)+"\n")
    print('Collected', len(rows), 'liquid-state observations;', len({tuple(r['components']) for r in rows}), 'ternary systems;', sum(r['y'] is not None for r in rows), 'with measured vapor compositions')
    print('Audit:', dict(Counter(r['reason'] for r in audit)))


if __name__ == "__main__":
    main()
