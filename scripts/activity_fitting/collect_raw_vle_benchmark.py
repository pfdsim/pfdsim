"""Collect measured binary VLE from local/NIST archives and script tables.

Pass the directory of downloaded ThermoML JSONs. Existing local ThermoMLs
are also scanned. Only measured bubble pressures/temperatures and measured
vapor mole fractions are retained. Pure points are separate diagnostics.
The output path must be new. No interaction fits are executed or modified.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from chemicals.identifiers import search_chemical

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def literal_assignments(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def identifier_kind(definition):
    kind = definition.get("VariableType", definition.get("ConstraintType", {}))
    if "eTemperature" in kind:
        return "T_K"
    if "ePressure" in kind:
        return "P_Pa"
    if kind.get("eComponentComposition") == "Mole fraction":
        return "composition"
    if kind.get("eComponentComposition") == "Mass fraction":
        return "mass_composition"
    return None


def thermoml(path, audit):
    payload = json.loads(path.read_text())
    doi = payload["Citation"].get("sDOI", path.stem)
    compounds = {}
    for item in payload["Compound"]:
        try:
            chemical = search_chemical("InChIKey=" + item["sStandardInChIKey"])
            if chemical.InChI_key != item["sStandardInChIKey"]:
                raise ValueError("identity key mismatch")
            compounds[item["RegNum"]["nOrgNum"]] = chemical.CASs
        except (ValueError, KeyError) as error:
            audit.append({"source": doi, "kind": "identity", "name": item.get("sCommonName"), "reason": str(error)})
    primary, vapor = [], []
    for dataset in payload["PureOrMixtureData"]:
        if len(dataset.get("Component", [])) != 2:
            continue
        try:
            orgs = [c["RegNum"]["nOrgNum"] for c in dataset["Component"]]
            pair = tuple(sorted(compounds[o] for o in orgs))
        except KeyError:
            continue
        phases = {p["ePhase"] for p in dataset.get("PhaseID", [])}
        if not ("Gas" in phases and "Liquid" in phases):
            continue
        definitions = {v["nVarNumber"]: v["VariableID"] for v in dataset.get("Variable", [])}

        def liquid_fraction(definition, value, kind):
            cas = compounds[definition["RegNum"]["nOrgNum"]]
            fraction = float(value)
            if kind == "mass_composition":
                other = pair[1] if cas == pair[0] else pair[0]
                moles = fraction/search_chemical(cas).MW
                other_moles = (1-fraction)/search_chemical(other).MW
                fraction = moles/(moles+other_moles)
            return fraction if cas == pair[0] else 1-fraction

        fixed = {}
        for c in dataset.get("Constraint", []):
            name = identifier_kind(c["ConstraintID"])
            if name in ("T_K", "P_Pa"):
                fixed[name] = float(c["nConstraintValue"]) * (1000 if name == "P_Pa" else 1)
            elif name in ("composition", "mass_composition"):
                fixed["x1"] = liquid_fraction(c["ConstraintID"], c["nConstraintValue"], name)
        for prop in dataset["Property"]:
            group = prop["Property-MethodID"]["PropertyGroup"]
            group = group[next(k for k in group if k != "tml_elements")]
            name = group["ePropName"]
            phase = prop["PropPhaseID"]["ePropPhase"]
            pressure = name == "Vapor or sublimation pressure, kPa"
            temperature = name == "Boiling temperature at pressure P, K"
            y_property = name == "Mole fraction" and phase == "Gas"
            if not (pressure or temperature or y_property):
                continue
            # These Barker-derived y columns are not independently measured.
            if y_property and doi in ("10.1016/j.fluid.2012.01.007", "10.1016/j.fluid.2016.05.031"):
                audit.append({"source": doi, "dataset": dataset["nPureOrMixtureDataNumber"], "kind": "derived_vapor_composition", "reason": "source y is calculated from its pressure/GE analysis"})
                continue
            method = group.get("sMethodName", group.get("eMethodName", "unspecified"))
            for number, observation in enumerate(dataset["NumValues"]):
                row = {"doi": doi, "cas1": pair[0], "cas2": pair[1], **fixed,
                       "dataset": dataset["nPureOrMixtureDataNumber"], "row": number,
                       "measurement_method": method}
                for value in observation.get("VariableValue", []):
                    definition = definitions[value["nVarNumber"]]
                    kind = identifier_kind(definition)
                    if kind in ("T_K", "P_Pa"):
                        row[kind] = float(value["nVarValue"]) * (1000 if kind == "P_Pa" else 1)
                    elif kind in ("composition", "mass_composition"):
                        row["x1"] = liquid_fraction(definition, value["nVarValue"], kind)
                        if kind == "mass_composition":
                            row["source_mass_fraction"] = float(value["nVarValue"])
                observed = next(v for v in observation["PropertyValue"] if v["nPropNumber"] == prop["nPropNumber"])
                value = float(observed["nPropValue"])
                if pressure:
                    row["P_Pa"] = value*1000
                    row["kind"] = "Pxy"
                elif temperature:
                    row["T_K"] = value
                    row["kind"] = "Txy"
                else:
                    cas = compounds[prop["Property-MethodID"]["RegNum"]["nOrgNum"]]
                    row["y1"] = value if cas == pair[0] else 1-value
                if "x1" not in row:
                    audit.append({"source": doi, "dataset": row["dataset"], "kind": "unsupported_composition", "reason": "no liquid mole-fraction variable"})
                    continue
                (vapor if y_property else primary).append(row)
    for row in vapor:
        candidates = [p for p in primary if (p["cas1"], p["cas2"]) == (row["cas1"], row["cas2"])
                      and abs(p["x1"]-row["x1"]) < 1e-7
                      and all(key not in row or abs(p[key]-row[key]) < (1e-4 if key == "T_K" else .1) for key in ("T_K", "P_Pa"))]
        if len(candidates) != 1:
            audit.append({"source": doi, "dataset": row["dataset"], "row": row["row"], "kind": "vapor_join", "reason": f"expected one matching measured bubble state, found {len(candidates)}"})
            continue
        candidate = candidates[0]
        if "y1" in candidate and abs(candidate["y1"]-row["y1"]) > 1e-7:
            raise ValueError(f"Conflicting measured vapor fractions: {doi}")
        candidate["y1"] = row["y1"]
        candidate["vapor_dataset"] = row["dataset"]
        candidate["vapor_measurement_method"] = row["measurement_method"]
    return doi, primary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit, rows, evidence = [], [], []
    visited = set()
    files = sorted(args.archive_dir.glob("*.json")) + sorted((ROOT / "data/reference/vapor-liquid-equilibria").glob("*thermoml.json"))
    for path in files:
        doi, raw = thermoml(path, audit)
        if doi in visited:
            continue
        visited.add(doi)
        rows.extend(raw)
        evidence.append({"doi": doi, "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "measured_bubble_records": len(raw)})
    # Source-local raw tables in safe, literal-only fitting-script assignments.
    source = ROOT / "scripts/activity_fitting/fit_fendu_2025_activity_parameters.py"
    values = literal_assignments(source)
    for key, cas in (("TABLE_3", ("78-83-1", "71-36-3")), ("TABLE_4", ("78-83-1", "104-76-7"))):
        for x, points in values[key].items():
            for t, p in points:
                rows.append({"doi": "10.3390/pr13072034", "cas1": cas[0], "cas2": cas[1], "x1": x, "T_K": t, "P_Pa": p*1000, "kind": "Pxy", "dataset": key})
    evidence.append({"doi": "10.3390/pr13072034", "file": str(source.relative_to(ROOT)), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "note": "literal measured tables; no regression executed"})
    source = ROOT / "scripts/activity_fitting/refit_uniquac_common_basis.py"
    values = literal_assignments(source)
    for x, t in values["HARTANTO"]:
        rows.append({"doi": "10.13140/RG.2.1.4542.3766", "cas1": "64-17-5", "cas2": "111-27-3", "x1": x, "T_K": t, "P_Pa": 100000, "kind": "Txy", "dataset": "Table 2"})
    for x, y, t, *_ in values["ARCE"]:
        rows.append({"doi": "10.1021/je00020a063", "cas1": "64-17-5", "cas2": "111-87-5", "x1": x, "y1": y, "T_K": t, "P_Pa": 101320, "kind": "Txy", "dataset": "Table 3 ethanol"})
    for key, cas, temperatures in (("MUSTAIN_BUTANOL", "71-36-3", (333.15,338.15,343.15)), ("MUSTAIN_ISOBUTANOL", "78-83-1", (313.15,318.15,323.15))):
        for x, *pressures in values[key]:
            for t, p in zip(temperatures, pressures):
                rows.append({"doi": "10.1021/acs.jced.1c00937", "cas1": cas, "cas2": "56-81-5", "x1": x, "T_K": t, "P_Pa": p*1000, "kind": "Pxy", "dataset": key})
    evidence.append({"file": str(source.relative_to(ROOT)), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "note": "literal-only Hartanto, Arce, Mustain measurement tables"})
    # Source correlation overrides retained for these already extracted tables.
    source_psat = {
        "10.13140/RG.2.1.4542.3766": {
            "64-17-5": {"A":5.33675,"B":1648.22,"C_K":230.918-273.15,"base":10,"factor_Pa":100000},
            "111-27-3": {"A":4.18948,"B":1295.59,"C_K":152.510-273.15,"base":10,"factor_Pa":100000}},
        "10.1021/je00020a063": {
            "64-17-5": {"A":7.16879,"B":1552.601,"C_K":222.419-273.15,"base":10,"factor_Pa":1000},
            "111-87-5": {"A":5.88511,"B":1264.322,"C_K":130.73-273.15,"base":10,"factor_Pa":1000}},
        "10.1021/acs.jced.1c00937": {
            "71-36-3": {"A":4.64930,"B":1395.140,"C_K":182.739-273.15,"base":10,"factor_Pa":100000},
            "78-83-1": {"A":4.34504,"B":1190.380,"C_K":166.670-273.15,"base":10,"factor_Pa":100000},
            "56-81-5": {"A":10.6190,"B":4487.040,"C_K":132.95-273.15,"base":2.718281828459045,"factor_Pa":100000}},
    }
    cesari = json.loads((ROOT / "data/source/activity_fitting/cesari_2017_phase_equilibria.json").read_text())
    source_psat["10.1016/j.fluid.2017.09.008"] = {cas:{"A":p["A_Pa"],"B":p["B"],"C_K":p["C_K"],"base":p["base"],"factor_Pa":1} for cas,p in cesari["vapor_pressure"].items()}
    # Huang 2016 Table 5, ln(P/kPa), kelvin denominators.
    source_psat["10.1016/j.fluid.2015.08.012"] = {
        cas:{"A":a,"B":b,"C_K":c,"base":2.718281828459045,"factor_Pa":1000}
        for cas,a,b,c in (("76-05-1",15.19548,3243.6286,-38.289),
                          ("64-17-5",16.8969,3803.98,-41.68),
                          ("383-63-1",23.9761,10081.33,185.03),
                          ("7732-18-5",16.2886,3816.44,-46.13))}
    # Reddy 2013 Table 8 uses ln(P/kPa)=A+B/(T+C), with negative printed B.
    source_psat["10.1016/j.jct.2012.11.005"] = {
        cas:{"A":a,"B":b,"C_K":c,"base":2.718281828459045,"factor_Pa":1000}
        for cas,a,b,c in (("108-20-3",14.162111,2864.844,-41.5397),
                          ("67-56-1",14.579031,2566.002,-80.0485),
                          ("64-17-5",16.411996,3536.759,-51.5975),
                          ("71-36-3",14.577264,2878.370,-102.5146))}
    # Wang 2020 Table 7: ln(P/kPa)=C1+C2/T+C5*ln(T)+C6*T**C7.
    source_psat["10.1021/acs.jced.0c00204"] = {
        cas:{"type":"dippr101","coefficients":[a,b,c,d,e],"factor_Pa":1000,"Tmin_K":lo,"Tmax_K":hi}
        for cas,a,b,c,d,e,lo,hi in (
            ("108-21-4",42.85,-5563.90,-3.88,2.48e-18,6,199.75,532.),
            ("67-63-0",103.81,-9040.,-12.68,5.54e-6,2,185.26,508.30),
            ("142-96-1",65.32,-7537.60,-7.06,9.14e-18,6,175.30,584.10),
            ("100-66-3",121.15,-9307.70,-16.69,1.49e-2,1,235.65,645.60))}
    unique, endpoints = {}, []
    for row in rows:
        if row["cas1"] > row["cas2"]:
            row["cas1"],row["cas2"] = row["cas2"],row["cas1"]
            row["x1"] = 1-row["x1"]
            if "y1" in row:row["y1"] = 1-row["y1"]
        for key in ("T_K","P_Pa","x1"):
            if key not in row:raise ValueError(f"Incomplete measured state: {row}")
        if not 0 <= row["x1"] <= 1 or row["T_K"] <= 0 or row["P_Pa"] <= 0:
            raise ValueError(f"Invalid measured state: {row}")
        if row["x1"] in (0,1):
            endpoints.append(row)
            continue
        key=(row["doi"],row["cas1"],row["cas2"],round(row["T_K"],7),round(row["P_Pa"],3),round(row["x1"],8))
        if key in unique:
            audit.append({"source":row["doi"],"kind":"duplicate_measured_state","reason":"same source/pair/T/P/x already retained"})
        else:
            unique[key] = row
    result = {"metadata":{"purpose":"Raw binary VLE benchmark; measured values only, no liquid parameter fitting", "aggregation":"equal weight per distinct CAS pair", "source_evidence":evidence,
                          "extractor_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "audit":audit},
              "rows":list(unique.values()), "pure_endpoints":endpoints,
              "source_psat":source_psat, "pure_data_fendu":values.get("PURE_DATA", {})}
    # Fendu's pure measurements are in its own script, not the common-basis one.
    result["pure_data_fendu"] = literal_assignments(ROOT / "scripts/activity_fitting/fit_fendu_2025_activity_parameters.py")["PURE_DATA"]
    with args.output.open("x") as handle:handle.write(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"points":len(result["rows"]),"pairs":len({(r['cas1'],r['cas2']) for r in result['rows']}),"with_measured_y":sum('y1'in r for r in result['rows']),"sources":len({r['doi'] for r in result['rows']}),"audit":dict(Counter(r['kind'] for r in audit))},indent=2))


if __name__ == "__main__":
    main()
