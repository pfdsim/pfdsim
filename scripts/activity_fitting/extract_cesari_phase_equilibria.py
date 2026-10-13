"""Normalize Cesari 2017 Tables 3/6 from NIST ThermoML and SI Table S2.

Input URLs and hashes are retained. No fitted interaction parameters are
modified. The output path must be new. Solid-liquid and excess-enthalpy data
are outside this VLE/LLE benchmark and are not extracted.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile


ORG = {
    1: ("90-05-1", "guaiacol"), 2: ("7732-18-5", "water"),
    3: ("91-10-1", "syringol"), 4: ("95-48-7", "o-cresol"),
    5: ("108-39-4", "m-cresol"), 6: ("106-44-5", "p-cresol"),
    7: ("120-80-9", "pyrocatechol"), 8: ("121-33-5", "vanillin"),
    9: ("108-95-2", "phenol"), 10: ("64-17-5", "ethanol"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thermoml", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.thermoml.read_text())
    vle, endpoints = [], defaultdict(dict)
    for dataset in source["PureOrMixtureData"]:
        prop = dataset["Property"][0]
        group = prop["Property-MethodID"]["PropertyGroup"]
        group = group[next(k for k in group if k != "tml_elements")]
        phase = prop["PropPhaseID"]["ePropPhase"]
        components = [c["RegNum"]["nOrgNum"] for c in dataset["Component"]]
        variables = {v["nVarNumber"]: v["VariableID"] for v in dataset["Variable"]}
        if group["ePropName"] == "Boiling temperature at pressure P, K":
            pc = next(c for c in components if c != 10)
            assert variables[2]["RegNum"]["nOrgNum"] == pc
            assert variables[1]["VariableType"]["ePressure"] == "Pressure, kPa"
            for row in dataset["NumValues"]:
                values = {v["nVarNumber"]: v["nVarValue"] for v in row["VariableValue"]}
                measurement = row["PropertyValue"][0]
                vle.append({"cas_pc": ORG[pc][0], "name_pc": ORG[pc][1], "x_pc": values[2],
                            "P_Pa": values[1]*1000, "T_K": measurement["nPropValue"],
                            "nist_dataset": dataset["nPureOrMixtureDataNumber"],
                            "nist_expanded_uncertainty_T_K": measurement.get("CombinedUncertainty", {}).get("nCombExpandUncertValue")})
        elif phase in ("Liquid mixture 1", "Liquid mixture 2") and group["ePropName"] == "Mole fraction":
            pc = prop["Property-MethodID"]["RegNum"]["nOrgNum"]
            assert 2 in components and pc != 2
            for row in dataset["NumValues"]:
                t = row["VariableValue"][0]["nVarValue"]
                endpoint = "x_pc_aqueous" if phase == "Liquid mixture 1" else "x_pc_organic"
                value = row["PropertyValue"][0]["nPropValue"]
                assert endpoint not in endpoints[pc, t]
                endpoints[pc, t][endpoint] = value
    lle = [{"cas_pc": ORG[pc][0], "name_pc": ORG[pc][1], "T_K": t, **row}
           for (pc, t), row in sorted(endpoints.items())]
    assert len(vle) == 172 and len(lle) == 32
    assert sum("x_pc_organic" in r for r in lle) == 26
    assert sum(r["cas_pc"] == "108-95-2" for r in vle) == 54
    assert sum(r["cas_pc"] == "90-05-1" for r in vle) == 63
    assert sum(r["cas_pc"] == "95-48-7" for r in vle) == 55
    # Cross-check independent PDF Table 3 / Table 6 values.
    assert any(r["cas_pc"] == "90-05-1" and r["T_K"] == 298.15 and r["x_pc_aqueous"] == .0036 and r["x_pc_organic"] == .7522 for r in lle)
    assert any(r["cas_pc"] == "108-95-2" and r["P_Pa"] == 9000 and r["x_pc"] == .1 and r["T_K"] == 303.09 for r in vle)

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(args.supplement) as archive:
        doc = ET.fromstring(archive.read("word/document.xml"))
    tables = []
    for table in doc.findall(".//w:tbl", ns):
        rows = [["".join(t.text or "" for t in c.findall(".//w:t", ns)) for c in r.findall("w:tc", ns)] for r in table.findall("w:tr", ns)]
        if any("Guaiacol" in cell for row in rows for cell in row) and any(row[0] == "A" for row in rows):
            tables.append(rows)
    assert len(tables) == 1
    coefficients = {row[0]: [float(row[i]) for i in (1, 3, 5, 7)] for row in tables[0] if row[0] in ("A", "B", "C")}
    saturation = {}
    for i, (cas, name, basis, units, base) in enumerate((
        ("95-48-7", "o-cresol", "degC", "mmHg", 10.0),
        ("108-95-2", "phenol", "degC", "mmHg", 10.0),
        ("90-05-1", "guaiacol", "K", "bar", 10.0),
        ("64-17-5", "ethanol", "K", "kPa", math.e),
    )):
        a, b, c = (coefficients[key][i] for key in ("A", "B", "C"))
        factor = {"mmHg": 101325/760, "bar": 100000, "kPa": 1000}[units]
        normalized_a = a + math.log(factor)/math.log(base)
        normalized_c = c - 273.15 if basis == "degC" else c
        tb = b/(normalized_a-math.log(101325)/math.log(base))-normalized_c
        assert 300 < tb < 500
        saturation[cas] = {"name": name, "original_A": a, "original_B": b, "original_C": c,
                           "original_pressure_units": units, "temperature_basis": basis,
                           "base": base, "A_Pa": normalized_a, "B": b, "C_K": normalized_c,
                           "normal_boiling_point_K": tb}
    payload = {
        "metadata": {"doi": "10.1016/j.fluid.2017.09.008", "source": "Cesari et al., Fluid Phase Equilibria 453 (2017) 58-66, Tables 3 and 6; supporting information Table S2",
                     "thermoml_url": "https://trc.nist.gov/ThermoML/10.1016/j.fluid.2017.09.008.json",
                     "supplement_url": "https://ars.els-cdn.com/content/image/1-s2.0-S0378381217303394-mmc1.docx",
                     "thermoml_sha256": hashlib.sha256(args.thermoml.read_bytes()).hexdigest(),
                     "supplement_sha256": hashlib.sha256(args.supplement.read_bytes()).hexdigest(),
                     "note": "SI footnotes specify log10 for phenolics and ln for ethanol. The ethanol C-unit cell says degrees C, but the listed ln(kPa) coefficients require kelvin temperature (normal boiling point near 351.5 K); the Celsius interpretation is physically inconsistent. The benchmark records that interpretation explicitly.",
                     "experimental_vapor_compositions": "Not reported; Table 6 provides bubble temperatures, pressure and liquid composition only.",
                     "uncertainties_from_paper": {"VLE_uT_K": .06, "VLE_uP_Pa": 40, "composition_u_upper_bound": .005, "LLE_uT_K": .1}},
        "vapor_pressure": saturation, "vle": vle, "lle": lle,
    }
    with args.output.open("x") as handle:
        handle.write(json.dumps(payload, indent=2)+"\n")
    print(f"Saved {len(vle)} bubble points, {len(lle)} LLE temperature records (26 complete tie-lines), and four SI saturation correlations to {args.output}")


if __name__ == "__main__":
    main()
