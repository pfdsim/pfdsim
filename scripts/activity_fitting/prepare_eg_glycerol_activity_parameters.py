#!/usr/bin/env python3
"""Prepare the reviewed ethylene-glycol/glycerol activity records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/source/activity_fitting/eg_glycerol_nrtl_uniquac_curated.json"

SAFE_PAIRS = {
    ("64-17-5", "56-81-5"),
    ("7732-18-5", "56-81-5"),
    ("71-36-3", "56-81-5"),
    ("78-83-1", "56-81-5"),
    ("64-17-5", "107-21-1"),
    ("7732-18-5", "107-21-1"),
    ("71-23-8", "107-21-1"),
}

RANGES = {
    ("64-17-5", "56-81-5"): (273.15, 353.15),
    ("7732-18-5", "56-81-5"): (273.15, 363.15),
    ("71-36-3", "56-81-5"): (333.15, 343.15),
    ("78-83-1", "56-81-5"): (313.15, 323.15),
    ("64-17-5", "107-21-1"): (354.0, 389.79),
    ("7732-18-5", "107-21-1"): (373.15, 470.27),
    ("71-23-8", "107-21-1"): (370.40, 470.27),
}

EXTRAPOLATION = {
    ("64-17-5", "56-81-5"): "inverse_square_cubic",
    ("7732-18-5", "56-81-5"): "unrestricted",
    ("71-36-3", "56-81-5"): "unrestricted",
    ("78-83-1", "56-81-5"): "unrestricted",
    ("64-17-5", "107-21-1"): "unrestricted",
    ("7732-18-5", "107-21-1"): "unrestricted",
    ("71-23-8", "107-21-1"): "unrestricted",
}

AUDIT_METRICS = {
    ("64-17-5", "56-81-5"): {
        "raw_points": 84,
        "NRTL_pressure_AARD_percent": 3.011790284809,
        "UNIQUAC_pressure_AARD_percent": 2.993706592493,
        "UNIFNIST_pressure_AARD_percent": 5.867982265170,
    },
    ("7732-18-5", "56-81-5"): {
        "raw_points": 94,
        "NRTL_pressure_AARD_percent": 2.531285565290,
        "UNIQUAC_pressure_AARD_percent": 4.034264718575,
        "UNIFNIST_pressure_AARD_percent": 5.229766142537,
    },
    ("71-36-3", "56-81-5"): {
        "raw_points": 30,
        "NRTL_pressure_AARD_percent": 1.625803827129,
        "UNIQUAC_pressure_AARD_percent": 2.231718641728,
        "UNIFNIST_pressure_AARD_percent": 5.632605556515,
    },
    ("78-83-1", "56-81-5"): {
        "raw_points": 30,
        "NRTL_pressure_AARD_percent": 1.612816638388,
        "UNIQUAC_pressure_AARD_percent": 3.004856301673,
        "UNIFNIST_pressure_AARD_percent": 8.837740808161,
    },
    ("64-17-5", "107-21-1"): {
        "raw_points": 15,
        "NRTL_temperature_MAE_K": 0.123973761585,
        "NRTL_y1_MAE": 0.001047893777,
        "UNIQUAC_temperature_MAE_K": 0.125479319751,
        "UNIQUAC_y1_MAE": 0.001049619823,
    },
    ("7732-18-5", "107-21-1"): {
        "raw_points": 19,
        "NRTL_temperature_MAE_K": 0.146512737339,
        "NRTL_y1_MAE": 0.003598537555,
        "UNIQUAC_temperature_MAE_K": 0.464042786350,
        "UNIQUAC_y1_MAE": 0.002619819074,
    },
    ("71-23-8", "107-21-1"): {
        "raw_points": 21,
        "NRTL_temperature_MAE_K": 0.606092664868,
        "NRTL_y1_MAE": 0.006770158495,
        "UNIQUAC_temperature_MAE_K": 0.366533007655,
        "UNIQUAC_y1_MAE": 0.004323409830,
    },
}

UNIQUAC_COMPONENTS = [
    {
        "cas": "107-21-1",
        "name": "ethylene glycol",
        "formula": "C2H6O2",
        "r": 2.4087,
        "q": 2.248,
        "source": "Pla-Franco et al. 2015 and Zhong et al. 2014; Aspen/DECHEMA basis",
    },
    {
        "cas": "56-81-5",
        "name": "glycerol",
        "formula": "C3H8O3",
        "r": 4.7957,
        "q": 4.908,
        "source": "Zaoui et al. 2014; Wiguno et al. 2016; Mustain et al. 2022",
    },
    {
        "cas": "71-36-3",
        "name": "1-butanol",
        "formula": "C4H10O",
        "r": 3.4543,
        "q": 3.052,
        "source": "Matsuda et al. 2025 standard basis; Mustain et al. 2022 source basis",
    },
    {
        "cas": "78-83-1",
        "name": "isobutanol",
        "formula": "C4H10O",
        "r": 3.4535,
        "q": 3.048,
        "source": "existing PFDSim standard basis; Mustain et al. 2022 source basis",
    },
]


def _runtime_record(pair: dict, model: str) -> dict:
    first = pair["component_1"]
    second = pair["component_2"]
    key = (first["cas"], second["cas"])
    selected = pair[model]
    record = {
        "model": model,
        "cas1": key[0],
        "cas2": key[1],
        "component1": first["name"],
        "component2": second["name"],
        "Tmin_K": RANGES[key][0],
        "Tmax_K": RANGES[key][1],
        "extrapolation": EXTRAPOLATION[key],
        "source": pair["source"]["authors"] + f" ({pair['source']['year']})",
        "source_doi": pair["source"].get("doi"),
        "source_file": "data/source/activity_fitting/eg_glycerol_nrtl_uniquac_curated.json",
        "fit_status": "recommended_source_reproduced",
        "comment": pair["screening_note"],
        "fit_evidence": {
            **pair["fit_details"],
            **AUDIT_METRICS[key],
            "confidence": pair["confidence"],
            "selected_fit": pair["selected_fit"],
        },
    }
    if record["source_doi"] is None:
        record.pop("source_doi")
    if model == "NRTL":
        record.update(
            alpha12=selected["alpha"],
            tau12_c=selected["12"]["c"],
            tau12_d=selected["12"]["d"],
            tau21_c=selected["21"]["c"],
            tau21_d=selected["21"]["d"],
        )
    else:
        record.update(
            model_variant="standard_uniquac",
            use_q_prime=False,
            tau12_a=selected["12"]["a"],
            tau12_b=selected["12"]["b"],
            tau21_a=selected["21"]["a"],
            tau21_b=selected["21"]["b"],
        )
    return record


def build_payload() -> dict:
    source = json.loads(SOURCE.read_text())
    selected_pairs = [
        pair
        for pair in source["pairs"]
        if (pair["component_1"]["cas"], pair["component_2"]["cas"]) in SAFE_PAIRS
    ]
    if len(selected_pairs) != len(SAFE_PAIRS):
        raise ValueError("The curated source does not contain every selected safe pair")
    interactions = [
        _runtime_record(pair, model)
        for pair in selected_pairs
        for model in ("NRTL", "UNIQUAC")
    ]
    return {
        "metadata": {
            "description": "Reviewed EG/glycerol NRTL and UNIQUAC records",
            "source_file": "data/source/activity_fitting/eg_glycerol_nrtl_uniquac_curated.json",
            "selected_pair_count": len(selected_pairs),
            "selected_record_count": len(interactions),
            "selection_rule": "Direct binary fits with reproduced source metrics and explicit UNIQUAC structural bases",
            "excluded_pairs": [
                "methanol + glycerol: irreconcilable glycerol-rich source data",
                "1-propanol + glycerol: no transferable joint temperature correlation",
                "2-propanol + glycerol: direct Wibawa/Soujanya pressure conflict",
            ],
            "audit_scripts": [
                "scripts/activity_fitting/fit_methanol_glycerol_activity.py",
                "scripts/activity_fitting/fit_propanol_glycerol_activity.py",
            ],
            "reference_files": [
                "data/reference/vapor-liquid-equilibria/2012-kamihama-ethanol-water-ethylene-glycol-thermoml.json",
                "data/reference/vapor-liquid-equilibria/2014-zaoui-ethanol-water-glycerol-thermoml.json",
                "data/reference/vapor-liquid-equilibria/2015-pla-franco-propanol-water-ethylene-glycol-vle.pdf",
                "data/reference/vapor-liquid-equilibria/2015-wibawa-ethanol-isopropanol-glycerol-vle.pdf",
                "data/reference/vapor-liquid-equilibria/2016-soujanya-methanol-isopropanol-water-glycerol-vle.pdf",
                "data/reference/vapor-liquid-equilibria/2016-zaoui-glycerol-binary-vle-modeling-thesis.pdf",
                "data/reference/vapor-liquid-equilibria/2020-batutah-ethanol-propanol-glycerol-vle.pdf",
                "data/reference/vapor-liquid-equilibria/2022-mustain-butanol-isobutanol-glycerol-water-vle.pdf",
                "data/reference/vapor-liquid-equilibria/2022-mustain-butanol-isobutanol-glycerol-water-vle-supporting-information.pdf",
            ],
        },
        "uniquac_components": UNIQUAC_COMPONENTS,
        "interactions": interactions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(build_payload(), indent=2, sort_keys=True) + "\n"
    if args.write is None:
        print(encoded, end="")
    else:
        args.write.write_text(encoded)
        print(f"Wrote {args.write}")


if __name__ == "__main__":
    main()
