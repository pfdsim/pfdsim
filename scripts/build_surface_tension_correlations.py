#!/usr/bin/env python3
"""Build CAS-keyed pure-component surface-tension correlations from chemicals."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import chemicals
import chemicals.interface as interface


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = DATA / "surface_tension_correlations_cas.json"


METHOD_METADATA: dict[str, dict[str, Any]] = {
    "mulero_cachadina_refprop": {
        "source_table": "chemicals.interface.sigma_data_Mulero_Cachadina",
        "source_file": "MuleroCachadinaParameters.tsv",
        "equation": "REFPROP_sigma",
        "equation_note": (
            "sigma = sigma0*tau^n0 + sigma1*tau^n1 + sigma2*tau^n2; "
            "tau = 1 - T/Tc"
        ),
        "sigma_units": "N/m",
        "coefficient_units": {
            "Tc": "K",
            "sigma0": "N/m",
            "n0": "dimensionless",
            "sigma1": "N/m",
            "n1": "dimensionless",
            "sigma2": "N/m",
            "n2": "dimensionless",
        },
        "reference": (
            "Mulero, Cachadiña, and Parra, J. Phys. Chem. Ref. Data 41, "
            "043105 (2012), via chemicals."
        ),
    },
    "jasper_lange": {
        "source_table": "chemicals.interface.sigma_data_Jasper_Lange",
        "source_file": "Jasper-Lange.tsv",
        "equation": "Jasper",
        "equation_note": (
            "sigma = max((a - b*(T - 273.15))*1e-3, 0); "
            "T is in K and the linear fit is in degrees Celsius internally"
        ),
        "sigma_units": "N/m",
        "coefficient_units": {
            "a": "mN/m",
            "b": "mN/(m*C)",
        },
        "reference": (
            "Jasper, J. Phys. Chem. Ref. Data 1, 841-1010 (1972), "
            "reprinted in Lange's Handbook of Chemistry, via chemicals."
        ),
    },
    "somayajulu": {
        "source_table": "chemicals.interface.sigma_data_Somayajulu",
        "source_file": "Somayajulu.tsv",
        "equation": "Somayajulu",
        "equation_note": (
            "sigma = X^(5/4)*(A + B*X + C*X^2)*1e-3; X = (Tc - T)/Tc"
        ),
        "sigma_units": "N/m",
        "coefficient_units": {
            "Tt": "K",
            "Tc": "K",
            "A": "mN/m",
            "B": "mN/m",
            "C": "mN/m",
        },
        "reference": (
            "Somayajulu, Int. J. Thermophys. 9, 559-566 (1988), via chemicals."
        ),
    },
    "somayajulu_revised": {
        "source_table": "chemicals.interface.sigma_data_Somayajulu2",
        "source_file": "SomayajuluRevised.tsv",
        "equation": "Somayajulu",
        "equation_note": (
            "sigma = X^(5/4)*(A + B*X + C*X^2)*1e-3; X = (Tc - T)/Tc"
        ),
        "sigma_units": "N/m",
        "coefficient_units": {
            "Tt": "K",
            "Tc": "K",
            "A": "mN/m",
            "B": "mN/m",
            "C": "mN/m",
        },
        "reference": (
            "Mulero, Parra, and Cachadina, Fluid Phase Equilibria 339, "
            "81-88 (2013), via chemicals."
        ),
    },
    "vdi_ppds_11_dippr_eq106": {
        "source_table": "chemicals.interface.sigma_data_VDI_PPDS_11",
        "source_file": "VDI PPDS surface tensions.tsv",
        "equation": "DIPPR_EQ106",
        "equation_note": (
            "sigma = A*(1 - T/Tc)^(B + C*Tr + D*Tr^2 + E*Tr^3); Tr = T/Tc"
        ),
        "sigma_units": "N/m",
        "coefficient_units": {
            "Tm": "K",
            "Tc": "K",
            "A": "N/m",
            "B": "dimensionless",
            "C": "dimensionless",
            "D": "dimensionless",
            "E": "dimensionless",
        },
        "reference": "VDI Heat Atlas, 2nd edition, via chemicals.",
    },
}


TABLES: tuple[tuple[str, str, str, dict[str, str]], ...] = (
    (
        "mulero_cachadina_refprop",
        "sigma_data_Mulero_Cachadina",
        "Fluid",
        {
            "sigma0": "sigma0",
            "n0": "n0",
            "sigma1": "sigma1",
            "n1": "n1",
            "sigma2": "sigma2",
            "n2": "n2",
            "Tc": "Tc",
            "Tmin": "Tmin",
            "Tmax": "Tmax",
        },
    ),
    (
        "jasper_lange",
        "sigma_data_Jasper_Lange",
        "Name",
        {
            "a": "a",
            "b": "b",
            "Tmin": "Tmin",
            "Tmax": "Tmax",
        },
    ),
    (
        "somayajulu",
        "sigma_data_Somayajulu",
        "Chemical",
        {
            "Tt": "Tt",
            "Tc": "Tc",
            "A": "A",
            "B": "B",
            "C": "C",
        },
    ),
    (
        "somayajulu_revised",
        "sigma_data_Somayajulu2",
        "Chemical",
        {
            "Tt": "Tt",
            "Tc": "Tc",
            "A": "A",
            "B": "B",
            "C": "C",
        },
    ),
    (
        "vdi_ppds_11_dippr_eq106",
        "sigma_data_VDI_PPDS_11",
        "Chemical",
        {
            "Tm": "Tm",
            "Tc": "Tc",
            "A": "A",
            "B": "B",
            "C": "C",
            "D": "D",
            "E": "E",
        },
    ),
)


def finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Expected finite float, got {value!r}")
    return result


def nullable_float(value: Any) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def clean_name(value: Any) -> str:
    return " ".join(str(value or "").split())


def build_payload() -> dict[str, Any]:
    interface.load_interface_dfs()

    by_cas: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "name": None,
            "correlations": {},
        }
    )
    source_counts: dict[str, int] = {}

    for method, dataframe_name, name_column, coefficient_columns in TABLES:
        dataframe = getattr(interface, dataframe_name)
        source_counts[method] = len(dataframe)
        for cas, row in dataframe.iterrows():
            cas = str(cas).strip()
            if not cas:
                continue
            record = by_cas[cas]
            name = clean_name(row.get(name_column))
            if name and not record["name"]:
                record["name"] = name

            coefficients = {}
            t_min = None
            t_max = None
            for output_name, source_name in coefficient_columns.items():
                if output_name == "Tmin":
                    t_min = nullable_float(row[source_name])
                elif output_name == "Tmax":
                    t_max = nullable_float(row[source_name])
                else:
                    coefficients[output_name] = finite_float(row[source_name])
            correlation: dict[str, Any] = {
                "equation": METHOD_METADATA[method]["equation"],
                "source_table": METHOD_METADATA[method]["source_table"],
                "coefficients": coefficients,
            }
            if t_min is not None:
                correlation["Tmin_K"] = t_min
            if t_max is not None:
                correlation["Tmax_K"] = t_max

            record["correlations"][method] = correlation

    chemicals_payload = {
        cas: record
        for cas, record in sorted(by_cas.items())
    }
    total_correlations = sum(
        len(record["correlations"]) for record in chemicals_payload.values()
    )
    return {
        "metadata": {
            "schema": "pfdsim_surface_tension_correlations_cas_v1",
            "description": (
                "Pure-component liquid-vapor/air surface-tension correlations "
                "extracted from the chemicals Python package."
            ),
            "source_package": "chemicals",
            "source_package_version": chemicals.__version__,
            "source_counts": source_counts,
            "compound_count": len(chemicals_payload),
            "correlation_count": total_correlations,
            "temperature_units": "K",
            "default_output_units": "N/m",
            "methods": METHOD_METADATA,
        },
        "chemicals": chemicals_payload,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help=f"output JSON path (default: {OUTPUT})",
    )
    args = parser.parse_args()

    payload = build_payload()
    write_json(args.output, payload)
    metadata = payload["metadata"]
    print(
        f"Wrote {metadata['compound_count']} compounds and "
        f"{metadata['correlation_count']} correlations to {args.output}"
    )
    print("Counts by method:")
    for method, count in sorted(metadata["source_counts"].items()):
        print(f"  {method}: {count}")


if __name__ == "__main__":
    main()
