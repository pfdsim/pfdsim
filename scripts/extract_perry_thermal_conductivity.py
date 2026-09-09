#!/usr/bin/env python3
"""Extract Perry thermal-conductivity correlations and tabulated values."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.extract_perry_properties import TableSpec, extract_table
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from extract_perry_properties import TableSpec, extract_table


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from property_resolver import PropertyResolver


OUTPUT = ROOT / "data" / "perry_thermal_conductivity.json"

VAPOR_TABLE = TableSpec(
    key="vapor_thermal_conductivity",
    table="2-145",
    pages=(324, 330),
    has_equation_id=True,
    kind="generic_correlation",
    title="Vapor thermal conductivity of inorganic and organic substances",
    units={"thermal_conductivity": "W/(m*K)", "temperature": "K"},
)
LIQUID_TABLE = TableSpec(
    key="liquid_thermal_conductivity",
    table="2-147",
    pages=(333, 339),
    has_equation_id=False,
    kind="generic_correlation",
    title="Thermal conductivity of inorganic and organic liquids",
    units={"thermal_conductivity": "W/(m*K)", "temperature": "K"},
)

# The PDF text layer loses two source characters that are recoverable from the
# printed endpoint values.  The cyclohexanone comma is split into two numbers;
# the other two signs are corrected only when both endpoints validate.
VAPOR_COEFFICIENT_CORRECTIONS = {
    "64-19-7": {
        "match": [3.3901e-6, 1.9588, 36053.0, 14086000.0],
        "replacement": [3.3901e-6, 1.9588, 36053.0, -14086000.0],
        "note": "C4 sign restored from both printed endpoint values",
    },
    "67-64-1": {
        "match": [-26.8, 0.9098, 126500000.0],
        "replacement": [26.8, 0.9098, 126500000.0],
        "note": "C1 sign restored from both printed endpoint values",
    },
    "108-94-1": {
        "match": [-1095.5, -0.023408, 498.0, 780.0, -7835500000.0],
        "replacement": [-1095.5, -0.023408, 498780.0, -7835500000.0],
        "note": "C3 thousands separator restored after PDF token splitting",
    },
}


def _samples(start_C: int, values: list[float]) -> list[list[float]]:
    """Pair a consecutive 10-degree Celsius run with SI temperatures."""
    return [
        [temperature + 273.15, value]
        for temperature, value in zip(
            range(start_C, start_C + 10 * len(values), 10), values, strict=True
        )
    ]


# Table 2-146 has no identities or correlations; preserve its displayed knots.
# Separate runs prevent interpolation across blank cells (notably propanol).
SATURATED_LIQUID_TABLE: dict[str, dict[str, Any]] = {
    "acetaldehyde": {
        "runs": [
            _samples(-50, [0.211, 0.206, 0.200, 0.195, 0.189, 0.184, 0.182, 0.180])
        ]
    },
    "acetic acid": {
        "runs": [_samples(20, [0.173, 0.170, 0.168, 0.167, 0.165, 0.163, 0.161])]
    },
    "aniline": {
        "runs": [
            _samples(
                0,
                [
                    0.186,
                    0.184,
                    0.182,
                    0.180,
                    0.177,
                    0.174,
                    0.171,
                    0.169,
                    0.168,
                    0.167,
                    0.167,
                ],
            )
        ]
    },
    "butanol": {
        "aliases": ["1-butanol"],
        "runs": [
            _samples(
                -50,
                [
                    0.175,
                    0.174,
                    0.173,
                    0.172,
                    0.171,
                    0.170,
                    0.168,
                    0.167,
                    0.166,
                    0.165,
                    0.164,
                    0.163,
                    0.162,
                    0.161,
                    0.160,
                    0.159,
                ],
            )
        ],
    },
    "carbon disulfide": {
        "runs": [
            _samples(
                -50,
                [
                    0.194,
                    0.190,
                    0.186,
                    0.182,
                    0.178,
                    0.174,
                    0.170,
                    0.166,
                    0.161,
                    0.158,
                    0.156,
                    0.154,
                    0.152,
                    0.150,
                ],
            )
        ]
    },
    "cyclohexane": {
        "runs": [_samples(10, [0.122, 0.120, 0.119, 0.118, 0.117, 0.116, 0.114, 0.112])]
    },
    "ethanol": {
        "runs": [
            _samples(
                -50,
                [
                    0.188,
                    0.186,
                    0.184,
                    0.181,
                    0.179,
                    0.177,
                    0.175,
                    0.173,
                    0.171,
                    0.168,
                    0.165,
                    0.162,
                    0.159,
                    0.156,
                    0.153,
                    0.151,
                ],
            )
        ]
    },
    "ethyl acetate": {
        "runs": [
            _samples(
                20, [0.145, 0.142, 0.139, 0.136, 0.133, 0.130, 0.127, 0.123, 0.119]
            )
        ]
    },
    "ethylamine": {"runs": [_samples(-50, [0.204, 0.201, 0.199, 0.196, 0.194, 0.191])]},
    "ethyl ether": {
        "aliases": ["diethyl ether"],
        "runs": [
            _samples(
                -50,
                [
                    0.159,
                    0.155,
                    0.151,
                    0.147,
                    0.144,
                    0.140,
                    0.139,
                    0.134,
                    0.129,
                    0.125,
                    0.120,
                    0.116,
                    0.112,
                ],
            )
        ],
    },
    "ethyl iodide": {
        "aliases": ["iodoethane"],
        "runs": [_samples(0, [0.092, 0.090, 0.088, 0.086, 0.085, 0.083, 0.081, 0.080])],
    },
    "ethylene glycol": {
        "runs": [_samples(0, [0.254, 0.255, 0.256, 0.258, 0.259, 0.260])]
    },
    "formic acid": {
        "runs": [
            _samples(
                0,
                [
                    0.265,
                    0.261,
                    0.257,
                    0.257,
                    0.253,
                    0.250,
                    0.246,
                    0.243,
                    0.240,
                    0.236,
                    0.232,
                ],
            )
        ]
    },
    "gasoline": {
        "runs": [
            _samples(
                -50,
                [
                    0.131,
                    0.128,
                    0.125,
                    0.123,
                    0.121,
                    0.120,
                    0.118,
                    0.116,
                    0.114,
                    0.112,
                    0.110,
                    0.108,
                    0.106,
                    0.104,
                    0.102,
                    0.100,
                ],
            )
        ]
    },
    "glycerol": {
        "aliases": ["glycerine"],
        "runs": [
            _samples(
                20, [0.284, 0.285, 0.287, 0.288, 0.289, 0.291, 0.293, 0.294, 0.295]
            )
        ],
    },
    "kerosene": {
        "runs": [_samples(0, [0.140, 0.139, 0.139, 0.138, 0.138, 0.137, 0.137])]
    },
    "methanol": {
        "runs": [
            _samples(
                -50,
                [
                    0.225,
                    0.222,
                    0.219,
                    0.216,
                    0.212,
                    0.209,
                    0.206,
                    0.203,
                    0.199,
                    0.195,
                    0.192,
                    0.189,
                    0.187,
                    0.184,
                    0.182,
                    0.180,
                ],
            )
        ]
    },
    "methyl formate": {
        "runs": [
            _samples(
                -50, [0.217, 0.213, 0.209, 0.205, 0.200, 0.195, 0.191, 0.186, 0.180]
            )
        ]
    },
    "oil, castor": {
        "aliases": ["castor oil"],
        "runs": [
            _samples(
                10,
                [0.182, 0.181, 0.180, 0.179, 0.178, 0.177, 0.176, 0.175, 0.174, 0.170],
            )
        ],
    },
    "oil, olive": {
        "aliases": ["olive oil"],
        "runs": [
            _samples(
                10,
                [0.170, 0.169, 0.168, 0.167, 0.166, 0.166, 0.165, 0.165, 0.164, 0.164],
            )
        ],
    },
    "pentane": {
        "aliases": ["n-pentane"],
        "runs": [
            _samples(
                -50,
                [
                    0.142,
                    0.139,
                    0.136,
                    0.132,
                    0.128,
                    0.125,
                    0.122,
                    0.119,
                    0.115,
                    0.112,
                    0.108,
                    0.105,
                    0.101,
                    0.098,
                    0.095,
                    0.091,
                ],
            )
        ],
    },
    "propanol": {
        "aliases": ["1-propanol"],
        "runs": [
            _samples(-50, [0.167, 0.166, 0.165]),
            _samples(30, [0.171, 0.169, 0.168, 0.167, 0.165, 0.164, 0.163, 0.162]),
        ],
    },
    "sulfuric acid": {"runs": [_samples(0, [0.314])]},
    "toluene": {
        "runs": [
            _samples(
                -50,
                [
                    0.152,
                    0.149,
                    0.147,
                    0.144,
                    0.142,
                    0.139,
                    0.137,
                    0.134,
                    0.132,
                    0.129,
                    0.126,
                    0.124,
                    0.122,
                    0.119,
                    0.117,
                    0.114,
                ],
            )
        ]
    },
    "turpentine": {"runs": [_samples(0, [0.130, 0.129, 0.128, 0.127, 0.126, 0.125])]},
}


def _apply_and_validate_corrections(chemicals: dict[str, dict[str, Any]]) -> None:
    for cas, correction in VAPOR_COEFFICIENT_CORRECTIONS.items():
        for row in chemicals[cas][VAPOR_TABLE.key]:
            if row["coefficients"] == correction["match"]:
                row["printed_coefficients"] = row["coefficients"]
                row["coefficients"] = correction["replacement"]
                row["extraction_correction"] = correction["note"]

    resolver = PropertyResolver()
    for entry in chemicals.values():
        for row in entry.get(LIQUID_TABLE.key, []):
            row["equation_id"] = 100
        for key in (VAPOR_TABLE.key, LIQUID_TABLE.key):
            for row in entry.get(key, []):
                errors = []
                correlation = resolver._normalized_perry_conductivity_correlation(row)
                for suffix in ("min", "max"):
                    expected = row[f"value_at_T_{suffix}"]
                    evaluated = resolver._evaluate_correlation(
                        correlation,
                        row[f"T_{suffix}_K"],
                    )
                    if evaluated is None:
                        raise ValueError(f"Cannot evaluate endpoint for {entry['cas']}")
                    actual, _ = evaluated
                    errors.append(abs(actual - expected) / expected)
                row["maximum_endpoint_relative_error"] = max(errors)
                if not math.isfinite(row["maximum_endpoint_relative_error"]):
                    raise ValueError(
                        f"Nonfinite endpoint validation for {entry['cas']}"
                    )


def main() -> None:
    chemicals: dict[str, dict[str, Any]] = {}
    counts = {
        VAPOR_TABLE.key: extract_table(VAPOR_TABLE, chemicals),
        LIQUID_TABLE.key: extract_table(LIQUID_TABLE, chemicals),
    }
    _apply_and_validate_corrections(chemicals)
    maximum_error = max(
        row["maximum_endpoint_relative_error"]
        for entry in chemicals.values()
        for key in (VAPOR_TABLE.key, LIQUID_TABLE.key)
        for row in entry.get(key, [])
    )
    if maximum_error > 0.01:
        raise ValueError(f"Perry endpoint validation exceeded 1%: {maximum_error:.3%}")

    payload = {
        "metadata": {
            "source_book": "Perry's Chemical Engineers' Handbook, 9th ed.",
            "source_pdf_pages": [324, 339],
            "tables": {
                VAPOR_TABLE.key: {
                    "table": VAPOR_TABLE.table,
                    "pages": list(VAPOR_TABLE.pages),
                    "rows_extracted": counts[VAPOR_TABLE.key],
                    "equations": {
                        "100": "k = C1 + C2*T + C3*T**2 + C4*T**3 + C5*T**4",
                        "102": "k = C1*T**C2/(1 + C3/T + C4/T**2)",
                    },
                },
                "saturated_liquid_thermal_conductivity": {
                    "table": "2-146",
                    "pages": [331, 332],
                    "substances_extracted": len(SATURATED_LIQUID_TABLE),
                    "interpolation": "linear only within consecutive 10 K runs",
                },
                LIQUID_TABLE.key: {
                    "table": LIQUID_TABLE.table,
                    "pages": list(LIQUID_TABLE.pages),
                    "rows_extracted": counts[LIQUID_TABLE.key],
                    "equations": {
                        "100": "k = C1 + C2*T + C3*T**2 + C4*T**3 + C5*T**4",
                    },
                },
            },
            "units": {"thermal_conductivity": "W/(m*K)", "temperature": "K"},
            "maximum_endpoint_relative_error": maximum_error,
        },
        "chemicals": dict(sorted(chemicals.items())),
        "saturated_liquids": SATURATED_LIQUID_TABLE,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    print(json.dumps(counts, indent=2, sort_keys=True))
    print(f"Unique CAS entries: {len(chemicals)}")
    print(f"Table 2-146 substances: {len(SATURATED_LIQUID_TABLE)}")
    print(f"Maximum endpoint relative error: {maximum_error:.6%}")


if __name__ == "__main__":
    main()
