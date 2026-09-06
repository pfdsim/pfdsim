#!/usr/bin/env python3
"""Compare Nagata extended UNIQUAC with the stored tau fit for ethanol/ethyl acetate."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from thermodynamics import create_thermodynamics


INTERACTION_PATH = ROOT / "data" / "uniquac_binary_interactions_cas.json"
CAS_TO_COMPONENT = {
    "64-17-5": "ethanol",
    "141-78-6": "ethyl acetate",
}
COMPONENTS = list(CAS_TO_COMPONENT.values())


def load_overrides() -> tuple[dict, dict]:
    with INTERACTION_PATH.open(encoding="utf-8") as handle:
        records = json.load(handle)["interactions"]

    pair = set(CAS_TO_COMPONENT)
    matching = [
        record
        for record in records
        if {record["cas1"], record["cas2"]} == pair
    ]
    reference = next(
        record
        for record in matching
        if not record.get("disabled") and "tau12_a" in record
    )
    modified = next(
        record
        for record in matching
        if record.get("source", "").startswith("Nagata and Gmehling")
    )

    def runtime_override(record: dict) -> dict:
        return {
            **record,
            "model": "UNIQUAC",
            "component1": CAS_TO_COMPONENT[record["cas1"]],
            "component2": CAS_TO_COMPONENT[record["cas2"]],
        }

    return runtime_override(reference), runtime_override(modified)


def vapor_ethanol(thermo, temperature: float, pressure_bar: float, liquid: dict) -> float:
    k_values = thermo.K_values(temperature, pressure_bar, liquid)
    total = sum(liquid[component] * k_values[component] for component in COMPONENTS)
    return liquid["ethanol"] * k_values["ethanol"] / total


def compare_at_pressure(
    reference,
    modified,
    pressure_bar: float,
    points: int,
) -> tuple[float, float]:
    temperature_errors = []
    vapor_errors = []
    for index in range(1, points + 1):
        x_ethanol = index / (points + 1)
        liquid = {
            "ethanol": x_ethanol,
            "ethyl acetate": 1.0 - x_ethanol,
        }
        reference_temperature = reference.bubble_point_T(liquid, pressure_bar)
        modified_temperature = modified.bubble_point_T(
            liquid,
            pressure_bar,
            reference_temperature,
        )
        reference_y = vapor_ethanol(
            reference, reference_temperature, pressure_bar, liquid
        )
        modified_y = vapor_ethanol(
            modified, modified_temperature, pressure_bar, liquid
        )
        temperature_errors.append(modified_temperature - reference_temperature)
        vapor_errors.append(modified_y - reference_y)

    rms_temperature = math.sqrt(
        sum(error * error for error in temperature_errors) / points
    )
    rms_vapor = math.sqrt(sum(error * error for error in vapor_errors) / points)
    return rms_temperature, rms_vapor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pressures-bar",
        nargs="+",
        type=float,
        default=[0.5, 1.01325],
    )
    parser.add_argument("--points", type=int, default=99)
    args = parser.parse_args()
    if args.points < 1:
        parser.error("--points must be at least 1")
    if any(pressure <= 0.0 for pressure in args.pressures_bar):
        parser.error("--pressures-bar values must be positive")

    reference_override, modified_override = load_overrides()
    reference = create_thermodynamics(
        COMPONENTS,
        "UNIQUAC",
        interaction_overrides=[reference_override],
    )
    modified = create_thermodynamics(
        COMPONENTS,
        "UNIQUAC",
        interaction_overrides=[modified_override],
    )

    print(f"composition grid: x_ethanol = i/{args.points + 1}, i=1..{args.points}")
    print("reference: stored active tau-based UNIQUAC record")
    print("candidate: disabled Nagata-Gmehling extended UNIQUAC record")
    print("pressure_bar\tRMS_T_K\tRMS_y_ethanol\tRMS_y_percentage_points")
    for pressure in args.pressures_bar:
        rms_temperature, rms_vapor = compare_at_pressure(
            reference,
            modified,
            pressure,
            args.points,
        )
        print(
            f"{pressure:.6g}\t{rms_temperature:.9g}\t"
            f"{rms_vapor:.9g}\t{100.0 * rms_vapor:.9g}"
        )


if __name__ == "__main__":
    main()
