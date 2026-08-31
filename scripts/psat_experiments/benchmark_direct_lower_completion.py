import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

from scipy.integrate import quad
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from benchmark_lower_clapeyron import (
    correlation_hvap_kJ_mol,
    hvap_row_at_tb,
)
from benchmark_perry_aw import (
    load_curves,
    percentile,
    perry_dlnp_dt,
    perry_ln_p,
)
from perry_properties import PerryPropertyLibrary
from property_resolution.common import R
from property_resolution.coolprop import (
    CoolPropFluidReference,
    coolprop_alias_map,
    coolprop_module,
    coolprop_saturation_pressure,
    coolprop_saturation_temperature,
)
from property_resolution.vapor_pressure_adapter import (
    _ambrose_walton_dln_pressure_dT,
    _ambrose_walton_ln_pressure,
    _ambrose_walton_omega_from_pressure,
    _ambrose_walton_omega_sensitivity,
)


ANCHOR_PRESSURES_BAR = (0.25, 0.10, 0.05)
TARGET_PRESSURES_BAR = (0.04, 0.02, 0.01, 0.005, 0.002, 0.001)
OUTPUT_PATH = Path("/tmp/direct_lower_completion_benchmark.csv")


@dataclass(frozen=True)
class DirectCompletionCase:
    dataset: str
    cas: str
    name: str
    Tc: float
    Pc_bar: float
    preferred_omega: float | None
    T_min: float
    T_max: float
    ln_pressure: object
    dln_pressure_dT: object
    temperature_at_pressure: object
    hvap_J_mol: object
    delta_z: object | None = None


def dynamic_omega_model(case, anchor_temperature, anchor_ln_pressure):
    endpoint_omega = _ambrose_walton_omega_from_pressure(
        anchor_temperature,
        anchor_ln_pressure,
        case.Tc,
        case.Pc_bar,
        preferred_omega=case.preferred_omega,
    )
    sensitivity = _ambrose_walton_omega_sensitivity(
        anchor_temperature,
        case.Tc,
        endpoint_omega,
    )
    if abs(sensitivity) <= 1.0e-10:
        return None
    omega_slope = (
        case.dln_pressure_dT(anchor_temperature)
        - _ambrose_walton_dln_pressure_dT(
            anchor_temperature,
            case.Tc,
            endpoint_omega,
        )
    ) / sensitivity

    def ln_pressure(temperature):
        omega = endpoint_omega + omega_slope * (
            temperature - anchor_temperature
        )
        return _ambrose_walton_ln_pressure(
            temperature,
            case.Tc,
            case.Pc_bar,
            omega,
        )

    return ln_pressure, endpoint_omega, omega_slope


def integrate_clapeyron(
    anchor_temperature,
    target_temperature,
    integrand,
):
    value, _error = quad(
        integrand,
        anchor_temperature,
        target_temperature,
        epsabs=1.0e-9,
        epsrel=1.0e-9,
        limit=100,
    )
    return value


def relaxed_delta_z_predictions(
    case,
    anchor_temperature,
    anchor_pressure,
    anchor_delta_z,
    targets,
    pressure_power,
):
    if (
        not targets
        or not math.isfinite(anchor_delta_z)
        or anchor_delta_z <= 0.0
    ):
        return {}
    minimum_temperature = min(
        temperature for _pressure, temperature in targets
    )

    def derivative(temperature, state):
        pressure_ratio = min(
            1.0,
            max(0.0, math.exp(state[0]) / anchor_pressure),
        )
        effective_delta_z = (
            1.0
            + (anchor_delta_z - 1.0)
            * pressure_ratio**pressure_power
        )
        if effective_delta_z <= 0.0:
            raise ValueError("invalid relaxed delta Z")
        return [
            case.hvap_J_mol(temperature)
            / (R * temperature**2 * effective_delta_z)
        ]

    solution = solve_ivp(
        derivative,
        (anchor_temperature, minimum_temperature),
        [math.log(anchor_pressure)],
        rtol=2.0e-9,
        atol=2.0e-11,
        dense_output=True,
        max_step=max(
            (anchor_temperature - minimum_temperature) / 50.0,
            0.1,
        ),
    )
    if not solution.success:
        return {}
    return {
        pressure: float(solution.sol(temperature)[0])
        for pressure, temperature in targets
    }


def evaluate_case(case):
    rows = []
    unavailable = []
    for anchor_pressure in ANCHOR_PRESSURES_BAR:
        try:
            anchor_temperature = case.temperature_at_pressure(anchor_pressure)
        except (ArithmeticError, ValueError):
            anchor_temperature = None
        if (
            anchor_temperature is None
            or not case.T_min <= anchor_temperature <= case.T_max
        ):
            unavailable.append((anchor_pressure, "anchor unavailable"))
            continue
        anchor_ln_pressure = math.log(anchor_pressure)
        try:
            anchor_round_trip = case.ln_pressure(anchor_temperature)
            dynamic = dynamic_omega_model(
                case,
                anchor_temperature,
                anchor_ln_pressure,
            )
            anchor_hvap = case.hvap_J_mol(anchor_temperature)
            anchor_slope = case.dln_pressure_dT(anchor_temperature)
        except (ArithmeticError, ValueError):
            unavailable.append((anchor_pressure, "anchor model unavailable"))
            continue
        if abs(math.exp(anchor_round_trip) / anchor_pressure - 1.0) > 1.0e-6:
            unavailable.append((anchor_pressure, "anchor round trip failed"))
            continue
        if (
            dynamic is None
            or anchor_hvap is None
            or not math.isfinite(anchor_hvap)
            or anchor_hvap <= 0.0
            or not math.isfinite(anchor_slope)
            or anchor_slope <= 0.0
        ):
            unavailable.append((anchor_pressure, "invalid anchor state"))
            continue
        dynamic_ln_pressure, endpoint_omega, omega_slope = dynamic
        required_enthalpy = R * anchor_temperature**2 * anchor_slope
        clapeyron_scale = required_enthalpy / anchor_hvap
        try:
            anchor_delta_z = (
                None
                if case.delta_z is None
                else case.delta_z(anchor_temperature)
            )
        except (ArithmeticError, TypeError, ValueError):
            anchor_delta_z = None
        target_states = []
        for target_pressure in TARGET_PRESSURES_BAR:
            if target_pressure >= anchor_pressure:
                continue
            try:
                target_temperature = case.temperature_at_pressure(
                    target_pressure
                )
            except (ArithmeticError, ValueError):
                target_temperature = None
            if (
                target_temperature is None
                or not case.T_min <= target_temperature < anchor_temperature
            ):
                continue
            try:
                target_round_trip = case.ln_pressure(target_temperature)
            except (ArithmeticError, TypeError, ValueError):
                continue
            if (
                abs(
                    math.exp(target_round_trip) / target_pressure - 1.0
                )
                > 1.0e-6
            ):
                continue
            target_states.append((target_pressure, target_temperature))
        inferred_delta_z = anchor_hvap / required_enthalpy
        relaxed_delta_z_sqrt = {}
        relaxed_delta_z_linear = {}
        try:
            relaxed_delta_z_sqrt = relaxed_delta_z_predictions(
                case,
                anchor_temperature,
                anchor_pressure,
                inferred_delta_z,
                target_states,
                0.5,
            )
            relaxed_delta_z_linear = relaxed_delta_z_predictions(
                case,
                anchor_temperature,
                anchor_pressure,
                inferred_delta_z,
                target_states,
                1.0,
            )
        except (ArithmeticError, TypeError, ValueError):
            pass
        for target_pressure, target_temperature in target_states:
            try:
                hvap_integral = integrate_clapeyron(
                    anchor_temperature,
                    target_temperature,
                    lambda temperature: (
                        case.hvap_J_mol(temperature)
                        / (R * temperature**2)
                    ),
                )
                aw_ln_pressure = dynamic_ln_pressure(target_temperature)
                raw_clapeyron_ln_pressure = (
                    anchor_ln_pressure + hvap_integral
                )
                scaled_clapeyron_ln_pressure = (
                    anchor_ln_pressure
                    + clapeyron_scale * hvap_integral
                )
            except (ArithmeticError, TypeError, ValueError):
                continue
            exact_delta_z_ln_pressure = None
            if case.delta_z is not None:
                try:
                    exact_delta_z_integral = integrate_clapeyron(
                        anchor_temperature,
                        target_temperature,
                        lambda temperature: (
                            case.hvap_J_mol(temperature)
                            / (
                                R
                                * temperature**2
                                * case.delta_z(temperature)
                            )
                        ),
                    )
                    exact_delta_z_ln_pressure = (
                        anchor_ln_pressure + exact_delta_z_integral
                    )
                except (ArithmeticError, TypeError, ValueError):
                    exact_delta_z_ln_pressure = None
            try:
                reference_ln_pressure = case.ln_pressure(
                    target_temperature
                )
            except (ArithmeticError, TypeError, ValueError):
                continue

            def relative_error(predicted):
                return math.exp(predicted - reference_ln_pressure) - 1.0

            rows.append({
                "dataset": case.dataset,
                "cas": case.cas,
                "name": case.name,
                "is_acid": "acid" in case.name.lower(),
                "anchor_pressure_bar": anchor_pressure,
                "anchor_temperature_K": anchor_temperature,
                "anchor_reduced_temperature": anchor_temperature / case.Tc,
                "target_pressure_bar": target_pressure,
                "target_temperature_K": target_temperature,
                "target_reduced_temperature": target_temperature / case.Tc,
                "endpoint_omega": endpoint_omega,
                "omega_slope_per_K": omega_slope,
                "anchor_hvap_kJ_mol": anchor_hvap / 1000.0,
                "required_enthalpy_kJ_mol": required_enthalpy / 1000.0,
                "clapeyron_scale": clapeyron_scale,
                "inferred_delta_z": inferred_delta_z,
                "anchor_delta_z": anchor_delta_z,
                "dynamic_omega_relative_error": relative_error(
                    aw_ln_pressure
                ),
                "raw_clapeyron_relative_error": relative_error(
                    raw_clapeyron_ln_pressure
                ),
                "scaled_clapeyron_relative_error": relative_error(
                    scaled_clapeyron_ln_pressure
                ),
                "relaxed_delta_z_sqrt_relative_error": (
                    None
                    if target_pressure not in relaxed_delta_z_sqrt
                    else relative_error(
                        relaxed_delta_z_sqrt[target_pressure]
                    )
                ),
                "relaxed_delta_z_linear_relative_error": (
                    None
                    if target_pressure not in relaxed_delta_z_linear
                    else relative_error(
                        relaxed_delta_z_linear[target_pressure]
                    )
                ),
                "exact_delta_z_relative_error": (
                    None
                    if exact_delta_z_ln_pressure is None
                    else relative_error(exact_delta_z_ln_pressure)
                ),
            })
    return rows, unavailable


def perry_cases():
    library = PerryPropertyLibrary()
    library._load()
    cases = []
    unavailable = []
    for curve in load_curves():
        row = hvap_row_at_tb(library, curve, curve.t_min)
        if row is None:
            entry = library.get(curve.cas)
            rows = [] if entry is None else entry.get(
                "heat_of_vaporization",
                [],
            )
            row = next(iter(rows), None)
        if row is None:
            unavailable.append((curve.name, "missing Perry 2-69 Hvap"))
            continue
        T_min = max(curve.t_min, float(row["T_min_K"]))
        T_max = min(curve.t_max, float(row["T_max_K"]), curve.tc)
        if T_min >= T_max:
            unavailable.append((curve.name, "no common Psat/Hvap range"))
            continue

        def temperature_at_pressure(
            pressure_bar,
            curve=curve,
            T_min=T_min,
            T_max=T_max,
        ):
            target = math.log(pressure_bar)
            lower = perry_ln_p(curve, T_min) - target
            upper = perry_ln_p(curve, T_max) - target
            if lower == 0.0:
                return T_min
            if upper == 0.0:
                return T_max
            if lower * upper > 0.0:
                return None
            return brentq(
                lambda temperature: (
                    perry_ln_p(curve, temperature) - target
                ),
                T_min,
                T_max,
            )

        def hvap(
            temperature,
            library=library,
            row=row,
            curve=curve,
        ):
            value = correlation_hvap_kJ_mol(
                library,
                row,
                curve,
                temperature,
            )
            if value is None or not math.isfinite(value) or value <= 0.0:
                raise ValueError("invalid Perry Hvap")
            return value * 1000.0

        cases.append(DirectCompletionCase(
            dataset="Perry 2-8 Psat / 2-69 Hvap",
            cas=curve.cas,
            name=curve.name,
            Tc=curve.tc,
            Pc_bar=curve.pc_bar,
            preferred_omega=curve.omega,
            T_min=T_min,
            T_max=T_max,
            ln_pressure=lambda temperature, curve=curve: perry_ln_p(
                curve,
                temperature,
            ),
            dln_pressure_dT=(
                lambda temperature, curve=curve: perry_dlnp_dt(
                    curve,
                    temperature,
                )
            ),
            temperature_at_pressure=temperature_at_pressure,
            hvap_J_mol=hvap,
        ))
    return cases, unavailable


def coolprop_cases():
    CP = coolprop_module()
    aliases = coolprop_alias_map()
    if CP is None or aliases is None:
        return [], [("CoolProp", "unavailable")]
    cases = []
    unavailable = []
    seen_fluids = set()
    for compact_cas, fluid in aliases.items():
        if fluid in seen_fluids:
            continue
        seen_fluids.add(fluid)
        reference = CoolPropFluidReference(fluid=fluid, backend="HEOS")
        qualified = reference.qualified_name
        try:
            T_min = float(CP.PropsSI("Ttriple", qualified))
            T_max = float(CP.PropsSI("Tcrit", qualified))
            Pc_bar = float(CP.PropsSI("pcrit", qualified)) / 100000.0
            preferred_omega = float(CP.PropsSI("acentric", qualified))
            cas = str(CP.get_fluid_param_string(fluid, "CAS")).strip()
        except Exception:
            unavailable.append((fluid, "critical constants unavailable"))
            continue
        if not (0.0 < T_min < T_max) or Pc_bar <= 0.0:
            unavailable.append((fluid, "invalid temperature range"))
            continue

        def ln_pressure(
            temperature,
            reference=reference,
        ):
            pressure = coolprop_saturation_pressure(
                reference,
                temperature,
            )
            if pressure is None or pressure <= 0.0:
                raise ValueError("CoolProp Psat unavailable")
            return math.log(pressure / 100000.0)

        def derivative(
            temperature,
            reference=reference,
            T_min=T_min,
            T_max=T_max,
            pressure_function=ln_pressure,
        ):
            step = min(
                max(1.0e-5 * temperature, 1.0e-4),
                (temperature - T_min) / 4.0,
                (T_max - temperature) / 4.0,
            )
            if step <= 0.0:
                raise ValueError("CoolProp derivative unavailable")
            return (
                pressure_function(temperature + step)
                - pressure_function(temperature - step)
            ) / (2.0 * step)

        def temperature_at_pressure(
            pressure_bar,
            reference=reference,
            T_min=T_min,
            T_max=T_max,
        ):
            temperature = coolprop_saturation_temperature(
                reference,
                pressure_bar * 100000.0,
            )
            if (
                temperature is None
                or temperature < T_min
                or temperature > T_max
            ):
                return None
            return temperature

        def hvap(
            temperature,
            qualified=qualified,
        ):
            liquid = float(CP.PropsSI(
                "Hmolar",
                "T",
                temperature,
                "Q",
                0.0,
                qualified,
            ))
            vapor = float(CP.PropsSI(
                "Hmolar",
                "T",
                temperature,
                "Q",
                1.0,
                qualified,
            ))
            value = vapor - liquid
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("CoolProp Hvap unavailable")
            return value

        def delta_z(
            temperature,
            qualified=qualified,
        ):
            liquid = float(CP.PropsSI(
                "Z",
                "T",
                temperature,
                "Q",
                0.0,
                qualified,
            ))
            vapor = float(CP.PropsSI(
                "Z",
                "T",
                temperature,
                "Q",
                1.0,
                qualified,
            ))
            value = vapor - liquid
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("CoolProp delta Z unavailable")
            return value

        cases.append(DirectCompletionCase(
            dataset="CoolProp HEOS",
            cas=cas or compact_cas,
            name=fluid,
            Tc=T_max,
            Pc_bar=Pc_bar,
            preferred_omega=preferred_omega,
            T_min=T_min,
            T_max=T_max,
            ln_pressure=ln_pressure,
            dln_pressure_dT=derivative,
            temperature_at_pressure=temperature_at_pressure,
            hvap_J_mol=hvap,
            delta_z=delta_z,
        ))
    return cases, unavailable


def summarize_method(rows, method):
    values = [
        abs(row[f"{method}_relative_error"])
        for row in rows
        if row.get(f"{method}_relative_error") is not None
    ]
    if not values:
        return "n=0"
    return (
        f"n={len(values)} median={100 * median(values):.3f}% "
        f"p95={100 * percentile(values, 0.95):.3f}% "
        f"mean={100 * mean(values):.3f}% "
        f"max={100 * max(values):.3f}%"
    )


def report_group(rows, label):
    print(f"\n{label}: curves={len({row['cas'] for row in rows})} points={len(rows)}")
    methods = (
        "dynamic_omega",
        "raw_clapeyron",
        "scaled_clapeyron",
        "relaxed_delta_z_sqrt",
        "relaxed_delta_z_linear",
        "exact_delta_z",
    )
    for method in methods:
        print(f"  {method}: {summarize_method(rows, method)}")
    for candidate in (
        "raw_clapeyron",
        "scaled_clapeyron",
        "relaxed_delta_z_sqrt",
        "relaxed_delta_z_linear",
    ):
        paired = [
            row for row in rows
            if row.get(f"{candidate}_relative_error") is not None
        ]
        wins = sum(
            abs(row[f"{candidate}_relative_error"])
            < abs(row["dynamic_omega_relative_error"])
            for row in paired
        )
        print(
            f"  {candidate} wins versus dynamic omega: "
            f"{wins}/{len(paired)}"
        )
    scales = [row["clapeyron_scale"] for row in rows]
    print(
        f"  boundary scale: median={median(scales):.4f} "
        f"p05={percentile(scales, 0.05):.4f} "
        f"p95={percentile(scales, 0.95):.4f}"
    )
    for target_pressure in TARGET_PRESSURES_BAR:
        subset = [
            row for row in rows
            if row["target_pressure_bar"] == target_pressure
        ]
        if not subset:
            continue
        print(
            f"  target {target_pressure:g} bar: "
            + "; ".join(
                f"{method} {summarize_method(subset, method)}"
                for method in methods
            )
        )


def report(rows):
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        report_group(dataset_rows, dataset)
        for anchor_pressure in ANCHOR_PRESSURES_BAR:
            subset = [
                row for row in dataset_rows
                if row["anchor_pressure_bar"] == anchor_pressure
            ]
            if subset:
                report_group(
                    subset,
                    f"{dataset}; hard endpoint={anchor_pressure:g} bar",
                )
        if dataset.startswith("Perry"):
            acid_rows = [row for row in dataset_rows if row["is_acid"]]
            nonacid_rows = [
                row for row in dataset_rows if not row["is_acid"]
            ]
            if acid_rows:
                report_group(acid_rows, f"{dataset}; acids")
            if nonacid_rows:
                report_group(nonacid_rows, f"{dataset}; non-acids")
        if dataset.startswith("CoolProp"):
            nonhelium_rows = [
                row for row in dataset_rows
                if row["name"] != "Helium"
            ]
            if len(nonhelium_rows) != len(dataset_rows):
                report_group(
                    nonhelium_rows,
                    f"{dataset}; excluding helium",
                )


def main():
    perry, perry_unavailable = perry_cases()
    coolprop, coolprop_unavailable = coolprop_cases()
    cases = perry + coolprop
    print("DIRECT HARD-BOUNDARY COMPLETION BELOW 0.25 BAR")
    print(
        f"cases={len(cases)} Perry={len(perry)} "
        f"CoolProp={len(coolprop)}"
    )
    rows = []
    unavailable = []
    for index, case in enumerate(cases, 1):
        case_rows, case_unavailable = evaluate_case(case)
        rows.extend(case_rows)
        unavailable.extend(
            (case.dataset, case.name, anchor, reason)
            for anchor, reason in case_unavailable
        )
        if index % 50 == 0:
            print(f"evaluated {index}/{len(cases)} cases")
    report(rows)
    print(
        f"\nunavailable sources: Perry={len(perry_unavailable)} "
        f"CoolProp={len(coolprop_unavailable)}"
    )
    print(f"unavailable anchors={len(unavailable)}")
    if rows:
        keys = sorted({key for row in rows for key in row})
        with OUTPUT_PATH.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
