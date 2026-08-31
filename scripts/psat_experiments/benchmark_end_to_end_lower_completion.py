import csv
import math
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, median

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import PchipInterpolator

from benchmark_direct_lower_completion import coolprop_cases, perry_cases
from benchmark_perry_aw import percentile
from cubic_eos import CubicEOS
from property_resolution import (
    PsatAnchorRegistry,
    PsatBoundaryConditions,
    PsatCanonicalizationAdapter,
    PsatDerivativeBasis,
    PsatEndpoint,
    PsatSegment,
    PsatSegmentType,
    trim_psat_segment_to_validity,
)
from property_resolution.common import R
from property_resolution.vapor_pressure_adapter import (
    _collect_anchored_ambrose_walton_lower_inputs,
)


HARD_PRESSURES_BAR = (1.01325, 0.75, 0.50, 0.35)
TARGET_PRESSURES_BAR = (0.04, 0.02, 0.01, 0.005, 0.002, 0.001)
SWITCH_PRESSURE_BAR = 0.25
WATSON_EXPONENT = 0.38
FIVE_POINT_WATSON_TEMPERATURES_K = (
    293.15,
    313.15,
    333.15,
    353.15,
    373.15,
)
FIVE_POINT_WATSON_EXPONENT_BOUNDS = (0.20, 0.60)
TB_LOCAL_WATSON_SPACINGS_K = (20.0, 10.0)
DIMER_ENTHALPY_J_MOL = -60500.0
DIMER_SUPPRESSED_ENTROPY_J_MOL_K = -1000.0
OUTPUT_PATH = Path("/tmp/end_to_end_lower_completion_benchmark.csv")
COOLPROP_CACHE_POINTS = 121
MONOCARBOXYLIC_ACID_CAS = frozenset({
    "64-18-6",
    "64-19-7",
    "65-85-0",
    "79-09-4",
    "79-10-7",
    "79-31-2",
    "79-41-4",
    "88-09-5",
    "107-92-6",
    "109-52-4",
    "111-14-8",
    "112-05-0",
    "116-53-0",
    "124-07-2",
    "142-62-1",
    "149-57-5",
    "3004-93-1",
    "334-48-5",
})
METHODS = (
    "continued_lower_aw",
    "raw_clapeyron",
    "relaxed_delta_z_sqrt",
    "relaxed_delta_z_linear",
    "watson_raw_clapeyron",
    "watson_relaxed_delta_z_sqrt",
    "watson_relaxed_delta_z_linear",
    "five_point_watson_raw_clapeyron",
    "five_point_watson_relaxed_delta_z_sqrt",
    "five_point_watson_relaxed_delta_z_linear",
    "five_point_watson_unbounded_raw_clapeyron",
    "five_point_watson_unbounded_relaxed_delta_z_sqrt",
    "five_point_watson_unbounded_relaxed_delta_z_linear",
    "tb_local_watson_raw_clapeyron",
    "tb_local_watson_relaxed_delta_z_sqrt",
    "tb_local_watson_relaxed_delta_z_linear",
    "peng_robinson_delta_z_clapeyron",
    "effective_omega_peng_robinson_delta_z_clapeyron",
    "watson_peng_robinson_delta_z_clapeyron",
    "five_point_watson_peng_robinson_delta_z_clapeyron",
    "tb_local_watson_peng_robinson_delta_z_clapeyron",
    "exact_delta_z",
    "calibrated_dimer_clapeyron",
)


@dataclass(frozen=True)
class PreparedBoundary:
    case: object
    hard_pressure_bar: float
    hard_temperature_K: float
    source_derivative_basis: PsatDerivativeBasis
    bound_relation: PsatSegment
    switch_segment: PsatSegment
    switch_temperature_K: float
    switch_slope: float
    switch_reference_temperature_K: float
    switch_temperature_error_K: float
    switch_pressure_error: float
    switch_slope_error: float
    route: str
    quality: float
    switch_effective_omega: float
    Tb_K: float
    hvap_at_tb_J_mol: float


def cached_coolprop_case(case):
    if not case.dataset.startswith("CoolProp"):
        return case
    temperatures = []
    for pressure in (
        *TARGET_PRESSURES_BAR,
        SWITCH_PRESSURE_BAR,
        *HARD_PRESSURES_BAR,
    ):
        try:
            temperature = case.temperature_at_pressure(pressure)
        except (ArithmeticError, TypeError, ValueError):
            temperature = None
        if temperature is not None and case.T_min <= temperature <= case.T_max:
            temperatures.append(float(temperature))
    temperatures.extend(
        temperature
        for temperature in FIVE_POINT_WATSON_TEMPERATURES_K
        if case.T_min <= temperature <= case.T_max
    )
    if len(temperatures) < 2:
        return case
    grid = np.linspace(
        min(temperatures),
        max(temperatures),
        COOLPROP_CACHE_POINTS,
    )
    ln_pressures = np.asarray([
        case.ln_pressure(float(temperature))
        for temperature in grid
    ])
    hvaps = np.asarray([
        case.hvap_J_mol(float(temperature))
        for temperature in grid
    ])
    ln_pressure_interpolator = PchipInterpolator(
        grid,
        ln_pressures,
        extrapolate=True,
    )
    pressure_derivative = ln_pressure_interpolator.derivative()
    hvap_interpolator = PchipInterpolator(
        grid,
        hvaps,
        extrapolate=True,
    )
    delta_z_interpolator = None
    if case.delta_z is not None:
        delta_z_interpolator = PchipInterpolator(
            grid,
            np.asarray([
                case.delta_z(float(temperature))
                for temperature in grid
            ]),
            extrapolate=True,
        )
    return replace(
        case,
        ln_pressure=(
            lambda temperature: float(
                ln_pressure_interpolator(temperature)
            )
        ),
        dln_pressure_dT=(
            lambda temperature: float(
                pressure_derivative(temperature)
            )
        ),
        hvap_J_mol=(
            lambda temperature: float(
                hvap_interpolator(temperature)
            )
        ),
        delta_z=(
            None
            if delta_z_interpolator is None
            else (
                lambda temperature: float(
                    delta_z_interpolator(temperature)
                )
            )
        ),
    )


def production_lower_aw_boundary(case, hard_pressure_bar):
    hard_temperature = case.temperature_at_pressure(hard_pressure_bar)
    switch_reference_temperature = case.temperature_at_pressure(
        SWITCH_PRESSURE_BAR
    )
    Tb = case.temperature_at_pressure(1.01325)
    if (
        hard_temperature is None
        or switch_reference_temperature is None
        or Tb is None
        or hard_temperature <= switch_reference_temperature
        or not case.T_min <= switch_reference_temperature < hard_temperature
        or hard_temperature >= case.T_max
    ):
        return None
    for pressure, temperature in (
        (hard_pressure_bar, hard_temperature),
        (SWITCH_PRESSURE_BAR, switch_reference_temperature),
        (1.01325, Tb),
    ):
        if abs(
            math.exp(case.ln_pressure(temperature)) / pressure - 1.0
        ) > 1.0e-6:
            return None
    is_coolprop = case.dataset.startswith("CoolProp")
    source_quality = 0.99 if is_coolprop else 0.98
    component = {
        "name": case.name,
        "CAS": case.cas,
        "Tc": case.Tc,
        "Pc": case.Pc_bar,
        "omega": case.preferred_omega,
        "Tb": Tb,
        "property_sources": {
            field_name: {
                "source": case.dataset,
                "method": "benchmark_source",
                "quality": source_quality,
            }
            for field_name in ("Tc", "Pc", "omega", "Tb")
        },
    }
    relation_inputs = PsatCanonicalizationAdapter(
        component,
        input_methods=(
            _collect_anchored_ambrose_walton_lower_inputs,
        ),
    ).collect_inputs(
        T_min=case.T_min,
        T_critical=case.Tc,
        P_critical_bar=case.Pc_bar,
        T_boiling=Tb,
    )
    if not relation_inputs.relations:
        return None
    derivative_function = (
        None
        if is_coolprop
        else case.dln_pressure_dT
    )
    derivative_basis = (
        PsatDerivativeBasis.NUMERICAL
        if is_coolprop
        else PsatDerivativeBasis.ANALYTIC
    )
    hard_segment = PsatSegment(
        source=case.dataset,
        method="simulated_retained_hard_segment",
        segment_type=PsatSegmentType.PINNED,
        priority=900 if is_coolprop else 600,
        T_min=hard_temperature,
        T_max=case.T_max,
        ln_pressure_function=case.ln_pressure,
        derivative_function=derivative_function,
        derivative_basis=derivative_basis,
        quality=source_quality,
    )
    anchors = PsatAnchorRegistry.from_tb_tc(
        T_critical=case.Tc,
        P_critical_bar=case.Pc_bar,
        critical_quality=source_quality,
        T_boiling=Tb,
        boiling_quality=source_quality,
    )
    bound = relation_inputs.relations[0].bind(PsatBoundaryConditions(
        T_min=case.T_min,
        T_max=hard_temperature,
        right=PsatEndpoint.from_segment(
            hard_segment,
            hard_temperature,
        ),
        anchors=anchors.points,
    ))
    switch_segment = trim_psat_segment_to_validity(bound)
    switch_temperature = switch_segment.T_min
    switch_slope = switch_segment.dln_pressure_dT(switch_temperature)
    reference_switch_pressure = math.exp(case.ln_pressure(switch_temperature))
    reference_switch_slope = case.dln_pressure_dT(switch_temperature)
    hvap_at_tb = case.hvap_J_mol(Tb)
    if (
        not math.isfinite(hvap_at_tb)
        or hvap_at_tb <= 0.0
    ):
        return None
    return PreparedBoundary(
        case=case,
        hard_pressure_bar=hard_pressure_bar,
        hard_temperature_K=hard_temperature,
        source_derivative_basis=derivative_basis,
        bound_relation=bound,
        switch_segment=switch_segment,
        switch_temperature_K=switch_temperature,
        switch_slope=switch_slope,
        switch_reference_temperature_K=switch_reference_temperature,
        switch_temperature_error_K=(
            switch_temperature - switch_reference_temperature
        ),
        switch_pressure_error=(
            reference_switch_pressure / SWITCH_PRESSURE_BAR - 1.0
        ),
        switch_slope_error=(
            switch_slope / reference_switch_slope - 1.0
        ),
        route=str(bound.metadata["bound_lower_aw_route"]),
        quality=bound.quality,
        switch_effective_omega=(
            float(bound.metadata["bound_endpoint_omega"])
            + float(bound.metadata["bound_omega_slope_per_K"])
            * (
                switch_temperature - hard_temperature
            )
        ),
        Tb_K=Tb,
        hvap_at_tb_J_mol=hvap_at_tb,
    )


def watson_hvap_function(boundary):
    denominator = 1.0 - boundary.Tb_K / boundary.case.Tc
    if denominator <= 0.0:
        return None

    def hvap(temperature):
        numerator = 1.0 - temperature / boundary.case.Tc
        if numerator <= 0.0:
            raise ValueError("Watson Hvap is outside the subcritical range")
        return (
            boundary.hvap_at_tb_J_mol
            * (numerator / denominator) ** WATSON_EXPONENT
        )

    return hvap


def watson_fit_from_temperatures(boundary, temperatures):
    if not all(
        boundary.case.T_min <= temperature < boundary.case.Tc
        and temperature <= boundary.case.T_max
        for temperature in temperatures
    ):
        return None
    coordinates = []
    responses = []
    samples = []
    for temperature in temperatures:
        hvap = boundary.case.hvap_J_mol(temperature)
        reduced_distance = 1.0 - temperature / boundary.case.Tc
        if (
            not math.isfinite(hvap)
            or hvap <= 0.0
            or reduced_distance <= 0.0
        ):
            return None
        coordinates.append(math.log(reduced_distance))
        responses.append(math.log(hvap))
        samples.append((temperature, hvap))
    coordinate_mean = mean(coordinates)
    response_mean = mean(responses)
    denominator = sum(
        (coordinate - coordinate_mean) ** 2
        for coordinate in coordinates
    )
    if denominator <= 0.0:
        return None
    exponent = sum(
        (coordinate - coordinate_mean)
        * (response - response_mean)
        for coordinate, response in zip(coordinates, responses)
    ) / denominator
    bounded_exponent = min(
        FIVE_POINT_WATSON_EXPONENT_BOUNDS[1],
        max(
            FIVE_POINT_WATSON_EXPONENT_BOUNDS[0],
            exponent,
        ),
    )
    unbounded_log_scale = mean(
        response - exponent * coordinate
        for coordinate, response in zip(coordinates, responses)
    )
    unbounded_scale = math.exp(unbounded_log_scale)
    bounded_log_scale = mean(
        response - bounded_exponent * coordinate
        for coordinate, response in zip(coordinates, responses)
    )
    bounded_scale = math.exp(bounded_log_scale)

    def bounded_hvap(temperature):
        reduced_distance = 1.0 - temperature / boundary.case.Tc
        if reduced_distance <= 0.0:
            raise ValueError("Five-point Watson fit is above Tc")
        return bounded_scale * reduced_distance**bounded_exponent

    def unbounded_hvap(temperature):
        reduced_distance = 1.0 - temperature / boundary.case.Tc
        if reduced_distance <= 0.0:
            raise ValueError("Five-point Watson fit is above Tc")
        return unbounded_scale * reduced_distance**exponent

    bounded_fit_errors = [
        abs(bounded_hvap(temperature) / source_hvap - 1.0)
        for temperature, source_hvap in samples
    ]
    unbounded_fit_errors = [
        abs(unbounded_hvap(temperature) / source_hvap - 1.0)
        for temperature, source_hvap in samples
    ]
    return {
        "function": bounded_hvap,
        "unbounded_function": unbounded_hvap,
        "exponent": exponent,
        "bounded_exponent": bounded_exponent,
        "scale_J_mol": bounded_scale,
        "unbounded_scale_J_mol": unbounded_scale,
        "maximum_sample_relative_error": max(bounded_fit_errors),
        "unbounded_maximum_sample_relative_error": max(
            unbounded_fit_errors
        ),
        "sample_temperatures_K": tuple(temperatures),
    }


def five_point_watson_fit(boundary):
    return watson_fit_from_temperatures(
        boundary,
        FIVE_POINT_WATSON_TEMPERATURES_K,
    )


def tb_local_watson_fit(boundary):
    for spacing in TB_LOCAL_WATSON_SPACINGS_K:
        temperatures = tuple(
            boundary.Tb_K - spacing * index
            for index in range(5)
        )
        fit = watson_fit_from_temperatures(
            boundary,
            temperatures,
        )
        if fit is not None:
            fit["spacing_K"] = spacing
            return fit
    return None


def raw_clapeyron_predictions(boundary, targets, hvap_function):
    predictions = {}
    for pressure, temperature in targets:
        integral, _error = quad(
            lambda T: hvap_function(T) / (R * T**2),
            boundary.switch_temperature_K,
            temperature,
            epsabs=1.0e-9,
            epsrel=1.0e-9,
            limit=100,
        )
        predictions[pressure] = math.log(SWITCH_PRESSURE_BAR) + integral
    return predictions


def relaxed_delta_z_predictions(
    boundary,
    targets,
    hvap_function,
    pressure_power,
):
    if not targets:
        return {}, None
    anchor_hvap = hvap_function(boundary.switch_temperature_K)
    required_enthalpy = (
        R
        * boundary.switch_temperature_K**2
        * boundary.switch_slope
    )
    anchor_delta_z = anchor_hvap / required_enthalpy
    if (
        not math.isfinite(anchor_delta_z)
        or anchor_delta_z <= 0.0
    ):
        return {}, anchor_delta_z
    minimum_temperature = min(
        temperature for _pressure, temperature in targets
    )

    def derivative(temperature, state):
        pressure_ratio = min(
            1.0,
            max(
                0.0,
                math.exp(state[0]) / SWITCH_PRESSURE_BAR,
            ),
        )
        delta_z = (
            1.0
            + (anchor_delta_z - 1.0)
            * pressure_ratio**pressure_power
        )
        if delta_z <= 0.0:
            raise ValueError("Relaxed delta Z became nonpositive")
        return [
            hvap_function(temperature)
            / (R * temperature**2 * delta_z)
        ]

    solution = solve_ivp(
        derivative,
        (
            boundary.switch_temperature_K,
            minimum_temperature,
        ),
        [math.log(SWITCH_PRESSURE_BAR)],
        rtol=2.0e-9,
        atol=2.0e-11,
        dense_output=True,
        max_step=max(
            (
                boundary.switch_temperature_K
                - minimum_temperature
            )
            / 25.0,
            0.1,
        ),
    )
    if not solution.success:
        return {}, anchor_delta_z
    return {
        pressure: float(solution.sol(temperature)[0])
        for pressure, temperature in targets
    }, anchor_delta_z


def exact_delta_z_predictions(boundary, targets):
    if boundary.case.delta_z is None:
        return {}
    predictions = {}
    for pressure, temperature in targets:
        integral, _error = quad(
            lambda T: (
                boundary.case.hvap_J_mol(T)
                / (
                    R
                    * T**2
                    * boundary.case.delta_z(T)
                )
            ),
            boundary.switch_temperature_K,
            temperature,
            epsabs=1.0e-9,
            epsrel=1.0e-9,
            limit=100,
        )
        predictions[pressure] = math.log(SWITCH_PRESSURE_BAR) + integral
    return predictions


def peng_robinson_delta_z(
    boundary,
    temperature,
    pressure_bar,
    omega,
):
    reduced_temperature = temperature / boundary.case.Tc
    reduced_pressure = pressure_bar / boundary.case.Pc_bar
    if (
        not math.isfinite(reduced_temperature)
        or reduced_temperature <= 0.0
        or reduced_temperature >= 1.0
        or not math.isfinite(reduced_pressure)
        or reduced_pressure <= 0.0
        or omega is None
        or not math.isfinite(omega)
    ):
        raise ValueError("Peng-Robinson state is unavailable")
    kappa = (
        0.37464
        + 1.54226 * omega
        - 0.26992 * omega**2
    )
    alpha = (
        1.0
        + kappa
        * (1.0 - math.sqrt(reduced_temperature))
    ) ** 2
    A = (
        0.45724
        * alpha
        * reduced_pressure
        / reduced_temperature**2
    )
    B = (
        0.07780
        * reduced_pressure
        / reduced_temperature
    )
    roots = sorted(
        root
        for root in CubicEOS._solve_monic_cubic(
            -(1.0 - B),
            A - 3.0 * B**2 - 2.0 * B,
            -(A * B - B**2 - B**3),
        )
        if root > B + 1.0e-12
    )
    if len(roots) < 2:
        raise ValueError("Peng-Robinson has no distinct saturated roots")
    delta_z = roots[-1] - roots[0]
    if not math.isfinite(delta_z) or delta_z <= 0.0:
        raise ValueError("Peng-Robinson delta Z is invalid")
    return delta_z


def peng_robinson_predictions(
    boundary,
    targets,
    omega,
    hvap_function=None,
):
    if not targets:
        return {}, None, None
    anchor_pr_delta_z = peng_robinson_delta_z(
        boundary,
        boundary.switch_temperature_K,
        SWITCH_PRESSURE_BAR,
        omega,
    )
    minimum_temperature = min(
        temperature for _pressure, temperature in targets
    )
    hvap = (
        boundary.case.hvap_J_mol
        if hvap_function is None
        else hvap_function
    )

    def derivative(temperature, state):
        pressure = math.exp(state[0])
        delta_z = peng_robinson_delta_z(
            boundary,
            temperature,
            pressure,
            omega,
        )
        return [
            hvap(temperature)
            / (R * temperature**2 * delta_z)
        ]

    solution = solve_ivp(
        derivative,
        (
            boundary.switch_temperature_K,
            minimum_temperature,
        ),
        [math.log(SWITCH_PRESSURE_BAR)],
        rtol=2.0e-9,
        atol=2.0e-11,
        dense_output=True,
        max_step=max(
            (
                boundary.switch_temperature_K
                - minimum_temperature
            )
            / 25.0,
            0.1,
        ),
    )
    if not solution.success:
        return {}, anchor_pr_delta_z
    return {
        pressure: float(solution.sol(temperature)[0])
        for pressure, temperature in targets
    }, anchor_pr_delta_z


def dimer_extent(temperature, ln_pressure, entropy):
    exponent = (
        entropy / R
        - DIMER_ENTHALPY_J_MOL / (R * temperature)
        + ln_pressure
    )
    equilibrium_pressure_product = math.exp(
        min(700.0, max(-700.0, exponent))
    )
    return 0.5 * (
        1.0
        - 1.0 / math.sqrt(
            1.0 + 4.0 * equilibrium_pressure_product
        )
    )


def dimer_entropy_from_boundary(boundary):
    anchor_hvap = boundary.case.hvap_J_mol(
        boundary.switch_temperature_K
    )
    required_enthalpy = (
        R
        * boundary.switch_temperature_K**2
        * boundary.switch_slope
    )
    if required_enthalpy <= anchor_hvap:
        return DIMER_SUPPRESSED_ENTROPY_J_MOL_K, 0.0
    extent = min(
        0.499999,
        max(
            0.0,
            1.0 - anchor_hvap / required_enthalpy,
        ),
    )
    if extent <= 1.0e-10:
        return DIMER_SUPPRESSED_ENTROPY_J_MOL_K, extent
    pressure_function = (
        extent
        * (1.0 - extent)
        / (1.0 - 2.0 * extent) ** 2
    )
    equilibrium_constant = (
        pressure_function / SWITCH_PRESSURE_BAR
    )
    entropy = (
        R * math.log(equilibrium_constant)
        + DIMER_ENTHALPY_J_MOL
        / boundary.switch_temperature_K
    )
    return entropy, extent


def dimer_predictions(boundary, targets):
    if boundary.case.cas not in MONOCARBOXYLIC_ACID_CAS or not targets:
        return {}, None, None
    entropy, anchor_extent = dimer_entropy_from_boundary(boundary)
    minimum_temperature = min(
        temperature for _pressure, temperature in targets
    )

    def derivative(temperature, state):
        extent = dimer_extent(
            temperature,
            state[0],
            entropy,
        )
        return [
            boundary.case.hvap_J_mol(temperature)
            / (
                R
                * temperature**2
                * (1.0 - extent)
            )
        ]

    solution = solve_ivp(
        derivative,
        (
            boundary.switch_temperature_K,
            minimum_temperature,
        ),
        [math.log(SWITCH_PRESSURE_BAR)],
        rtol=2.0e-9,
        atol=2.0e-11,
        dense_output=True,
        max_step=max(
            (
                boundary.switch_temperature_K
                - minimum_temperature
            )
            / 25.0,
            0.1,
        ),
    )
    if not solution.success:
        return {}, entropy, anchor_extent
    return {
        pressure: float(solution.sol(temperature)[0])
        for pressure, temperature in targets
    }, entropy, anchor_extent


def evaluate_boundary(boundary):
    targets = []
    for pressure in TARGET_PRESSURES_BAR:
        try:
            temperature = boundary.case.temperature_at_pressure(
                pressure
            )
        except (ArithmeticError, TypeError, ValueError):
            temperature = None
        if (
            temperature is None
            or not boundary.case.T_min <= temperature
            or temperature >= boundary.switch_temperature_K
        ):
            continue
        if abs(
            math.exp(boundary.case.ln_pressure(temperature))
            / pressure
            - 1.0
        ) > 1.0e-6:
            continue
        targets.append((pressure, temperature))
    if not targets:
        return []

    full_hvap = boundary.case.hvap_J_mol
    watson_hvap = watson_hvap_function(boundary)
    five_point_watson = five_point_watson_fit(boundary)
    tb_local_watson = tb_local_watson_fit(boundary)
    predictions = {
        "continued_lower_aw": {
            pressure: boundary.bound_relation.raw_ln_pressure(
                temperature
            )
            for pressure, temperature in targets
        },
        "raw_clapeyron": raw_clapeyron_predictions(
            boundary,
            targets,
            full_hvap,
        ),
        "exact_delta_z": exact_delta_z_predictions(
            boundary,
            targets,
        ),
    }
    (
        predictions["peng_robinson_delta_z_clapeyron"],
        peng_robinson_anchor_delta_z,
    ) = peng_robinson_predictions(
        boundary,
        targets,
        boundary.case.preferred_omega,
    )
    (
        predictions[
            "effective_omega_peng_robinson_delta_z_clapeyron"
        ],
        effective_omega_peng_robinson_anchor_delta_z,
    ) = peng_robinson_predictions(
        boundary,
        targets,
        boundary.switch_effective_omega,
    )
    (
        predictions["relaxed_delta_z_sqrt"],
        full_anchor_delta_z,
    ) = relaxed_delta_z_predictions(
        boundary,
        targets,
        full_hvap,
        0.5,
    )
    (
        predictions["relaxed_delta_z_linear"],
        _full_anchor_delta_z_linear,
    ) = relaxed_delta_z_predictions(
        boundary,
        targets,
        full_hvap,
        1.0,
    )
    watson_anchor_delta_z = None
    if watson_hvap is not None:
        predictions["watson_raw_clapeyron"] = (
            raw_clapeyron_predictions(
                boundary,
                targets,
                watson_hvap,
            )
        )
        (
            predictions["watson_relaxed_delta_z_sqrt"],
            watson_anchor_delta_z,
        ) = relaxed_delta_z_predictions(
            boundary,
            targets,
            watson_hvap,
            0.5,
        )
        (
            predictions["watson_relaxed_delta_z_linear"],
            _watson_anchor_delta_z_linear,
        ) = relaxed_delta_z_predictions(
            boundary,
            targets,
            watson_hvap,
            1.0,
        )
        (
            predictions["watson_peng_robinson_delta_z_clapeyron"],
            _watson_peng_robinson_anchor_delta_z,
        ) = peng_robinson_predictions(
            boundary,
            targets,
            boundary.case.preferred_omega,
            watson_hvap,
        )
    five_point_anchor_delta_z = None
    if five_point_watson is not None:
        five_point_hvap = five_point_watson["function"]
        five_point_unbounded_hvap = five_point_watson[
            "unbounded_function"
        ]
        predictions["five_point_watson_raw_clapeyron"] = (
            raw_clapeyron_predictions(
                boundary,
                targets,
                five_point_hvap,
            )
        )
        (
            predictions["five_point_watson_relaxed_delta_z_sqrt"],
            five_point_anchor_delta_z,
        ) = relaxed_delta_z_predictions(
            boundary,
            targets,
            five_point_hvap,
            0.5,
        )
        (
            predictions["five_point_watson_relaxed_delta_z_linear"],
            _five_point_anchor_delta_z_linear,
        ) = relaxed_delta_z_predictions(
            boundary,
            targets,
            five_point_hvap,
            1.0,
        )
        predictions[
            "five_point_watson_unbounded_raw_clapeyron"
        ] = raw_clapeyron_predictions(
            boundary,
            targets,
            five_point_unbounded_hvap,
        )
        (
            predictions[
                "five_point_watson_unbounded_relaxed_delta_z_sqrt"
            ],
            _five_point_unbounded_anchor_delta_z,
        ) = relaxed_delta_z_predictions(
            boundary,
            targets,
            five_point_unbounded_hvap,
            0.5,
        )
        (
            predictions[
                "five_point_watson_unbounded_relaxed_delta_z_linear"
            ],
            _five_point_unbounded_anchor_delta_z_linear,
        ) = relaxed_delta_z_predictions(
            boundary,
            targets,
            five_point_unbounded_hvap,
            1.0,
        )
        (
            predictions[
                "five_point_watson_peng_robinson_delta_z_clapeyron"
            ],
            _five_point_peng_robinson_anchor_delta_z,
        ) = peng_robinson_predictions(
            boundary,
            targets,
            boundary.case.preferred_omega,
            five_point_hvap,
        )
    tb_local_anchor_delta_z = None
    if tb_local_watson is not None:
        tb_local_hvap = tb_local_watson["function"]
        predictions["tb_local_watson_raw_clapeyron"] = (
            raw_clapeyron_predictions(
                boundary,
                targets,
                tb_local_hvap,
            )
        )
        (
            predictions["tb_local_watson_relaxed_delta_z_sqrt"],
            tb_local_anchor_delta_z,
        ) = relaxed_delta_z_predictions(
            boundary,
            targets,
            tb_local_hvap,
            0.5,
        )
        (
            predictions["tb_local_watson_relaxed_delta_z_linear"],
            _tb_local_anchor_delta_z_linear,
        ) = relaxed_delta_z_predictions(
            boundary,
            targets,
            tb_local_hvap,
            1.0,
        )
        (
            predictions[
                "tb_local_watson_peng_robinson_delta_z_clapeyron"
            ],
            _tb_local_peng_robinson_anchor_delta_z,
        ) = peng_robinson_predictions(
            boundary,
            targets,
            boundary.case.preferred_omega,
            tb_local_hvap,
        )
    (
        predictions["calibrated_dimer_clapeyron"],
        dimer_entropy,
        dimer_anchor_extent,
    ) = dimer_predictions(boundary, targets)

    rows = []
    for pressure, temperature in targets:
        reference_ln_pressure = boundary.case.ln_pressure(
            temperature
        )
        row = {
            "dataset": boundary.case.dataset,
            "cas": boundary.case.cas,
            "name": boundary.case.name,
            "is_acid": "acid" in boundary.case.name.lower(),
            "is_monocarboxylic_acid": (
                boundary.case.cas in MONOCARBOXYLIC_ACID_CAS
            ),
            "hard_pressure_bar": boundary.hard_pressure_bar,
            "hard_temperature_K": boundary.hard_temperature_K,
            "hard_derivative_basis": (
                boundary.source_derivative_basis.value
            ),
            "lower_aw_route": boundary.route,
            "lower_aw_quality": boundary.quality,
            "switch_temperature_K": boundary.switch_temperature_K,
            "reference_switch_temperature_K": (
                boundary.switch_reference_temperature_K
            ),
            "switch_temperature_error_K": (
                boundary.switch_temperature_error_K
            ),
            "switch_pressure_error": boundary.switch_pressure_error,
            "switch_slope_error": boundary.switch_slope_error,
            "switch_slope": boundary.switch_slope,
            "full_anchor_delta_z": full_anchor_delta_z,
            "peng_robinson_anchor_delta_z": (
                peng_robinson_anchor_delta_z
            ),
            "effective_omega_peng_robinson_anchor_delta_z": (
                effective_omega_peng_robinson_anchor_delta_z
            ),
            "switch_effective_omega": (
                boundary.switch_effective_omega
            ),
            "watson_anchor_delta_z": watson_anchor_delta_z,
            "five_point_watson_anchor_delta_z": (
                five_point_anchor_delta_z
            ),
            "tb_local_watson_anchor_delta_z": (
                tb_local_anchor_delta_z
            ),
            "watson_reference_temperature_K": boundary.Tb_K,
            "watson_reference_hvap_kJ_mol": (
                boundary.hvap_at_tb_J_mol / 1000.0
            ),
            "five_point_watson_exponent": (
                None
                if five_point_watson is None
                else five_point_watson["exponent"]
            ),
            "five_point_watson_bounded_exponent": (
                None
                if five_point_watson is None
                else five_point_watson["bounded_exponent"]
            ),
            "five_point_watson_maximum_sample_relative_error": (
                None
                if five_point_watson is None
                else five_point_watson[
                    "maximum_sample_relative_error"
                ]
            ),
            "five_point_watson_unbounded_maximum_sample_relative_error": (
                None
                if five_point_watson is None
                else five_point_watson[
                    "unbounded_maximum_sample_relative_error"
                ]
            ),
            "tb_local_watson_spacing_K": (
                None
                if tb_local_watson is None
                else tb_local_watson["spacing_K"]
            ),
            "tb_local_watson_exponent": (
                None
                if tb_local_watson is None
                else tb_local_watson["exponent"]
            ),
            "tb_local_watson_bounded_exponent": (
                None
                if tb_local_watson is None
                else tb_local_watson["bounded_exponent"]
            ),
            "tb_local_watson_maximum_sample_relative_error": (
                None
                if tb_local_watson is None
                else tb_local_watson[
                    "maximum_sample_relative_error"
                ]
            ),
            "dimer_entropy_J_mol_K": dimer_entropy,
            "dimer_anchor_extent": dimer_anchor_extent,
            "target_pressure_bar": pressure,
            "target_temperature_K": temperature,
        }
        for method in METHODS:
            predicted = predictions.get(method, {}).get(pressure)
            row[f"{method}_relative_error"] = (
                None
                if predicted is None
                else math.exp(
                    predicted - reference_ln_pressure
                ) - 1.0
            )
        rows.append(row)
    return rows


def method_statistics(rows, method):
    selected = [
        row for row in rows
        if row.get(f"{method}_relative_error") is not None
    ]
    if not selected:
        return None
    point_errors = [
        abs(float(row[f"{method}_relative_error"]))
        for row in selected
    ]
    case_errors = {}
    for row in selected:
        key = (
            row["dataset"],
            row["cas"],
            row["hard_pressure_bar"],
        )
        case_errors.setdefault(key, []).append(
            abs(float(row[f"{method}_relative_error"]))
        )
    case_mards = [
        mean(values) for values in case_errors.values()
    ]
    return {
        "points": len(point_errors),
        "cases": len(case_mards),
        "point_median": median(point_errors),
        "point_p95": percentile(point_errors, 0.95),
        "point_maximum": max(point_errors),
        "case_median": median(case_mards),
        "case_p95": percentile(case_mards, 0.95),
        "case_maximum": max(case_mards),
    }


def report_group(rows, label):
    print(
        f"\n{label}: "
        f"curves={len({row['cas'] for row in rows})} "
        f"cases={len({(row['cas'], row['hard_pressure_bar']) for row in rows})} "
        f"points={len(rows)}"
    )
    print(
        "method".ljust(39)
        + " point med/p95/max   curve-MARD med/p95/max"
    )
    for method in METHODS:
        statistics = method_statistics(rows, method)
        if statistics is None:
            continue
        print(
            method.ljust(39)
            + f" {100 * statistics['point_median']:6.3f}/"
            + f"{100 * statistics['point_p95']:6.3f}/"
            + f"{100 * statistics['point_maximum']:7.3f}"
            + f"   {100 * statistics['case_median']:6.3f}/"
            + f"{100 * statistics['case_p95']:6.3f}/"
            + f"{100 * statistics['case_maximum']:7.3f}"
        )
    switch_cases = {}
    for row in rows:
        key = (
            row["dataset"],
            row["cas"],
            row["hard_pressure_bar"],
        )
        switch_cases[key] = row
    switch_rows = list(switch_cases.values())
    for field_name, label_text, multiplier in (
        ("switch_temperature_error_K", "|switch dT| K", 1.0),
        ("switch_pressure_error", "|switch dP| %", 100.0),
        ("switch_slope_error", "|switch slope| %", 100.0),
    ):
        values = [
            abs(float(row[field_name]))
            for row in switch_rows
        ]
        print(
            f"  {label_text}: "
            f"median={multiplier * median(values):.4f} "
            f"p95={multiplier * percentile(values, 0.95):.4f} "
            f"max={multiplier * max(values):.4f}"
        )


def report(rows):
    for dataset in sorted({row["dataset"] for row in rows}):
        dataset_rows = [
            row for row in rows
            if row["dataset"] == dataset
        ]
        report_group(dataset_rows, dataset)
        for hard_pressure in HARD_PRESSURES_BAR:
            subset = [
                row for row in dataset_rows
                if row["hard_pressure_bar"] == hard_pressure
            ]
            if subset:
                report_group(
                    subset,
                    f"{dataset}; hard endpoint={hard_pressure:g} bar",
                )
        ordinary = [
            row for row in dataset_rows
            if not row["is_acid"]
        ]
        monoacids = [
            row for row in dataset_rows
            if row["is_monocarboxylic_acid"]
        ]
        other_acids = [
            row for row in dataset_rows
            if row["is_acid"]
            and not row["is_monocarboxylic_acid"]
        ]
        if ordinary:
            report_group(ordinary, f"{dataset}; ordinary fluids")
        if monoacids:
            report_group(
                monoacids,
                f"{dataset}; monocarboxylic acids",
            )
        if other_acids:
            report_group(
                other_acids,
                f"{dataset}; other acids",
            )


def main():
    perry, perry_unavailable = perry_cases()
    coolprop, coolprop_unavailable = coolprop_cases()
    cases = perry + coolprop
    rows = []
    unavailable = []
    print("END-TO-END LOWER PSAT COMPLETION")
    print(
        f"cases={len(cases)} Perry={len(perry)} "
        f"CoolProp={len(coolprop)}"
    )
    for case_index, original_case in enumerate(cases, 1):
        case = cached_coolprop_case(original_case)
        for hard_pressure in HARD_PRESSURES_BAR:
            try:
                boundary = production_lower_aw_boundary(
                    case,
                    hard_pressure,
                )
                if boundary is None:
                    unavailable.append((
                        case.dataset,
                        case.name,
                        hard_pressure,
                        "boundary unavailable",
                    ))
                    continue
                case_rows = evaluate_boundary(boundary)
            except (
                ArithmeticError,
                TypeError,
                ValueError,
            ) as error:
                unavailable.append((
                    case.dataset,
                    case.name,
                    hard_pressure,
                    str(error),
                ))
                continue
            if not case_rows:
                unavailable.append((
                    case.dataset,
                    case.name,
                    hard_pressure,
                    "no target points",
                ))
                continue
            rows.extend(case_rows)
        if case_index % 50 == 0:
            print(
                f"evaluated {case_index}/{len(cases)} cases",
                flush=True,
            )
    report(rows)
    print(
        f"\nsource unavailable: Perry={len(perry_unavailable)} "
        f"CoolProp={len(coolprop_unavailable)}"
    )
    print(f"unavailable hard endpoints={len(unavailable)}")
    if unavailable:
        for item in unavailable[:20]:
            print("  unavailable:", item)
    if rows:
        keys = sorted({key for row in rows for key in row})
        with OUTPUT_PATH.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
