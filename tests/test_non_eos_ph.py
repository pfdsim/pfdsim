"""Homogeneous PH acceleration across non-EOS caloric models."""

from unittest.mock import patch

import pytest

from thermodynamics import create_thermodynamics
from unit_operations_basic import _ThermoStateSolver


@pytest.mark.parametrize(
    "model",
    [
        "IDEAL",
        "NRTL",
        "UNIQUAC",
        "UNIFAC",
        "UNIFAC2",
        "UNIFDMD",
        "UNIFM2",
        "UNIFNIST",
        "UNIFLBY",
        "NRTL-RK",
        "UNIQUAC-PR",
        "UNIFAC-BV",
    ],
)
@pytest.mark.parametrize("phase, temperature", [("liquid", 340.0), ("vapor", 440.0)])
def test_non_eos_ph_closes_authoritative_enthalpy_without_states(
    model, phase, temperature
):
    thermo = create_thermodynamics(["water", "ethanol"], model)
    composition = {"water": 0.4, "ethanol": 0.6}
    fraction = float(phase == "vapor")
    target = thermo.mixture_enthalpy(composition, temperature, fraction, P=2.0)
    solver = _ThermoStateSolver(thermo, "non-EOS PH test")
    with patch.object(
        thermo, "calculate_state", side_effect=AssertionError("full state built")
    ):
        actual, residual = solver.temperature_at_enthalpy(
            2.0,
            10.0,
            composition,
            target,
            temperature - 30.0,
            force_phase=phase,
        )
    assert abs(residual) < 1.0e-5
    assert (
        abs(thermo.mixture_enthalpy(composition, actual, fraction, P=2.0) - target)
        < 1.0e-5
    )
    state = thermo.calculate_state_PH(
        2.0,
        target,
        10.0,
        composition,
        phase=phase,
        T_guess=temperature - 30.0,
        include=("H",),
    )
    assert abs(state.H - target) < 1.0e-5
    assert state.vapor_fraction == fraction


@pytest.mark.parametrize("phase, temperature", [("liquid", 340.0), ("vapor", 440.0)])
def test_ph_without_compiled_curves_preserves_model_properties(phase, temperature):
    thermo = create_thermodynamics(["water", "ethanol"], "NRTL")
    composition = {"water": 0.4, "ethanol": 0.6}
    fraction = float(phase == "vapor")
    target = thermo.mixture_enthalpy(composition, temperature, fraction, P=2.0)
    with (
        patch.object(thermo, "_compiled_caloric_backend", return_value=None),
        patch.object(
            thermo, "calculate_state", side_effect=AssertionError("full state built")
        ),
    ):
        actual, residual = thermo.temperature_at_PH(
            2.0, target, composition, phase=phase, T_guess=300.0
        )
    assert abs(residual) < 1.0e-5
    assert (
        abs(thermo.mixture_enthalpy(composition, actual, fraction, P=2.0) - target)
        < 1.0e-5
    )


def test_ph_does_not_guess_phase_or_bypass_custom_enthalpy():
    thermo = create_thermodynamics(["N2"], "IDEAL")
    with pytest.raises(NotImplementedError):
        thermo.temperature_at_PH(1.0, 0.0, {"N2": 1.0})
    original = thermo.mixture_enthalpy
    thermo.mixture_enthalpy = lambda *args, **kwargs: original(*args, **kwargs) + 1000.0
    target = thermo.mixture_enthalpy({"N2": 1.0}, 600.0, P=1.0)
    temperature, residual = thermo.temperature_at_PH(
        1.0, target, {"N2": 2.0}, phase="gas", T_guess=400.0
    )
    assert abs(residual) < 1.0e-5
    assert (
        abs(thermo.mixture_enthalpy({"N2": 1.0}, temperature, P=1.0) - target) < 1.0e-5
    )


@pytest.mark.parametrize("model", ["NRTL", "UNIQUAC", "UNIFAC", "UNIFDMD", "UNIFNIST"])
def test_liquid_activity_ph_is_fused(model):
    from compiled_ph import CompiledActivityPHBackend

    thermo = create_thermodynamics(["water", "ethanol"], model)
    backend, evaluate = thermo._ph_caloric_evaluator(
        2.0, {"water": 0.4, "ethanol": 0.6}, "liquid"
    )
    assert isinstance(backend, CompiledActivityPHBackend)
    assert evaluate is None


@pytest.mark.parametrize("model", ["NRTL-RK", "NRTL-PR", "UNIQUAC-RK", "UNIFAC-PR"])
def test_gamma_phi_vapor_ph_reuses_compiled_departures(model):
    thermo = create_thermodynamics(["water", "ethanol"], model)
    composition = {"water": 0.4, "ethanol": 0.6}
    backend, evaluate = thermo._ph_caloric_evaluator(2.0, composition, "vapor")
    assert backend.cubic_backend is thermo.vapor_eos._compiled_backend
    assert evaluate is None
    target = thermo.mixture_enthalpy(composition, 440.0, 1.0, P=2.0)
    actual, residual = thermo.temperature_at_PH(
        2.0, target, composition, phase="vapor", T_guess=400.0
    )
    assert abs(residual) < 1.0e-5
    assert (
        abs(thermo.mixture_enthalpy(composition, actual, 1.0, P=2.0) - target) < 1.0e-5
    )


@pytest.mark.parametrize("model", ["NRTL", "UNIQUAC"])
def test_fused_ph_preserves_interaction_temperature_clamping(model):
    record = dict(
        component1="water",
        component2="ethanol",
        model=model,
        extrapolation="clamp",
        Tmin_K=320.0,
        Tmax_K=350.0,
    )
    if model == "NRTL":
        record.update(
            alpha12=0.3, tau12_c=0.2, tau12_d=100.0, tau21_c=-0.1, tau21_d=200.0
        )
    else:
        record.update(tau12_a=0.2, tau12_b=100.0, tau21_a=-0.1, tau21_b=200.0)
    thermo = create_thermodynamics(
        ["water", "ethanol"], model, interaction_overrides=[record]
    )
    composition = {"water": 0.4, "ethanol": 0.6}
    backend, _ = thermo._ph_caloric_evaluator(2.0, composition, "liquid")
    from compiled_ph import _activity_ph_enthalpy_cp_numba
    import numpy as np

    solver = _ThermoStateSolver(thermo, "clamped interactions")
    for T in (310.0, 320.0, 340.0, 350.0, 360.0):
        target = thermo.mixture_enthalpy(composition, T, 0.0, P=2.0)
        compiled_h, compiled_cp = _activity_ph_enthalpy_cp_numba(
            T,
            np.array([composition[c] for c in thermo.components]),
            backend.pure._cp_state(),
            backend.indices,
            backend.activity_states,
        )
        assert compiled_h == pytest.approx(target, abs=1.0e-7)
        assert compiled_cp == pytest.approx(
            thermo.mixture_Cp(composition, T, 0.0, 2.0), rel=1.0e-8
        )
        # Boundary kinks may require the authoritative bracketing fallback.
        actual, residual = solver.temperature_at_enthalpy(
            2.0,
            1.0,
            composition,
            target,
            T - 2.0,
            force_phase="liquid",
        )
        assert abs(residual) < 1.0e-5
        assert (
            abs(thermo.mixture_enthalpy(composition, actual, 0.0, P=2.0) - target)
            < 1.0e-5
        )


@pytest.mark.parametrize("phase, temperature", [("liquid", 320.0), ("vapor", 600.0)])
def test_steam_temperature_ph_uses_native_properties(phase, temperature):
    thermo = create_thermodynamics(["H2O"], "STEAM")
    target = thermo.calculate_state(
        temperature, 5.0, 1.0, {"H2O": 1.0}, phase=phase, include=("H",)
    ).H
    with patch.object(
        thermo, "calculate_state", side_effect=AssertionError("full state built")
    ):
        actual, residual = thermo.temperature_at_PH(
            5.0, target, {"H2O": 1.0}, phase=phase
        )
    assert abs(residual) < 1.0e-5
    state = thermo.calculate_state(
        actual, 5.0, 1.0, {"H2O": 1.0}, phase=phase, include=("H",)
    )
    assert abs(state.H - target) < 1.0e-5


def test_steam_temperature_ph_rejects_wet_target():
    thermo = create_thermodynamics(["H2O"], "STEAM")
    wet = thermo.calculate_state_PQ(5.0, 0.5, 1.0, {"H2O": 1.0}, include=("H",))
    for phase in ("liquid", "vapor"):
        with pytest.raises(NotImplementedError):
            thermo.temperature_at_PH(5.0, wet.H, {"H2O": 1.0}, phase=phase)


@pytest.mark.parametrize("phase, quality", [("liquid", 0.0), ("vapor", 1.0)])
def test_steam_temperature_ph_accepts_saturated_endpoints(phase, quality):
    thermo = create_thermodynamics(["H2O"], "STEAM")
    state = thermo.calculate_state_PQ(5.0, quality, 1.0, {"H2O": 1.0}, include=("H",))
    temperature, residual = thermo.temperature_at_PH(
        5.0, state.H, {"H2O": 1.0}, phase=phase
    )
    assert abs(residual) < 1.0e-5
    assert temperature == pytest.approx(state.T, abs=1.0e-6)


@pytest.mark.parametrize(
    "model", ["NRTL-VDM", "UNIQUAC-VDM", "UNIFAC-VDM", "UNIFDMD-VDM", "UNIFNIST-VDM"]
)
@pytest.mark.parametrize("phase, temperature", [("liquid", 340.0), ("vapor", 440.0)])
def test_vdm_ph_keeps_association_and_liquid_reference(model, phase, temperature):
    thermo = create_thermodynamics(["CH3COOH", "H2O"], model)
    composition = {"CH3COOH": 0.2, "H2O": 0.8}
    fraction = float(phase == "vapor")
    target = thermo.mixture_enthalpy(composition, temperature, fraction, P=2.0)
    with patch.object(
        thermo, "calculate_state", side_effect=AssertionError("full state built")
    ):
        actual, residual = thermo.temperature_at_PH(
            2.0, target, composition, phase=phase, T_guess=temperature - 20.0
        )
    assert abs(residual) < 1.0e-5
    assert (
        abs(thermo.mixture_enthalpy(composition, actual, fraction, P=2.0) - target)
        < 1.0e-5
    )


@pytest.mark.parametrize(
    "kind", ["constant", "polynomial", "shomate", "linear", "zabransky", "scaled"]
)
def test_compact_liquid_cp_forms_preserve_integrals_and_clamping(kind):
    from compiled_ph import CompiledPHBackend
    from property_resolution import liquid_cp as liquid
    from property_resolution.ideal_gas_cp import PolynomialCpKernel

    common = dict(Tmin=300.0, Tmax=500.0, quality=1.0, source="test", method="test")
    kernels = {
        "constant": lambda: liquid.ConstantLiquidCpKernel(**common, value=80.0),
        "polynomial": lambda: liquid.PolynomialLiquidCpKernel(
            **common, coefficients=(50.0, 0.1)
        ),
        "shomate": lambda: liquid.ShomateLiquidCpKernel(
            **common, coefficients=(60.0, 20.0, 1.0, 0.1, 0.2)
        ),
        "linear": lambda: liquid.load_bundled_liquid_kernel("64-17-5"),
        "zabransky": lambda: liquid.load_bundled_liquid_kernel("117-81-7"),
        "scaled": lambda: liquid.ScaledIdealGasLiquidCpKernel(
            **common,
            scale_factor=1.5,
            ideal_gas_kernel=PolynomialCpKernel(**common, coefficients=(40.0, 0.01)),
        ),
    }
    kernel = kernels[kind]()
    thermo = create_thermodynamics(["ethanol"], "IDEAL")
    thermo._liquid_cp_kernels["ethanol"] = kernel
    backend = CompiledPHBackend.from_thermo(thermo, phase="liquid")
    assert backend is not None
    reference = 1000.0 * (
        thermo.enthalpy_ideal_gas("ethanol", 298.15)
        - thermo.Hvap_at_T("ethanol", 298.15)
    )
    for T in (
        kernel.Tmin - 30.0,
        kernel.Tmin - 5.0,
        kernel.Tmin,
        (kernel.Tmin + kernel.Tmax) / 2.0,
        kernel.Tmax + 5.0,
        kernel.Tmax + 30.0,
    ):
        H, cp = backend.enthalpy_cp(T, 1.0, [1.0], "liquid")
        assert H == pytest.approx(reference + kernel.delta_h(298.15, T), abs=1.0e-7)
        assert cp == pytest.approx(kernel.cp(T), rel=1.0e-10)
