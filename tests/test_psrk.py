import json
import math
import tempfile
import unittest
from pathlib import Path

from thermodynamics_models.psrk import (
    PSRK,
    PSRKUnsupportedComponentError,
    R_BAR_CM3_PER_MOL_K,
)


ETHANOL = "64-17-5"
WATER = "7732-18-5"
METHANE = "74-82-8"
METHANOL = "67-56-1"
CARBON_DIOXIDE = "124-38-9"
CARBON_MONOXIDE = "630-08-0"
HYDROGEN = "1333-74-0"


class PSRKTests(unittest.TestCase):
    def test_compiled_kernel_matches_python_reference(self):
        cases = (
            (
                [ETHANOL, WATER],
                350.0,
                5.0,
                {ETHANOL: 0.4, WATER: 0.6},
            ),
            (
                [METHANE, WATER],
                420.0,
                25.0,
                {METHANE: 0.15, WATER: 0.85},
            ),
            (
                [HYDROGEN, CARBON_MONOXIDE, CARBON_DIOXIDE, METHANOL, ETHANOL, WATER],
                343.15,
                1.2,
                {
                    HYDROGEN: 0.001,
                    CARBON_MONOXIDE: 0.001,
                    CARBON_DIOXIDE: 0.008,
                    METHANOL: 0.47,
                    ETHANOL: 0.325,
                    WATER: 0.195,
                },
            ),
        )
        for components, temperature, pressure, composition in cases:
            with self.subTest(components=components):
                model = PSRK(components)
                if model._compiled_backend is None:
                    self.skipTest("Numba compiled PSRK backend is unavailable")
                self.assertTrue(model._compiled_backend.compilation_complete)
                normalized = model._normalize_composition(composition)
                reference = model._mixture_parameters_reference(
                    temperature,
                    normalized,
                )
                compiled = model._mixture_parameters(
                    temperature,
                    normalized,
                )
                for name in ('b', 'D', 'dD_dT'):
                    self.assertAlmostEqual(
                        getattr(compiled, name),
                        getattr(reference, name),
                        places=12,
                    )
                for compiled_values, reference_values in (
                    (compiled.pure_a, reference.pure_a),
                    (compiled.pure_d, reference.pure_d),
                    (compiled.sigma, reference.sigma),
                ):
                    for compiled_value, reference_value in zip(
                        compiled_values,
                        reference_values,
                    ):
                        self.assertAlmostEqual(
                            compiled_value,
                            reference_value,
                            places=11,
                        )

                reference_activity = (
                    model._ln_activity_coefficients_and_derivative(
                        normalized,
                        temperature,
                    )
                )
                compiled_activity = model._compiled_backend.excess_gibbs_state(
                    normalized,
                    temperature,
                )
                for compiled_values, reference_values in zip(
                    compiled_activity,
                    reference_activity,
                ):
                    for compiled_value, reference_value in zip(
                        compiled_values,
                        reference_values,
                    ):
                        self.assertAlmostEqual(
                            float(compiled_value),
                            reference_value,
                            places=11,
                        )

                compiled_phi = model.fugacity_coefficients(
                    temperature,
                    pressure,
                    composition,
                    phase='liquid',
                )
                compiled_K = model.phi_phi_K_values(
                    temperature,
                    pressure,
                    composition,
                )
                model._compiled_backend = None
                model._mixture_parameter_cache.clear()
                model._phi_phi_k_cache.clear()
                reference_phi = model.fugacity_coefficients(
                    temperature,
                    pressure,
                    composition,
                    phase='liquid',
                )
                for component in components:
                    self.assertAlmostEqual(
                        compiled_phi[component],
                        reference_phi[component],
                        places=11,
                    )
                reference_K = model.phi_phi_K_values(
                    temperature,
                    pressure,
                    composition,
                )
                for component in components:
                    self.assertAlmostEqual(
                        compiled_K[component],
                        reference_K[component],
                        places=10,
                    )

    def test_pure_component_reduces_to_srk_fugacity(self):
        model = PSRK([WATER])
        temperature = 500.0
        pressure = 10.0
        composition = (1.0,)
        mixture = model._mixture_parameters(temperature, composition)
        B = mixture.b * pressure / (R_BAR_CM3_PER_MOL_K * temperature)
        A = mixture.D * B
        Z = max(model._compressibility_roots(A, B))

        expected_ln_phi = (
            Z - 1.0
            - math.log(Z - B)
            - mixture.D * math.log1p(B / Z)
        )
        result = model.fugacity_coefficients(
            temperature,
            pressure,
            {WATER: 1.0},
        )

        self.assertAlmostEqual(result[WATER], math.exp(expected_ln_phi), places=12)
        self.assertAlmostEqual(mixture.sigma[0], mixture.D, places=12)

    def test_binary_fugacity_matches_helmholtz_composition_derivative(self):
        model = PSRK([ETHANOL, WATER])
        temperature = 350.0
        pressure = 5.0
        mole_numbers = [0.4, 0.6]
        composition = tuple(mole_numbers)
        mixture = model._mixture_parameters(temperature, composition)
        B = mixture.b * pressure / (R_BAR_CM3_PER_MOL_K * temperature)
        A = mixture.D * B
        Z = max(model._compressibility_roots(A, B))
        total_volume = Z * R_BAR_CM3_PER_MOL_K * temperature / pressure

        def residual_helmholtz(numbers):
            total = sum(numbers)
            fractions = tuple(value / total for value in numbers)
            varied = model._mixture_parameters(temperature, fractions)
            packing = total * varied.b / total_volume
            return (
                -total * math.log(1.0 - packing)
                - total * varied.D * math.log1p(packing)
            )

        phi = model.fugacity_coefficients(
            temperature,
            pressure,
            dict(zip(model.components, composition)),
        )
        step = 1.0e-6
        for index, cas in enumerate(model.components):
            upper = mole_numbers.copy()
            lower = mole_numbers.copy()
            upper[index] += step
            lower[index] -= step
            chemical_potential_residual = (
                residual_helmholtz(upper) - residual_helmholtz(lower)
            ) / (2.0 * step)
            numerical_ln_phi = chemical_potential_residual - math.log(Z)
            self.assertAlmostEqual(math.log(phi[cas]), numerical_ln_phi, places=8)

    def test_binary_numeric_regression_and_component_order_invariance(self):
        forward = PSRK([ETHANOL, WATER])
        reverse = PSRK([WATER, ETHANOL])
        composition = {ETHANOL: 0.5, WATER: 0.5}

        forward_phi = forward.fugacity_coefficients(350.0, 1.0, composition)
        reverse_phi = reverse.fugacity_coefficients(350.0, 1.0, composition)
        liquid_phi = forward.fugacity_coefficients(
            350.0,
            1.0,
            composition,
            phase="liquid",
        )

        self.assertAlmostEqual(forward_phi[ETHANOL], 0.9794176273509974, places=12)
        self.assertAlmostEqual(forward_phi[WATER], 0.9900967543715719, places=12)
        self.assertAlmostEqual(forward_phi[ETHANOL], reverse_phi[ETHANOL], places=12)
        self.assertAlmostEqual(forward_phi[WATER], reverse_phi[WATER], places=12)
        self.assertAlmostEqual(liquid_phi[ETHANOL], 1.1388618715648329, places=12)
        self.assertAlmostEqual(liquid_phi[WATER], 0.6054606414838393, places=12)

    def test_vapor_fugacity_tends_to_ideal_at_zero_pressure(self):
        model = PSRK([ETHANOL, WATER])
        phi = model.fugacity_coefficients(
            400.0,
            1.0e-7,
            {ETHANOL: 0.25, WATER: 0.75},
        )
        for value in phi.values():
            self.assertAlmostEqual(value, 1.0, places=7)

    def test_mathias_copeman_supercritical_branch_uses_only_c1(self):
        model = PSRK([WATER])
        component = model._component_data[0]
        temperature = 1.2 * component.Tc_K
        theta = 1.0 - math.sqrt(1.2)
        expected = (1.0 + component.c1 * theta) ** 2
        untruncated = (
            1.0
            + component.c1 * theta
            + component.c2 * theta**2
            + component.c3 * theta**3
        ) ** 2

        self.assertAlmostEqual(model._alpha(component, temperature), expected, places=14)
        self.assertNotAlmostEqual(expected, untruncated, places=8)

    def test_analytical_temperature_derivatives(self):
        model = PSRK([METHANE, WATER])
        composition = (0.15, 0.85)
        temperature = 350.0
        step = 1.0e-3

        mixture = model._mixture_parameters(temperature, composition)
        numerical_dD_dT = (
            model._mixture_parameters(temperature + step, composition).D
            - model._mixture_parameters(temperature - step, composition).D
        ) / (2.0 * step)

        self.assertAlmostEqual(mixture.dD_dT, numerical_dD_dT, places=9)

    def test_departure_properties_obey_thermodynamic_identities(self):
        model = PSRK([ETHANOL, WATER])
        temperature = 350.0
        pressure = 1.0
        composition = {ETHANOL: 0.5, WATER: 0.5}
        phase = "liquid"

        enthalpy = model.departure_enthalpy(
            temperature, pressure, composition, phase
        )
        entropy = model.departure_entropy(
            temperature, pressure, composition, phase
        )
        gibbs = model.departure_gibbs(
            temperature, pressure, composition, phase
        )
        heat_capacity = model.departure_heat_capacity(
            temperature, pressure, composition, phase
        )
        phi = model.fugacity_coefficients(
            temperature, pressure, composition, phase
        )
        gibbs_from_fugacity = (
            0.1
            * R_BAR_CM3_PER_MOL_K
            * temperature
            * sum(composition[cas] * math.log(phi[cas]) for cas in model.components)
        )

        step = 1.0e-3
        dimensionless_gibbs_upper = sum(
            composition[cas] * math.log(value)
            for cas, value in model.fugacity_coefficients(
                temperature + step, pressure, composition, phase
            ).items()
        )
        dimensionless_gibbs_lower = sum(
            composition[cas] * math.log(value)
            for cas, value in model.fugacity_coefficients(
                temperature - step, pressure, composition, phase
            ).items()
        )
        enthalpy_from_gibbs_derivative = (
            -0.1
            * R_BAR_CM3_PER_MOL_K
            * temperature**2
            * (dimensionless_gibbs_upper - dimensionless_gibbs_lower)
            / (2.0 * step)
        )
        heat_capacity_step = 0.2
        heat_capacity_from_enthalpy = (
            model.departure_enthalpy(
                temperature + heat_capacity_step,
                pressure,
                composition,
                phase,
            )
            - model.departure_enthalpy(
                temperature - heat_capacity_step,
                pressure,
                composition,
                phase,
            )
        ) / (2.0 * heat_capacity_step)

        self.assertAlmostEqual(gibbs, enthalpy - temperature * entropy, places=10)
        self.assertAlmostEqual(gibbs, gibbs_from_fugacity, places=9)
        self.assertAlmostEqual(enthalpy, enthalpy_from_gibbs_derivative, places=5)
        self.assertAlmostEqual(heat_capacity, heat_capacity_from_enthalpy, places=5)

    def test_departure_properties_tend_to_zero_at_zero_pressure(self):
        model = PSRK([ETHANOL, WATER])
        composition = {ETHANOL: 0.25, WATER: 0.75}
        pressure = 1.0e-8

        self.assertAlmostEqual(
            model.departure_enthalpy(400.0, pressure, composition), 0.0, places=4
        )
        self.assertAlmostEqual(
            model.departure_entropy(400.0, pressure, composition), 0.0, places=7
        )
        self.assertAlmostEqual(
            model.departure_gibbs(400.0, pressure, composition), 0.0, places=4
        )

    def test_supercritical_syngas_allows_finite_negative_mixture_D(self):
        model = PSRK([CARBON_MONOXIDE, HYDROGEN])
        composition = {CARBON_MONOXIDE: 0.333, HYDROGEN: 0.667}
        normalized = model._normalize_composition(composition)
        mixture = model._mixture_parameters(323.15, normalized)

        self.assertLess(mixture.D, 0.0)
        phi = model.fugacity_coefficients(323.15, 50.0, composition)
        for value in phi.values():
            self.assertTrue(math.isfinite(value))
            self.assertGreater(value, 0.0)

    def test_estimated_critical_component_is_rejected(self):
        with self.assertRaisesRegex(
            PSRKUnsupportedComponentError,
            "estimated Pc",
        ):
            PSRK(["75-07-0"])  # Acetaldehyde has a starred Pc.

    def test_estimated_critical_component_accepts_resolved_soave_fallback(self):
        model = PSRK(
            ["75-07-0"],
            component_fallbacks={
                "75-07-0": {
                    "Tc_K": 466.0,
                    "Pc_bar": 55.7,
                    "omega": 0.303,
                    "source": "test resolver",
                },
            },
        )
        component = model._component_data[0]
        temperature = 350.0
        theta = 1.0 - math.sqrt(temperature / component.Tc_K)
        coefficient = (
            0.48 + 1.574 * component.omega - 0.176 * component.omega**2
        )

        self.assertFalse(component.uses_mathias_copeman)
        self.assertEqual(model.component_parameter_sources["75-07-0"], "test resolver")
        self.assertAlmostEqual(
            model._alpha(component, temperature),
            (1.0 + coefficient * theta) ** 2,
            places=14,
        )
        self.assertTrue(any("generalized omega-based Soave" in item for item in model.warnings))

    def test_missing_published_bundle_accepts_resolved_fallback(self):
        payload = {
            "components": [{
                "CAS": WATER,
                "name": "Water",
                "Tc_K": None,
                "Pc_kPa": None,
                "omega": None,
                "Tc_estimated": False,
                "Pc_estimated": False,
                "mathias_copeman": None,
                "subgroups": [{"subgroup_id": 16, "count": 1}],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "pure.json")
            path.write_text(json.dumps(payload), encoding="utf-8")
            model = PSRK(
                [WATER],
                pure_data_path=path,
                component_fallbacks={
                    WATER: {
                        "Tc_K": 647.096,
                        "Pc_bar": 220.64,
                        "omega": 0.344,
                    },
                },
            )

        self.assertFalse(model._component_data[0].uses_mathias_copeman)
        self.assertAlmostEqual(model._component_data[0].Tc_K, 647.096)

    def test_component_without_psrk_groups_is_rejected(self):
        with self.assertRaisesRegex(
            PSRKUnsupportedComponentError,
            "no complete published subgroup assignment",
        ):
            PSRK(["75-44-5"])  # Phosgene is pure-SRK-only in the supplement.


if __name__ == "__main__":
    unittest.main()
