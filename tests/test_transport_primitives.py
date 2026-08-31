import math
import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from axial_solver import (
    AxialEvent,
    AxialIntegrationError,
    AxialSolverOptions,
    integrate_axial,
)
from transport_correlations import (
    CircularConduitGeometry,
    TransportCorrelationError,
    acceleration_pressure_gradient,
    beggs_brill_two_phase_flow,
    churchill_friction_factor,
    darcy_friction_factor,
    local_loss_pressure_drop,
    single_phase_flow,
)


class TransportCorrelationTests(unittest.TestCase):
    def test_churchill_covers_laminar_and_turbulent_flow(self):
        self.assertAlmostEqual(churchill_friction_factor(1000.0), 0.064, places=12)
        self.assertAlmostEqual(
            churchill_friction_factor(100000.0, 1e-4),
            0.018462624566280075,
            places=14,
        )

    def test_selectable_turbulent_friction_models_are_consistent(self):
        values = {
            model: darcy_friction_factor(100000.0, 1e-4, model)
            for model in ('churchill', 'haaland', 'swamee-jain', 'colebrook')
        }

        for value in values.values():
            self.assertGreater(value, 0.018)
            self.assertLess(value, 0.019)
        self.assertAlmostEqual(values['colebrook'], 0.018513866077471283, places=14)

    def test_geometry_profiles_support_vertical_tapered_conduits(self):
        geometry = CircularConduitGeometry(
            diameter_m=lambda position: 0.10 - 0.002 * position,
            roughness_m=1e-5,
            elevation_gradient=lambda _position: 1.0,
        )

        self.assertAlmostEqual(geometry.diameter_at(10.0), 0.08)
        self.assertAlmostEqual(geometry.area_at(10.0), math.pi * 0.08**2 / 4.0)
        self.assertEqual(geometry.roughness_at(4.0), 1e-5)
        self.assertEqual(geometry.elevation_gradient_at(4.0), 1.0)

    def test_single_phase_flow_separates_pressure_gradient_terms(self):
        result = single_phase_flow(
            mass_flow_kg_s=1.0,
            density_kg_m3=1000.0,
            viscosity_pa_s=1e-3,
            diameter_m=0.1,
            roughness_m=1e-5,
            elevation_gradient=1.0,
            velocity_gradient_s_inv=0.02,
        )

        self.assertAlmostEqual(result.velocity_m_s, 0.12732395447351627)
        self.assertAlmostEqual(result.reynolds_number, 12732.395447351628)
        self.assertAlmostEqual(result.pressure_gradient.friction, 2.3698471924738502)
        self.assertAlmostEqual(result.pressure_gradient.gravity, 9806.65)
        self.assertAlmostEqual(
            result.pressure_gradient.acceleration,
            acceleration_pressure_gradient(1000.0, result.velocity_m_s, 0.02),
        )
        self.assertAlmostEqual(
            result.pressure_gradient.total,
            result.pressure_gradient.friction
            + result.pressure_gradient.gravity
            + result.pressure_gradient.acceleration,
        )

    def test_beggs_brill_two_phase_flow_reports_holdup_and_regime(self):
        result = beggs_brill_two_phase_flow(
            mass_flow_kg_s=0.0500425,
            mass_quality=0.4,
            liquid_density_kg_m3=958.6368896760326,
            vapor_density_kg_m3=0.5903109235445778,
            liquid_viscosity_pa_s=0.00028275367508684765,
            vapor_viscosity_pa_s=1.2218469398388997e-05,
            surface_tension_n_m=0.05899725063258518,
            pressure_pa=100000.0,
            diameter_m=0.1,
            roughness_m=4.5e-5,
            elevation_gradient=0.0,
        )

        self.assertEqual(result.flow_regime, 'segregated')
        self.assertAlmostEqual(result.liquid_holdup, 0.02567086611894085)
        self.assertAlmostEqual(result.no_slip_liquid_fraction, 0.0009228199632070394)
        self.assertAlmostEqual(result.no_slip_reynolds_number, 51103.18633892556)
        self.assertAlmostEqual(result.darcy_friction_factor, 0.03226191649429374)
        self.assertAlmostEqual(result.pressure_gradient.friction, 4.441590670670646)
        self.assertAlmostEqual(result.pressure_gradient.acceleration, 0.0012221807074174322)

    def test_local_loss_uses_dynamic_pressure(self):
        self.assertAlmostEqual(
            local_loss_pressure_drop(1.5, 998.0, 2.0),
            2994.0,
        )

    def test_invalid_geometry_and_flow_direction_are_rejected(self):
        with self.assertRaises(TransportCorrelationError):
            CircularConduitGeometry(0.1, elevation_gradient=1.1).elevation_gradient_at(0.0)
        with self.assertRaises(TransportCorrelationError):
            single_phase_flow(-1.0, 1000.0, 1e-3, 0.1)


class AxialSolverTests(unittest.TestCase):
    def test_integrates_coupled_state_with_requested_output_grid(self):
        solution = integrate_axial(
            lambda _position, values: (-values[0], 2.0),
            2.0,
            (1.0, 3.0),
            options=AxialSolverOptions(
                relative_tolerance=1e-10,
                absolute_tolerance=(1e-12, 1e-12),
                maximum_step_m=0.1,
            ),
            evaluation_positions_m=(0.5, 1.0, 1.5),
        )

        np.testing.assert_allclose(solution.position_m, (0.0, 0.5, 1.0, 1.5, 2.0))
        self.assertAlmostEqual(solution.outlet_values[0], math.exp(-2.0), places=10)
        self.assertAlmostEqual(solution.outlet_values[1], 7.0, places=11)
        np.testing.assert_allclose(solution.inlet_values, (1.0, 3.0))

    def test_terminal_event_stops_integration(self):
        solution = integrate_axial(
            lambda _position, _values: (-1.0,),
            2.0,
            (1.0,),
            events=(
                AxialEvent(
                    'minimum_pressure',
                    lambda _position, values: values[0] - 0.25,
                    direction=-1.0,
                ),
            ),
            evaluation_positions_m=(0.0, 1.0, 2.0),
        )

        self.assertTrue(solution.terminated_by_event)
        self.assertAlmostEqual(solution.outlet_position_m, 0.75, places=12)
        self.assertAlmostEqual(solution.outlet_values[0], 0.25, places=12)
        self.assertAlmostEqual(
            solution.event_positions_m['minimum_pressure'][0], 0.75, places=12
        )

    def test_bad_rhs_is_reported_with_axial_context(self):
        with self.assertRaisesRegex(AxialIntegrationError, 'shape'):
            integrate_axial(lambda _position, _values: (1.0, 2.0), 1.0, (0.0,))

    def test_geometry_and_pressure_primitives_compose_in_axial_rhs(self):
        geometry = CircularConduitGeometry(
            diameter_m=lambda position: 0.12 - 0.002 * position,
            roughness_m=1e-5,
            elevation_gradient=0.1,
        )

        def pressure_balance(position, _values):
            local = single_phase_flow(
                1.0,
                950.0,
                8e-4,
                geometry.diameter_at(position),
                geometry.roughness_at(position),
                geometry.elevation_gradient_at(position),
            )
            return (-local.pressure_gradient.total / 1e5,)

        solution = integrate_axial(
            pressure_balance,
            10.0,
            (10.0,),
            options=AxialSolverOptions(maximum_step_m=0.25),
        )

        self.assertAlmostEqual(solution.outlet_values[0], 9.906681302916132, places=10)


if __name__ == '__main__':
    unittest.main()
