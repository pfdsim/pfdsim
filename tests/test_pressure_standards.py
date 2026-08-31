import math
import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from chemical_properties import ChemicalDatabase
from pressure_standards import (
    NORMAL_BOILING_PRESSURE_BAR,
    THERMOCHEMICAL_STANDARD_PRESSURE_BAR,
)
from property_resolver import PropertyResolver
from thermodynamics import P_REF, R, create_thermodynamics
from flowsheet_solver import FlowsheetSolver
from unit_operations_basic import Flash


class PressureStandardsTests(unittest.TestCase):
    def test_named_pressure_standards_stay_distinct(self):
        self.assertEqual(THERMOCHEMICAL_STANDARD_PRESSURE_BAR, 1.0)
        self.assertEqual(P_REF, THERMOCHEMICAL_STANDARD_PRESSURE_BAR)
        self.assertAlmostEqual(NORMAL_BOILING_PRESSURE_BAR, 1.01325, places=12)
        self.assertNotEqual(P_REF, NORMAL_BOILING_PRESSURE_BAR)

    def test_entropy_pressure_correction_is_zero_at_one_bar(self):
        thermo = create_thermodynamics(['O2'], 'IDEAL')

        self.assertAlmostEqual(thermo._pressure_entropy_correction(P_REF), 0.0, places=12)
        self.assertAlmostEqual(
            thermo._pressure_entropy_correction(NORMAL_BOILING_PRESSURE_BAR),
            -R * math.log(NORMAL_BOILING_PRESSURE_BAR / THERMOCHEMICAL_STANDARD_PRESSURE_BAR),
            places=12,
        )

    def test_default_ideal_gas_state_uses_one_bar_standard_entropy(self):
        thermo = create_thermodynamics(['O2'], 'IDEAL')
        composition = {'O2': 1.0}

        default_state = thermo.calculate_state(
            298.15,
            P_REF,
            1.0,
            composition,
            phase='vapor',
            flash=False,
            include=('S',),
        )
        normal_boiling_pressure_state = thermo.calculate_state(
            298.15,
            NORMAL_BOILING_PRESSURE_BAR,
            1.0,
            composition,
            phase='vapor',
            flash=False,
            include=('S',),
        )

        self.assertAlmostEqual(default_state.S, thermo.entropy_ideal_gas('O2', 298.15), places=12)
        self.assertAlmostEqual(
            normal_boiling_pressure_state.S - default_state.S,
            -R * math.log(NORMAL_BOILING_PRESSURE_BAR / THERMOCHEMICAL_STANDARD_PRESSURE_BAR),
            places=12,
        )

    def test_canonical_psat_tb_anchor_remains_one_atm(self):
        resolver = PropertyResolver()
        props = ChemicalDatabase(enable_online=False).get(
            'water',
            fetch_online=False,
        )

        self.assertAlmostEqual(
            resolver.resolve_vapor_pressure(
                'water',
                props.Tb,
                props.to_dict(),
                allow_online=False,
            ).value,
            NORMAL_BOILING_PRESSURE_BAR,
            places=10,
        )

    def test_local_tbs_reproduce_one_atm_saturation_pressure(self):
        database = ChemicalDatabase(enable_online=False)

        for identifier in ['water', 'ethanol', 'benzene']:
            with self.subTest(identifier=identifier):
                props = database.get(identifier, fetch_online=False)
                self.assertIsNotNone(props)
                self.assertIsNotNone(props.Tb)
                self.assertAlmostEqual(
                    props.Psat(props.Tb),
                    NORMAL_BOILING_PRESSURE_BAR,
                    delta=0.02,
                )

    def test_pressure_unit_parsing_keeps_atm_as_unit_conversion(self):
        self.assertAlmostEqual(
            FlowsheetSolver._pressure_unit_factor('atm'),
            NORMAL_BOILING_PRESSURE_BAR,
            places=12,
        )

    def test_flash_pressure_grid_contains_one_bar_default_and_one_atm_candidate(self):
        pressure_grid = Flash._pressure_grid(THERMOCHEMICAL_STANDARD_PRESSURE_BAR)

        self.assertIn(THERMOCHEMICAL_STANDARD_PRESSURE_BAR, pressure_grid)
        self.assertIn(NORMAL_BOILING_PRESSURE_BAR, pressure_grid)
