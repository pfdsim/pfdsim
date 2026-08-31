import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chemical_properties import ChemicalDatabase
from compound_identity import classify_identifier
from pfd_parser import PFDParser, ParseError
from property_resolver import PropertyResolver
from simulator import SimulationError, Simulator
from unit_operations_distillation import McCabeThieleDistillation


class PFDComponentPropertyTests(unittest.TestCase):
    maxDiff = None

    def parse(self, text):
        return PFDParser().parse(text)

    def test_top_level_psat_minimum_pressure_parses_and_round_trips(self):
        pfd = self.parse(
            'PROCESS: Deep Vacuum\n'
            'VERSION: 1.0\n'
            'PSAT_MINIMUM_PRESSURE: 1 [Pa]\n'
        )

        self.assertEqual(pfd.metadata.psat_minimum_pressure_bar, 1.0e-5)
        self.assertEqual(
            pfd.to_dict()['metadata']['psat_minimum_pressure_bar'],
            1.0e-5,
        )
        self.assertEqual(
            type(pfd).from_dict(
                pfd.to_dict()
            ).metadata.psat_minimum_pressure_bar,
            1.0e-5,
        )
        serialized = pfd.to_pfd()
        self.assertIn(
            'PSAT_MINIMUM_PRESSURE: 1e-05 [bar]',
            serialized,
        )
        self.assertEqual(
            self.parse(serialized).metadata.psat_minimum_pressure_bar,
            1.0e-5,
        )

        for declaration in (
            'MINIMUM_PRESSURE: 0.001 [kPa]',
            'MINIMUM_PSAT_PRESSURE: 1e-5 bar',
            'MINIMUM_VAPOR_PRESSURE: 0.00001',
        ):
            with self.subTest(declaration=declaration):
                parsed = self.parse(declaration)
                self.assertEqual(
                    parsed.metadata.psat_minimum_pressure_bar,
                    1.0e-5,
                )

        for declaration in (
            'PSAT_MINIMUM_PRESSURE: 0 [bar]',
            'PSAT_MINIMUM_PRESSURE: 1 [not-a-unit]',
        ):
            with self.subTest(declaration=declaration):
                with self.assertRaises(ParseError):
                    self.parse(declaration)

    def test_activity_interaction_limits_parse_and_round_trip(self):
        pfd = self.parse(
            'ACTIVITY_INTERACTION_MAX_PSAT: 2 [MPa]\n'
            'ACTIVITY_INTERACTION_MAX_TEMPERATURE: 150 [C]\n'
        )

        self.assertEqual(pfd.metadata.activity_interaction_max_psat_bar, 20.0)
        self.assertAlmostEqual(
            pfd.metadata.activity_interaction_max_temperature_K,
            423.15,
        )
        restored = type(pfd).from_dict(pfd.to_dict())
        self.assertEqual(restored.metadata.activity_interaction_max_psat_bar, 20.0)
        self.assertAlmostEqual(
            restored.metadata.activity_interaction_max_temperature_K,
            423.15,
        )
        serialized = pfd.to_pfd()
        self.assertIn('ACTIVITY_INTERACTION_MAX_PSAT: 20 [bar]', serialized)
        self.assertIn(
            'ACTIVITY_INTERACTION_MAX_TEMPERATURE: 423.15 [K]',
            serialized,
        )

        for declaration in (
            'ACTIVITY_INTERACTION_MAX_PSAT: 0 [bar]',
            'ACTIVITY_INTERACTION_MAX_TEMPERATURE: -1 [K]',
            'ACTIVITY_INTERACTION_MAX_TEMPERATURE: 300 [rankine]',
        ):
            with self.subTest(declaration=declaration):
                with self.assertRaises(ParseError):
                    self.parse(declaration)

    def test_simulator_configures_activity_limits_after_pfd_snapshot(self):
        from thermodynamics_models.nrtl_uniquac import UNIQUACThermodynamics

        source = '''ONLINE_LOOKUP: false
THERMO_METHOD: UNIQUAC
ACTIVITY_INTERACTION_MAX_PSAT: 10 [bar]
ACTIVITY_INTERACTION_MAX_TEMPERATURE: 350 [K]
COMPONENTS:
    W | Water
    E | Ethanol
'''
        original = UNIQUACThermodynamics.configure_activity_interaction_limits
        observed = []

        def record_configuration(thermo, **options):
            observed.append({
                'options': dict(options),
                'snapshot_ready': all(
                    '_allow_online_lookup' in thermo._resolver_known_props[component]
                    for component in thermo.components
                ),
            })
            return original(thermo, **options)

        with patch.object(
            UNIQUACThermodynamics,
            'configure_activity_interaction_limits',
            record_configuration,
        ):
            thermo = Simulator(PFDParser().parse(source)).initialize().thermo

        active = [
            item for item in observed
            if item['options'].get('max_psat_bar') == 10.0
        ]
        self.assertEqual(len(active), 1)
        self.assertTrue(active[0]['snapshot_ready'])
        self.assertEqual(thermo.activity_interaction_max_psat_bar, 10.0)
        self.assertEqual(thermo.activity_interaction_max_temperature_K, 350.0)

    def test_component_inline_properties_parse_all_supported_fields(self):
        pfd = self.parse(
            'PROCESS: Inline Properties\n'
            'VERSION: 1.0\n'
            '\n'
            'COMPONENTS:\n'
            '    MYST | Mystery liquid | formula=C4H8O, CAS=123-45-6, MW=88.0, '
            'Tc=512.0, Pc=42.0, Vc=250.0, Zc=0.247, omega=0.72, '
            'mc_c1=1.1, mc_c2=-0.2, mc_c3=0.03, '
            'kappa1=0.4, kappa2=-0.5, kappa3=0.6, '
            'twu_l=0.7, twu_m=0.8, twu_n=1.9, twu_c=1e-5, '
            'henry_Hcp=2.5e-4, henry_B=3100.0, henry_Tmin=280.0, '
            'henry_Tmax=330.0, henry_Vinf=42.0, henry_Vinf_uncertainty=1.5, '
            'Tb=350.0, Tt=205.0, Pt=0.004, Tm=210.0, '
            'Hf=-120.0, Gf=-80.0, S=250.0, '
            'Hf_liquid=-150.0, Gf_liquid=-100.0, S_liquid=180.0, '
            'Hf_solid=-160.0, Gf_solid=-110.0, S_solid=90.0, '
            'Hcomb=-2500.0, Hcomb_gross=-2600.0, Hvap=35.0, Hfus=8.0, '
            'Cp_liquid=125.0, Cp_coeffs=[20.0, 0.1, -2e-4, 3e-7], '
            'antoine_A=4.1, antoine_B=1200.0, antoine_C=220.0, '
            'antoine_Tmin=280.0, antoine_Tmax=390.0, '
            'antoine_source="bench, with comma", rho=810.0, rho_T=298.15, '
            'uniquac_r=3.25, uniquac_q=2.75, '
            'UNIFAC={CH3:1,CH2:2,OH:1}, SMILES=CCCO, '
            'phase_at_STP=liquid, critical_properties_unavailable=false\n'
        )

        comp = pfd.components[0]
        self.assertEqual(comp.symbol, 'MYST')
        self.assertEqual(comp.formula, 'C4H8O')
        self.assertEqual(comp.CAS, '123-45-6')
        self.assertAlmostEqual(comp.molecular_weight, 88.0)
        for attr, expected in {
            'Tc': 512.0,
            'Pc': 42.0,
            'Vc': 250.0,
            'Zc': 0.247,
            'omega': 0.72,
            'mc_c1': 1.1,
            'mc_c2': -0.2,
            'mc_c3': 0.03,
            'kappa1': 0.4,
            'kappa2': -0.5,
            'kappa3': 0.6,
            'twu_l': 0.7,
            'twu_m': 0.8,
            'twu_n': 1.9,
            'twu_c': 1e-5,
            'henry_Hcp': 2.5e-4,
            'henry_B': 3100.0,
            'henry_Tmin': 280.0,
            'henry_Tmax': 330.0,
            'henry_Vinf': 42.0,
            'henry_Vinf_uncertainty': 1.5,
            'Tb': 350.0,
            'Tt': 205.0,
            'Pt': 0.004,
            'Tm': 210.0,
            'Hf': -120.0,
            'Gf': -80.0,
            'S': 250.0,
            'Hf_liquid': -150.0,
            'Gf_liquid': -100.0,
            'S_liquid': 180.0,
            'Hf_solid': -160.0,
            'Gf_solid': -110.0,
            'S_solid': 90.0,
            'Hcomb': -2500.0,
            'Hcomb_gross': -2600.0,
            'Hvap': 35.0,
            'Hfus': 8.0,
            'Cp_liquid': 125.0,
            'antoine_A': 4.1,
            'antoine_B': 1200.0,
            'antoine_C': 220.0,
            'antoine_Tmin': 280.0,
            'antoine_Tmax': 390.0,
            'rho': 810.0,
            'rho_T': 298.15,
            'uniquac_r': 3.25,
            'uniquac_q': 2.75,
        }.items():
            self.assertAlmostEqual(getattr(comp, attr), expected)
        self.assertEqual(comp.Cp_coeffs, [20.0, 0.1, -2e-4, 3e-7])
        self.assertEqual(comp.antoine_source, 'bench, with comma')
        self.assertEqual(comp.unifac_groups, {'CH3': 1, 'CH2': 2, 'OH': 1})
        self.assertEqual(comp.smiles, 'CCCO')
        self.assertEqual(comp.phase_at_STP, 'liquid')
        self.assertFalse(comp.critical_properties_unavailable)

    def test_unknown_inline_component_property_is_rejected_with_suggestion(self):
        source = (
            'PROCESS: Misspelled Component Override\n'
            'VERSION: 1.0\n\n'
            'COMPONENTS:\n'
            '    X | Typo test | MW=50, Hvapp=35\n'
        )

        with self.assertRaisesRegex(
            ParseError,
            r"Line 5: .*Unknown component property 'Hvapp'\. Did you mean 'Hvap'\?",
        ):
            self.parse(source)

    def test_malformed_override_entry_is_rejected(self):
        source = (
            'PROCESS: Malformed Component Override\n'
            'VERSION: 1.0\n\n'
            'COMPONENTS:\n'
            '    X | Typo test | MW=50, Hvap 35\n'
        )

        with self.assertRaisesRegex(
            ParseError,
            r"Invalid override entry 'Hvap 35'; expected key=value",
        ):
            self.parse(source)

    def test_inline_comments_preserve_hashes_inside_values(self):
        parsed = self.parse(
            'PROCESS: Inline comments # process note\n'
            'COMPONENTS: # component block\n'
            '    HCN | Hydrogen cyanide | SMILES=C#N, '
            'antoine_source="source #42" # component note\n'
            'STREAM Feed : FEED -> U-1.in # stream note\n'
            '    T = 25 [C] # ambient\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = HCN:1 # pure feed\n'
            'UNIT U-1 # unit note\n'
            '    TYPE: Heater # operation\n'
            '    PORTS:\n'
            '        in : inlet # feed port\n'
            '    PARAMS:\n'
            '        T = 30 [C] # target\n'
        )

        self.assertEqual(parsed.metadata.process_name, 'Inline comments')
        self.assertEqual(parsed.components[0].smiles, 'C#N')
        self.assertEqual(parsed.components[0].antoine_source, 'source #42')
        self.assertEqual(parsed.streams[0].properties[0].value, '25')
        self.assertEqual(parsed.units[0].ports[0].id, 'in')
        self.assertEqual(parsed.units[0].params[0].value, '30')

    def test_parser_reports_all_recoverable_syntax_errors_with_suggestions(self):
        source = (
            'THERMO_METHD: IDEAL\n'
            'THERMO_METHOD: WILSON\n'
            'RECYCLE_METHOD: WEGSTIEN\n'
            'COMPONENTS:\n'
            '    BROKEN\n'
            '    WATR | Water | MW=18\n'
            '    ETH | Ethanol | MW=[46\n'
            'PROPERTY_CORRELATIONS:\n'
            '    WATR.Psta | equation=poly_x, A=1\n'
            'INTERACTION_PARAMETERS:\n'
            '    WATR/ETH | model=NRTLL, alpha=0.3\n'
            'STREAM Feed : FEED -> U1.in\n'
            '    x = WATR:not-a-number\n'
            '    VAPOR_FRACITON = 0.2\n'
            'UNIT U1\n'
            '    TYPE: Flash\n'
            '    PORTS:\n'
            '        in : inelt\n'
            '        malformed port\n'
            '    PARAMS:\n'
            '        temperature 25\n'
            '    REACTIONS:\n'
            '        WATR -> WATR | conversion\n'
            'UNIT Broken extra\n'
            'WHATEVER: x\n'
        )

        with self.assertRaises(ParseError) as caught:
            self.parse(source)

        message = str(caught.exception)
        self.assertIn('PFD parsing failed with 15 errors:', message)
        for expected in (
            "Line 1: Unknown top-level directive 'THERMO_METHD'. "
            "Did you mean 'THERMO_METHOD'?",
            "Line 2: Unsupported thermodynamics method 'WILSON'.",
            'Try NRTL or UNIQUAC',
            'Wilson is a future implementation target.',
            "Line 3: Unknown recycle method 'WEGSTIEN'. Did you mean 'WEGSTEIN'?",
            'Line 5: Invalid COMPONENTS row.',
            "Line 7: Invalid component property for 'ETH': Unterminated collection",
            "Line 9: Unknown PROPERTY_CORRELATIONS property 'Psta'. "
            "Did you mean 'Psat'?",
            "Line 11: Unknown interaction model 'NRTLL'. Did you mean 'NRTL'?",
            "Line 13: Invalid composition for stream 'Feed'",
            "Line 14: Unknown stream property 'VAPOR_FRACITON'. "
            "Did you mean 'VAPOR_FRACTION'?",
            "Line 18: Invalid port in UNIT 'U1': Unknown port type 'inelt'. "
            "Did you mean 'inlet'?",
            "Line 19: Invalid port in UNIT 'U1'",
            "Line 21: Invalid parameter in UNIT 'U1'.",
            "Line 23: Invalid reaction in UNIT 'U1'",
            'Line 24: Invalid UNIT header.',
            "Line 25: Unknown top-level directive 'WHATEVER'.",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, message)

    def test_aggregate_recovery_keeps_valid_rows_after_bad_siblings(self):
        source = (
            'COMPONENTS:\n'
            '    BROKEN\n'
            '    A | First | MW=10\n'
            '    B | | MW=20\n'
            '    C | Third | MW=30\n'
            'PROPERTY_CORRELATIONS:\n'
            '    A.Psta | equation=poly_x, A=1\n'
            '    C.Cpg | equation=poly_x, A=30, Tmin=250, Tmax=800\n'
            'INTERACTION_PARAMETERS:\n'
            '    A/C | model=NRTLL, alpha=0.3\n'
            '    A/C | model=PR, kij=0.1\n'
            'STREAM MissingArrow\n'
            '    ignored = with_the_bad_header\n'
            'STREAM Good : FEED -> PRODUCT\n'
            '    VAPOR_FRACITON = 0.2\n'
            '    T = 25 [C]\n'
            'UNIT U-1\n'
            '    PORTS:\n'
            '        malformed\n'
            '        in : inlet\n'
            '    PARAMS:\n'
            '        malformed\n'
            '        T = 30 [C]\n'
        )
        parser = PFDParser()

        with self.assertRaises(ParseError) as caught:
            parser.parse(source)

        self.assertIn('PFD parsing failed with 8 errors:', str(caught.exception))
        self.assertEqual(
            [component.symbol for component in parser.pfd.components],
            ['A', 'C'],
        )
        self.assertIn('Cpg', parser.pfd.components[1].property_correlations)
        self.assertEqual(len(parser.pfd.interaction_parameters), 1)
        self.assertEqual(parser.pfd.interaction_parameters[0].model, 'PR')
        self.assertEqual([stream.id for stream in parser.pfd.streams], ['Good'])
        self.assertEqual(
            [prop.name for prop in parser.pfd.streams[0].properties],
            ['T'],
        )
        self.assertEqual([port.id for port in parser.pfd.units[0].ports], ['in'])
        self.assertEqual(
            [param.name for param in parser.pfd.units[0].params],
            ['T'],
        )

    def test_unsupported_thermo_diagnostics_distinguish_typos_and_known_methods(self):
        with self.assertRaisesRegex(
            ParseError,
            r"Did you mean 'PENG-ROBINSON' \(available as PR\)\?",
        ):
            self.parse('THERMO_METHOD: PENG-ROBINSN\n')

        with self.assertRaisesRegex(
            ParseError,
            r"NRTL-HOC.*Try NRTL.*future implementation target",
        ):
            self.parse('THERMO_METHOD: NRTL-HOC\n')

        planned_methods = {
            'PRWS': ('Wong-Sandler', 'PSRK'),
            'PR-WONG-SANDLER': ('Wong-Sandler', 'PSRK'),
            'RKSWS': ('Wong-Sandler', 'PSRK'),
            'SRK-WS': ('Wong-Sandler', 'PSRK'),
            'PC-SAFT': ('PC-SAFT', 'PR or SRK'),
            'CPA': ('Cubic-Plus-Association', 'PR or SRK'),
            'LKP': ('Lee-Kesler-Plocker', 'PR or SRK'),
            'LEE-KESLER-PLOCKER': ('Lee-Kesler-Plocker', 'PR or SRK'),
            'IAPWS-95': ('IAPWS-95', 'STEAM'),
            'IAWPS-95': ('IAPWS-95', 'standard formulation name is IAPWS-95'),
            'AMINES': ('Amine and acidic-gas', 'no amine/acid-gas-capable substitute'),
        }
        for method, (planned_name, alternative) in planned_methods.items():
            with self.subTest(method=method):
                with self.assertRaises(ParseError) as caught:
                    self.parse(f'THERMO_METHOD: {method}\n')
                message = str(caught.exception)
                self.assertIn(
                    f"Unsupported thermodynamics method '{method}'",
                    message,
                )
                self.assertIn(planned_name, message)
                self.assertIn(alternative, message)
                self.assertIn('will be implemented in the future', message)

    def test_component_vdm_bundle_aliases_parse_and_round_trip(self):
        source = (
            'PROCESS: VDM Component Overrides\n'
            'VERSION: 1.0\n\n'
            'COMPONENTS:\n'
            '    A | First associator | MW=50, '
            'VDM={delta_H:-52000,delta_S:-130}\n'
            '    B | Second associator | MW=60, '
            'vapor_dimerization={'
            'delta_H_J_per_mol:-47000,delta_S_J_per_mol_K:-115}\n'
        )

        parsed = self.parse(source)
        self.assertEqual(parsed.components[0].vapor_dimerization, {
            'delta_H_J_per_mol': -52000.0,
            'delta_S_J_per_mol_K': -130.0,
        })
        self.assertEqual(parsed.components[1].vapor_dimerization, {
            'delta_H_J_per_mol': -47000.0,
            'delta_S_J_per_mol_K': -115.0,
        })

        reparsed = self.parse(parsed.to_pfd())
        self.assertEqual(
            reparsed.components[0].vapor_dimerization,
            parsed.components[0].vapor_dimerization,
        )
        self.assertIn('VDM={delta_H:-52000, delta_S:-130}', parsed.to_pfd())

    def test_component_vdm_bundle_normalizes_through_dict_construction(self):
        parsed = self.parse(
            'PROCESS: VDM Dictionary Round Trip\n'
            'VERSION: 1.0\n\n'
            'COMPONENTS:\n'
            '    A | Associator | MW=50\n'
        )
        data = parsed.to_dict()
        data['components'][0]['vapor_dimerization'] = {
            'delta_H': -52000,
            'delta_S_J_per_mol_K': -130,
        }

        reconstructed = type(parsed).from_dict(data)

        self.assertEqual(
            reconstructed.components[0].vapor_dimerization,
            {
                'delta_H_J_per_mol': -52000.0,
                'delta_S_J_per_mol_K': -130.0,
            },
        )
        self.assertIn(
            'VDM={delta_H:-52000, delta_S:-130}',
            reconstructed.to_pfd(),
        )

    def test_invalid_component_vdm_bundle_is_rejected(self):
        cases = {
            'not a bundle': ('VDM=-130', r'VDM must be a \{key:value\} parameter bundle'),
            'partial': (
                'VDM={delta_S:-130}',
                r'VDM requires the atomic delta_H and delta_S bundle; missing delta_H_J_per_mol',
            ),
            'unknown': (
                'VDM={delta_H:-52000,delta_SS:-130}',
                r"Unknown VDM parameter 'delta_SS'\. Did you mean 'delta_S'\?",
            ),
            'duplicate alias': (
                'VDM={delta_H:-52000,delta_H_J_per_mol:-51000,delta_S:-130}',
                r"duplicates another alias for 'delta_H_J_per_mol'",
            ),
            'nonfinite': (
                'VDM={delta_H:-52000,delta_S:nan}',
                r"VDM parameter 'delta_S' must be finite",
            ),
        }
        for label, (declaration, message) in cases.items():
            with self.subTest(label=label):
                source = (
                    'PROCESS: Invalid VDM Component Override\n'
                    'VERSION: 1.0\n\n'
                    'COMPONENTS:\n'
                    f'    A | Associator | MW=50, {declaration}\n'
                )
                with self.assertRaisesRegex(ParseError, message):
                    self.parse(source)

    def test_incomplete_or_invalid_inline_antoine_override_fails_during_parse(self):
        cases = {
            'missing range field': (
                'antoine_A=4.1, antoine_B=1200, antoine_C=220, '
                'antoine_Tmin=280',
                'atomic A, B, C, Tmin, Tmax bundle',
            ),
            'nonpositive B': (
                'antoine_A=4.1, antoine_B=0, antoine_C=220, '
                'antoine_Tmin=280, antoine_Tmax=390',
                'Antoine B must be positive',
            ),
            'invalid range': (
                'antoine_A=4.1, antoine_B=1200, antoine_C=220, '
                'antoine_Tmin=390, antoine_Tmax=280',
                '0 < antoine_Tmin < antoine_Tmax',
            ),
            'singular range': (
                'antoine_A=4.1, antoine_B=1200, antoine_C=0, '
                'antoine_Tmin=250, antoine_Tmax=300',
                'denominator is singular',
            ),
            'nonfinite coefficient': (
                'antoine_A=nan, antoine_B=1200, antoine_C=220, '
                'antoine_Tmin=280, antoine_Tmax=390',
                'Antoine values must be finite',
            ),
        }
        for label, (properties, message) in cases.items():
            with self.subTest(label=label):
                source = (
                    'PROCESS: Invalid Antoine\n'
                    'VERSION: 1.0\n\n'
                    'COMPONENTS:\n'
                    f'    X | Invalid Antoine | {properties}\n'
                )
                with self.assertRaisesRegex(ParseError, message):
                    self.parse(source)

    def test_property_correlations_block_parses_and_round_trips(self):
        source = (
            'PROCESS: Correlations\n'
            'VERSION: 1.0\n'
            '\n'
            'COMPONENTS:\n'
            '    MYST | Mystery liquid | MW=88.0, Tb=350.0, Tt=205.0, '
            'Pt=0.004, Cp_coeffs=[1, 2, 3, 4]\n'
            '\n'
            'PROPERTY_CORRELATIONS:\n'
            '    MYST.Hvap | equation=poly_x, Tmin_K=300.0, Tmax_K=400.0, A=40.0, B=-5.0, source="bench, data"\n'
            '    MYST.Cpl | equation=poly_x, Tmin=250.0, Tmax=500.0, coefficients={A:80.0,B:4.0}\n'
            '    MYST.rhol | equation=poly_x, Tmin_K=250.0, Tmax_K=350.0, A=620.0, B=-80.0\n'
            '    MYST.mug | equation=viscosity_exp_rhor, A=2.0, E=1.0, x=0.5, y=0.25\n'
        )
        pfd = self.parse(source)

        comp = pfd.components[0]
        self.assertEqual(comp.Cp_coeffs, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(comp.property_correlations['Hvap']['source'], 'bench, data')
        self.assertEqual(
            comp.property_correlations['Hvap']['coefficients'],
            {'A': 40.0, 'B': -5.0},
        )
        self.assertEqual(comp.property_correlations['Cpl']['Tmin_K'], 250.0)
        self.assertEqual(comp.property_correlations['Cpl']['coefficients']['B'], 4.0)
        self.assertEqual(comp.property_correlations['rhol']['coefficients']['A'], 620.0)
        self.assertEqual(
            comp.property_correlations['mug']['coefficients'],
            {'A': 2.0, 'E': 1.0, 'x': 0.5, 'y': 0.25},
        )

        serialized = pfd.to_pfd()
        reparsed = self.parse(serialized)
        round_trip = reparsed.components[0]
        self.assertEqual(round_trip.Tb, 350.0)
        self.assertEqual(round_trip.Tt, 205.0)
        self.assertEqual(round_trip.Pt, 0.004)
        self.assertEqual(round_trip.Cp_coeffs, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(
            round_trip.property_correlations['Hvap']['coefficients'],
            {'A': 40.0, 'B': -5.0},
        )
        self.assertEqual(
            round_trip.property_correlations['mug']['coefficients'],
            {'A': 2.0, 'E': 1.0, 'x': 0.5, 'y': 0.25},
        )

    def test_unknown_property_correlation_names_are_rejected_with_suggestions(self):
        cases = {
            'property': (
                '    X.Psaat | equation=poly_x, A=1\n',
                r"Unknown PROPERTY_CORRELATIONS property 'Psaat'\. Did you mean 'Psat'\?",
            ),
            'field': (
                '    X.Cpl | equation=poly_x, Tmix_K=250, A=80\n',
                r"Unknown Cpl correlation field 'Tmix_K'\. Did you mean 'Tmin_K'\?",
            ),
            'equation': (
                '    X.Cpl | equation=pol_x, A=80\n',
                r"Unknown Cpl correlation equation 'pol_x'\. Did you mean 'poly_x'\?",
            ),
            'nested coefficient': (
                '    X.Cpl | equation=poly_x, coefficients={AA:80}\n',
                r"Unknown Cpl poly_x coefficient 'AA'\. Did you mean 'A'\?",
            ),
        }
        for label, (correlation_line, message) in cases.items():
            with self.subTest(label=label):
                source = (
                    'PROCESS: Misspelled Correlation Override\n'
                    'VERSION: 1.0\n\n'
                    'COMPONENTS:\n'
                    '    X | Typo test | MW=50\n\n'
                    'PROPERTY_CORRELATIONS:\n'
                    f'{correlation_line}'
                )
                with self.assertRaisesRegex(ParseError, message):
                    self.parse(source)

    def test_misspelled_psat_equation_preserves_specific_error_and_suggests(self):
        source = (
            'PROCESS: Misspelled Psat Equation\n'
            'VERSION: 1.0\n\n'
            'COMPONENTS:\n'
            '    X | Typo test | MW=50, Tc=500, Pc=40\n\n'
            'PROPERTY_CORRELATIONS:\n'
            '    X.Psat | equation=pol_x, A=1\n'
        )

        with self.assertRaisesRegex(
            ParseError,
            r"Psat equation 'pol_x' is unsupported\. Did you mean 'poly_x'\?",
        ):
            self.parse(source)

    def test_canonical_psat_af_parses_round_trips_and_feeds_resolver(self):
        source = (
            'PROCESS: Canonical Psat Override\n'
            'VERSION: 1.0\n'
            '\n'
            'COMPONENTS:\n'
            '    XAF | Canonical test | MW=50.0, Tc=500.0, Pc=20.0855369232, Tb=350.0\n'
            '\n'
            'PROPERTY_CORRELATIONS:\n'
            '    XAF.Psat | equation=canonical_psat, Tmin_K=200.0, Tmax_K=500.0, '
            'Tc=500.0, Pc_bar=20.0855369232, Tb=350.0, '
            'A=5.0, B=-1000.0, C=0.0, D=0.0, E=0.0, F=0.0\n'
        )
        pfd = self.parse(source)
        correlation = pfd.components[0].property_correlations['Psat']

        self.assertEqual(correlation['equation'], 'canonical_psat')
        self.assertEqual(correlation['Tc_K'], 500.0)
        self.assertEqual(correlation['Pc_bar'], 20.0855369232)
        self.assertEqual(correlation['Tb_K'], 350.0)
        self.assertEqual(
            correlation['coefficients'],
            {
                'A': 5.0,
                'B': -1000.0,
                'C': 0.0,
                'D': 0.0,
                'E': 0.0,
                'F': 0.0,
            },
        )

        round_trip = self.parse(pfd.to_pfd()).components[0].property_correlations['Psat']
        self.assertEqual(round_trip, correlation)

        props = {
            'Tc': 500.0,
            'Pc': 20.0855369232,
            'property_correlations': {
                'Psat': {**correlation, 'quality': 1.0, '_pfd_override': True},
            },
        }
        result = PropertyResolver().resolve_vapor_pressure(
            'XAF',
            400.0,
            props,
            allow_online=False,
        )
        self.assertEqual(result.method, 'pfd_psat_canonical_psat')
        self.assertEqual(result.quality, 1.0)
        self.assertAlmostEqual(result.value, math.exp(2.5), delta=0.02)

    def test_canonical_psat_extended_forms_parse_and_round_trip(self):
        source = (
            'PROCESS: Extended Canonical Psat\n'
            'VERSION: 1.0\n'
            '\n'
            'COMPONENTS:\n'
            '    XAG | A-G test | MW=50.0, Tc=500.0, Pc=20.0\n'
            '    XAH | A-H test | MW=50.0, Tc=500.0, Pc=20.0\n'
            '\n'
            'PROPERTY_CORRELATIONS:\n'
            '    XAG.Psat | equation=canonical_psat_ag, A=1, B=2, C=3, D=4, E=5, F=6, G=7\n'
            '    XAH.Psat | equation=canonical_psat_ah, A=1, B=2, C=3, D=4, E=5, F=6, G=7, H=8, inverse_power=-5\n'
        )
        pfd = self.parse(source)
        round_trip = self.parse(pfd.to_pfd())

        correlations = {
            component.symbol: component.property_correlations['Psat']
            for component in round_trip.components
        }
        self.assertEqual(correlations['XAG']['equation'], 'canonical_psat_ag')
        self.assertEqual(correlations['XAG']['coefficients']['G'], 7.0)
        self.assertEqual(correlations['XAH']['equation'], 'canonical_psat_ah')
        self.assertEqual(correlations['XAH']['coefficients']['H'], 8.0)
        self.assertEqual(correlations['XAH']['inverse_power'], -5.0)

    def test_invalid_psat_schema_fails_during_parse(self):
        cases = {
            'missing equation': ('A=1', 'requires equation'),
            'unsupported equation': (
                'equation=unknown_psat, A=1',
                'is unsupported',
            ),
            'missing polynomial coefficient': (
                'equation=poly_x, B=1',
                'requires coefficient.*A',
            ),
            'missing reduced coefficient': (
                'equation=reduced_vapor_pressure, A=1, B=2, C=3',
                'requires coefficient.*D',
            ),
            'missing canonical coefficient': (
                'equation=canonical_psat, A=1, B=2, C=3, D=4, E=5',
                'requires coefficient.*F',
            ),
            'unexpected coefficient': (
                'equation=reduced_vapor_pressure, A=1, B=2, C=3, D=4, E=5',
                'does not accept coefficient.*E',
            ),
            'missing inverse power': (
                'equation=canonical_psat_ah, A=1, B=2, C=3, D=4, E=5, '
                'F=6, G=7, H=8',
                'requires inverse_power',
            ),
            'invalid inverse power': (
                'equation=canonical_psat_ah, A=1, B=2, C=3, D=4, E=5, '
                'F=6, G=7, H=8, inverse_power=-4',
                'must be -3, -5, or -7',
            ),
            'inverse power on A-F': (
                'equation=canonical_psat, A=1, B=2, C=3, D=4, E=5, F=6, '
                'inverse_power=-3',
                'cannot declare inverse_power',
            ),
            'invalid temperature range': (
                'equation=poly_x, A=1, Tmin_K=400, Tmax_K=300',
                'requires Tmin_K < Tmax_K',
            ),
            'invalid pressure range': (
                'equation=poly_x, A=1, Pmin_bar=2, Pmax_bar=1',
                'requires Pmin_bar <= Pmax_bar',
            ),
            'negative pressure at declared endpoint': (
                'equation=poly_x, Tmin_K=290, Tmax_K=395, A=0.03, B=1.33',
                'must remain positive and finite throughout',
            ),
            'nonpositive pressure inside declared range': (
                'equation=poly_x, Tmin_K=298.15, Tmax_K=398.15, '
                'A=1, B=-4, C=4',
                'must remain positive and finite throughout',
            ),
            'nonfinite coefficient': (
                'equation=poly_x, A=nan',
                'coefficient A must be finite',
            ),
            'invalid quality': (
                'equation=poly_x, A=1, quality=1.1',
                'quality must be between 0 and 1',
            ),
            'conflicting critical pressures': (
                'equation=poly_x, A=1, Pc_bar=20, Pc_Pa=3000000',
                'Pc_bar and Pc_Pa conflict',
            ),
        }
        for label, (correlation, message) in cases.items():
            with self.subTest(label=label):
                source = (
                    'PROCESS: Invalid Psat\n'
                    'VERSION: 1.0\n\n'
                    'COMPONENTS:\n'
                    '    X | Invalid Psat | Tc=500, Pc=40\n\n'
                    'PROPERTY_CORRELATIONS:\n'
                    f'    X.Psat | {correlation}\n'
                )
                with self.assertRaisesRegex(ParseError, message):
                    self.parse(source)

    def test_interaction_parameters_block_parses_and_round_trips(self):
        source = (
            'PROCESS: Interaction Overrides\n'
            'VERSION: 1.0\n'
            '\n'
            'COMPONENTS:\n'
            '    E | Ethanol | MW=46.07\n'
            '    W | Water | MW=18.015\n'
            '\n'
            'INTERACTION_PARAMETERS:\n'
            '    W/E | model=NRTL, alpha=0.2, tau12_c=7.0, tau12_f=0.001, '
            'tau21_c=3.0, tau21_f=-0.002\n'
            '    E/W | model=PR, kij=0.125, Tmin_K=290.0, Tmax_K=330.0\n'
        )
        pfd = self.parse(source)

        self.assertEqual(len(pfd.interaction_parameters), 2)
        nrtl = pfd.interaction_parameters[0]
        self.assertEqual(nrtl.component1, 'W')
        self.assertEqual(nrtl.component2, 'E')
        self.assertEqual(nrtl.model, 'NRTL')
        self.assertAlmostEqual(nrtl.parameters['alpha'], 0.2)
        self.assertAlmostEqual(nrtl.parameters['tau12_c'], 7.0)
        self.assertAlmostEqual(nrtl.parameters['tau12_f'], 0.001)

        serialized = pfd.to_pfd()
        reparsed = self.parse(serialized)
        self.assertEqual(len(reparsed.interaction_parameters), 2)
        pr = reparsed.interaction_parameters[1]
        self.assertEqual(pr.model, 'PR')
        self.assertAlmostEqual(pr.parameters['kij'], 0.125)
        self.assertAlmostEqual(pr.parameters['Tmax_K'], 330.0)

    def test_interaction_estimation_block_parses_pair_overrides_and_round_trips(self):
        source = (
            'PROCESS: Estimated Interactions\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: NRTL\n\n'
            'COMPONENTS:\n'
            '    W | Water\n'
            '    K | Acetone\n\n'
            'INTERACTION_ESTIMATION:\n'
            '    NRTL | source=UNIFDMD, policy=missing_only, '
            'parameter_order=source, alpha=0.3, Tmin=20 [C], Tmax=150 [C]\n'
            '    W/K | model=NRTL, alpha=0.2, Tmin=30 [C], Tmax=120 [C]\n'
        )
        pfd = self.parse(source)
        self.assertEqual(len(pfd.interaction_estimation), 2)
        global_rule, pair_rule = pfd.interaction_estimation
        self.assertEqual(global_rule.model, 'NRTL')
        self.assertIsNone(global_rule.component1)
        self.assertEqual(global_rule.parameters['source'], 'UNIFDMD')
        self.assertEqual(global_rule.parameters['Tmin'], '20 [C]')
        self.assertEqual(pair_rule.component1, 'W')
        self.assertEqual(pair_rule.component2, 'K')
        self.assertAlmostEqual(pair_rule.parameters['alpha'], 0.2)

        reparsed = self.parse(pfd.to_pfd())
        self.assertEqual(
            reparsed.to_dict()['interaction_estimation'],
            pfd.to_dict()['interaction_estimation'],
        )

    def test_interaction_estimation_rejects_unknown_fields_and_pair_components(self):
        base = (
            'THERMO_METHOD: UNIQUAC\n'
            'COMPONENTS:\n'
            '    W | Water\n'
            '    K | Acetone\n'
            'INTERACTION_ESTIMATION:\n'
        )
        with self.assertRaisesRegex(Exception, "Unknown UNIQUAC INTERACTION_ESTIMATION field 'polciy'"):
            self.parse(
                base
                + '    UNIQUAC | source=UNIFAC, polciy=missing_only, '
                'Tmin=290, Tmax=400\n'
            )
        with self.assertRaisesRegex(Exception, "Unknown INTERACTION_ESTIMATION component 'X'"):
            self.parse(
                base
                + '    UNIQUAC | source=UNIFAC, Tmin=290, Tmax=400\n'
                + '    W/X | model=UNIQUAC, Tmin=300, Tmax=350\n'
            )

    def test_unknown_interaction_fields_are_rejected_with_suggestions(self):
        cases = {
            'NRTL': ('alhpa=0.3, a12=1, a21=2', 'alhpa', 'alpha'),
            'UNIQUAC': ('use_q_prme=true, a12=1, a21=2', 'use_q_prme', 'use_q_prime'),
            'PR': ('kji=0.1', 'kji', 'kij'),
            'LIQUID_VISCOSITY': (
                'form=grunberg_nissan, G12x=0.5',
                'G12x',
                'g12',
            ),
        }
        for model, (parameters, misspelled, suggestion) in cases.items():
            with self.subTest(model=model):
                source = (
                    'PROCESS: Misspelled Interaction Override\n'
                    'VERSION: 1.0\n\n'
                    'COMPONENTS:\n'
                    '    A | First | MW=10\n'
                    '    B | Second | MW=20\n\n'
                    'INTERACTION_PARAMETERS:\n'
                    f'    A/B | model={model}, {parameters}\n'
                )
                with self.assertRaisesRegex(
                    ParseError,
                    rf"Unknown {model} INTERACTION_PARAMETERS field '{misspelled}'\. "
                    rf"Did you mean '{suggestion}'\?",
                ):
                    self.parse(source)

    def test_vdm_cross_residual_aliases_parse_and_round_trip(self):
        source = (
            'PROCESS: VDM Cross Override\n'
            'VERSION: 1.0\n\n'
            'COMPONENTS:\n'
            '    A | First | MW=10\n'
            '    B | Second | MW=20\n\n'
            'INTERACTION_PARAMETERS:\n'
            '    A/B | model=VAPOR_DIMERIZATION, delta_H_residual=1200, '
            'delta_S_residual=-3.5\n'
        )

        parsed = self.parse(source)
        interaction = parsed.interaction_parameters[0]
        self.assertEqual(interaction.model, 'VAPOR_DIMERIZATION')
        self.assertEqual(interaction.parameters['delta_H_residual'], 1200.0)
        self.assertEqual(interaction.parameters['delta_S_residual'], -3.5)

        reparsed = self.parse(parsed.to_pfd())
        self.assertEqual(
            reparsed.interaction_parameters[0].parameters,
            interaction.parameters,
        )

    def test_invalid_vdm_cross_residual_bundle_is_rejected(self):
        cases = {
            'partial': (
                'delta_H_residual=1200',
                r'atomic delta_H_residual and delta_S_residual bundle; missing '
                r'delta_S_residual_J_per_mol_K',
            ),
            'unknown': (
                'delta_H_residual=1200, delta_S_residul=-3.5',
                r"Unknown VDM INTERACTION_PARAMETERS field 'delta_S_residul'\. "
                r"Did you mean 'delta_s_residual'\?",
            ),
            'duplicate alias': (
                'delta_H_residual=1200, delta_H_residual_J_per_mol=1100, '
                'delta_S_residual=-3.5',
                r"duplicates another alias for 'delta_H_residual_J_per_mol'",
            ),
        }
        for label, (parameters, message) in cases.items():
            with self.subTest(label=label):
                source = (
                    'PROCESS: Invalid VDM Cross Override\n'
                    'VERSION: 1.0\n\n'
                    'COMPONENTS:\n'
                    '    A | First | MW=10\n'
                    '    B | Second | MW=20\n\n'
                    'INTERACTION_PARAMETERS:\n'
                    f'    A/B | model=VDM, {parameters}\n'
                )
                with self.assertRaisesRegex(ParseError, message):
                    self.parse(source)

    def test_pfd_nrtl_interaction_override_wins_and_orients_by_component_pair(self):
        pfd = (
            'PROCESS: NRTL Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: NRTL\n'
            'ONLINE_LOOKUP: false\n'
            '\n'
            'COMPONENTS:\n'
            '    E | Ethanol | MW=46.068, CAS=64-17-5\n'
            '    W | Water | MW=18.015, CAS=7732-18-5\n'
            '\n'
            'INTERACTION_PARAMETERS:\n'
            '    W/E | model=NRTL, alpha=0.2, tau12_c=7.0, tau21_c=3.0, tau12_d=70.0, tau21_d=30.0\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = E:0.5, W:0.5\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        self.assertTrue(result.converged)
        tau, alpha = sim.thermo._nrtl_matrices(300.0)
        self.assertAlmostEqual(tau[0][1], 3.0 + 30.0 / 300.0)
        self.assertAlmostEqual(tau[1][0], 7.0 + 70.0 / 300.0)
        self.assertAlmostEqual(alpha[0][1], 0.2)
        self.assertFalse(any('NRTL binary interaction parameters missing' in warning for warning in sim.thermo.warnings))

    def test_pfd_uniquac_interaction_override_wins_and_feeds_q_prime_flag(self):
        pfd = (
            'PROCESS: UNIQUAC Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: UNIQUAC\n'
            'ONLINE_LOOKUP: false\n'
            '\n'
            'COMPONENTS:\n'
            '    E | Ethanol | MW=46.068, CAS=64-17-5\n'
            '    W | Water | MW=18.015, CAS=7732-18-5\n'
            '\n'
            'INTERACTION_PARAMETERS:\n'
            '    E/W | model=UNIQUAC, tau12_a=0.25, tau21_a=-0.5, tau12_b=30.0, tau21_b=-60.0, use_q_prime=true\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = E:0.5, W:0.5\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        self.assertTrue(result.converged)
        tau = sim.thermo._uniquac_tau_matrix(300.0)
        params = sim.thermo._uniquac_parameter_matrices()
        self.assertAlmostEqual(tau[0][1], math.exp(0.25 + 30.0 / 300.0))
        self.assertAlmostEqual(tau[1][0], math.exp(-0.5 - 60.0 / 300.0))
        self.assertTrue(params['use_q_prime'][0][1])
        self.assertFalse(any('UNIQUAC binary interaction parameters missing' in warning for warning in sim.thermo.warnings))

    def test_pfd_uniquac_quoted_false_q_prime_stays_false(self):
        pfd = (
            'PROCESS: UNIQUAC Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: UNIQUAC\n'
            'ONLINE_LOOKUP: false\n'
            '\n'
            'COMPONENTS:\n'
            '    E | Ethanol | MW=46.068, CAS=64-17-5\n'
            '    W | Water | MW=18.015, CAS=7732-18-5\n'
            '\n'
            'INTERACTION_PARAMETERS:\n'
            '    E/W | model=UNIQUAC, tau12_a=0.25, tau21_a=-0.5, use_q_prime="false"\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = E:0.5, W:0.5\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        self.assertTrue(result.converged)
        params = sim.thermo._uniquac_parameter_matrices()
        self.assertFalse(params['use_q_prime'][0][1])

    def test_pfd_pr_ranged_kij_override_applies_only_inside_temperature_range(self):
        pfd = (
            'PROCESS: PR Ranged Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: PR\n'
            'ONLINE_LOOKUP: false\n'
            '\n'
            'COMPONENTS:\n'
            '    CO2 | Carbon dioxide | CAS=124-38-9\n'
            '    H2S | Hydrogen sulfide | CAS=7783-06-4\n'
            '\n'
            'INTERACTION_PARAMETERS:\n'
            '    CO2/H2S | model=PR, kij=0.5, Tmin_K=290.0, Tmax_K=310.0\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 10 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = CO2:0.5, H2S:0.5\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        self.assertTrue(result.converged)
        self.assertAlmostEqual(sim.thermo.cubic._kij('CO2', 'H2S', 300.0), 0.5)
        self.assertAlmostEqual(sim.thermo.cubic._kij('CO2', 'H2S', 400.0), 0.09835, places=6)

    def test_pfd_pr_static_temperature_dependent_kij_override_wins_everywhere(self):
        pfd = (
            'PROCESS: PR Static Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: PR\n'
            'ONLINE_LOOKUP: false\n'
            '\n'
            'COMPONENTS:\n'
            '    CO2 | Carbon dioxide | CAS=124-38-9\n'
            '    H2S | Hydrogen sulfide | CAS=7783-06-4\n'
            '\n'
            'INTERACTION_PARAMETERS:\n'
            '    CO2/H2S | model=PR, kij_a=0.1, kij_b=30.0, kij_c=0.001\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 10 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = CO2:0.5, H2S:0.5\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        self.assertTrue(result.converged)
        self.assertAlmostEqual(sim.thermo.cubic._kij('CO2', 'H2S', 300.0), 0.1 + 30.0 / 300.0 + 0.001 * 300.0)
        self.assertAlmostEqual(sim.thermo.cubic._kij('CO2', 'H2S', 400.0), 0.1 + 30.0 / 400.0 + 0.001 * 400.0)

    def test_pfd_liquid_viscosity_interaction_override_selects_temperature_range(self):
        pfd = (
            'PROCESS: Viscosity Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: IDEAL\n'
            'ONLINE_LOOKUP: false\n'
            '\n'
            'COMPONENTS:\n'
            '    E | Ethanol | MW=46.068, CAS=64-17-5\n'
            '    W | Water | MW=18.015, CAS=7732-18-5\n'
            '\n'
            'INTERACTION_PARAMETERS:\n'
            '    E/W | model=LIQUID_VISCOSITY, form=grunberg_nissan, G=0.5, Tmin_K=290.0, Tmax_K=310.0\n'
            '    E/W | model=LIQUID_VISCOSITY, form=grunberg_nissan, G=1.0, Tmin_K=320.0, Tmax_K=340.0\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = E:0.5, W:0.5\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        self.assertTrue(result.converged)

        for T, G in ((298.15, 0.5), (330.0, 1.0)):
            pure_e = sim.thermo._pure_viscosity('E', T, 1.0, 'liquid')
            pure_w = sim.thermo._pure_viscosity('W', T, 1.0, 'liquid')
            ideal = math.exp(0.5 * math.log(pure_e) + 0.5 * math.log(pure_w))
            viscosity = sim.thermo.mixture_viscosity(
                {'E': 0.5, 'W': 0.5},
                T,
                1.0,
                vapor_fraction=0.0,
            )
            self.assertAlmostEqual(viscosity, ideal * math.exp(0.25 * G), places=15)

    def test_pfd_combined_nrtl_pr_interaction_overrides_both_activity_and_eos_layers(self):
        pfd = (
            'PROCESS: NRTL PR Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: NRTL-PR\n'
            'ONLINE_LOOKUP: false\n'
            '\n'
            'COMPONENTS:\n'
            '    E | Ethanol | MW=46.068, CAS=64-17-5\n'
            '    W | Water | MW=18.015, CAS=7732-18-5\n'
            '\n'
            'INTERACTION_PARAMETERS:\n'
            '    E/W | model=NRTL, alpha=0.25, tau12_c=1.2, tau21_c=-0.4\n'
            '    E/W | model=PR, kij=-0.078\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = E:0.5, W:0.5\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        self.assertTrue(result.converged)
        tau, alpha = sim.thermo._nrtl_matrices(300.0)
        self.assertAlmostEqual(tau[0][1], 1.2)
        self.assertAlmostEqual(tau[1][0], -0.4)
        self.assertAlmostEqual(alpha[0][1], 0.25)
        self.assertAlmostEqual(sim.thermo.rk._kij('E', 'W', 300.0), -0.078)

    def test_pfd_activity_duplicate_interaction_override_is_rejected(self):
        pfd = (
            'PROCESS: Duplicate Activity Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: NRTL\n'
            'ONLINE_LOOKUP: false\n'
            '\n'
            'COMPONENTS:\n'
            '    E | Ethanol | MW=46.068, CAS=64-17-5\n'
            '    W | Water | MW=18.015, CAS=7732-18-5\n'
            '\n'
            'INTERACTION_PARAMETERS:\n'
            '    E/W | model=NRTL, alpha=0.2, tau12_c=1.0, tau21_c=2.0\n'
            '    W/E | model=NRTL, alpha=0.3, tau12_c=3.0, tau21_c=4.0\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = E:0.5, W:0.5\n'
        )

        sim = Simulator.from_string(pfd)
        with self.assertRaisesRegex(Exception, 'Duplicate NRTL INTERACTION_PARAMETERS'):
            sim.run()

    def test_pfd_eos_overlapping_temperature_override_is_rejected(self):
        pfd = (
            'PROCESS: Duplicate EOS Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: PR\n'
            'ONLINE_LOOKUP: false\n'
            '\n'
            'COMPONENTS:\n'
            '    CO2 | Carbon dioxide | CAS=124-38-9\n'
            '    H2S | Hydrogen sulfide | CAS=7783-06-4\n'
            '\n'
            'INTERACTION_PARAMETERS:\n'
            '    CO2/H2S | model=PR, kij=0.1, Tmin_K=290.0, Tmax_K=320.0\n'
            '    CO2/H2S | model=PR, kij=0.2, Tmin_K=310.0, Tmax_K=330.0\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 10 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = CO2:0.5, H2S:0.5\n'
        )

        sim = Simulator.from_string(pfd)
        with self.assertRaisesRegex(Exception, 'Overlapping PR INTERACTION_PARAMETERS'):
            sim.run()

    def test_pfd_overrides_apply_to_resolver_snapshot_after_lookup(self):
        pfd = (
            'PROCESS: Override Snapshot\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: IDEAL\n'
            '\n'
            'COMPONENTS:\n'
            '    XHV | X Hvap test | MW=50.0, Tc=600.0, Pc=30.0, Tb=350.0, Hvap=40.0, '
            'Cp_liquid=100.0, antoine_A=4.0, antoine_B=1000.0, antoine_C=200.0, '
            'antoine_Tmin=250.0, antoine_Tmax=450.0\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = XHV:1.0\n'
        )

        with patch('chemical_properties.OnlinePropertyFetcher.fetch_from_pubchem', return_value=None):
            sim = Simulator.from_string(pfd)
            result = sim.run()

        self.assertTrue(result.converged)
        known = sim.thermo._resolver_known_props['XHV']
        self.assertEqual(known['Hvap'], 40.0)
        self.assertEqual(known['Cp_liquid'], 100.0)
        resolver = PropertyResolver()
        hvap = resolver.resolve_hvap('XHV', known, T=300.0, allow_online=False, allow_estimation=False)
        expected = 40.0 * ((1.0 - 300.0 / 600.0) / (1.0 - 350.0 / 600.0)) ** 0.38
        self.assertEqual(hvap.method, 'watson_hvap')
        self.assertAlmostEqual(hvap.value, expected, places=10)
        self.assertAlmostEqual(sim.thermo.Hvap_at_T('XHV', 300.0), expected, places=10)

    def test_online_lookup_false_blocks_runtime_resolver_online_paths(self):
        pfd = (
            'PROCESS: No Online Runtime\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: IDEAL\n'
            'ONLINE_LOOKUP: false\n'
            'PSAT_MINIMUM_PRESSURE: 1 [Pa]\n'
            '\n'
            'COMPONENTS:\n'
            '    XNO | No online component | MW=72.0, Tc=560.0, Pc=35.0, '
            'omega=0.25, Tb=355.0, Hvap=36.0, Cp_liquid=120.0, '
            'Cp_coeffs=[30.0, 0.01, 0.0, 0.0]\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = XNO:1.0\n'
        )

        with patch.object(PropertyResolver, '_prime_nist_antoine_cache', side_effect=AssertionError):
            with patch.object(PropertyResolver, 'get_antoine_online', side_effect=AssertionError):
                with patch.object(PropertyResolver, 'get_hvap_online', side_effect=AssertionError):
                    with patch.object(PropertyResolver, '_fetch_cp_online', side_effect=AssertionError):
                        sim = Simulator.from_string(pfd)
                        result = sim.run()
                        self.assertTrue(result.converged)
                        self.assertFalse(sim.thermo._resolver_known_props['XNO']['_allow_online_lookup'])
                        self.assertEqual(
                            sim.thermo._resolver_known_props['XNO'][
                                '_psat_minimum_pressure_bar'
                            ],
                            1.0e-5,
                        )
                        self.assertGreater(sim.thermo.Psat('XNO', 300.0), 0.0)
                        self.assertGreater(sim.thermo.Cp_ideal_gas('XNO', 300.0), 0.0)

    def test_pfd_correlation_override_marker_is_per_correlation_key(self):
        props = {
            'property_correlations': {
                'rhol': {'equation': 'poly_x', 'coefficients': {'A': 800.0}},
                'Hvap': {
                    'equation': 'poly_x',
                    'coefficients': {'A': 35.0},
                    '_pfd_override': True,
                },
            },
            'property_sources': {
                'property_correlations': {
                    'method': 'pfd_property_correlations',
                },
            },
        }

        self.assertTrue(PropertyResolver._is_pfd_correlation_override(props, 'Hvap'))
        self.assertFalse(PropertyResolver._is_pfd_correlation_override(props, 'rhol'))

    def test_pfd_density_reference_feeds_rackett_path(self):
        pfd = (
            'PROCESS: Density Reference\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: IDEAL\n'
            '\n'
            'COMPONENTS:\n'
            '    XRHO | X density test | MW=72.15, Tc=469.7, Pc=33.7, omega=0.25, rho=620.0, rho_T=298.15\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = XRHO:1.0\n'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'liquid_volume_zra_cache.json'
            with (
                patch.object(PropertyResolver, 'LIQUID_VOLUME_ZRA_CACHE_PATH', cache_path),
                patch('chemical_properties.OnlinePropertyFetcher.fetch_from_pubchem', return_value=None),
            ):
                sim = Simulator.from_string(pfd)
                result = sim.run()
                self.assertTrue(result.converged)
                props = sim.thermo._resolver_known_props['XRHO']
                volume = PropertyResolver().resolve_liquid_molar_volume(
                    'XRHO',
                    360.0,
                    props,
                )
                sqlite_path = cache_path.with_suffix('.sqlite')
                with sqlite3.connect(sqlite_path) as connection:
                    persisted = connection.execute(
                        "SELECT count(*) FROM runtime_json_cache "
                        "WHERE namespace = 'liquid_volume_zra_v1'"
                    ).fetchone()[0]

        self.assertEqual(props['property_correlations']['rhol']['equation'], 'density_reference')
        self.assertEqual(volume.method, 'rackett_fitted_zra')
        self.assertIn('provided_rho_reference', volume.notes)
        self.assertTrue(math.isfinite(volume.value))
        self.assertGreater(volume.value, 0.0)
        self.assertEqual(persisted, 0)

    def test_pfd_property_correlations_feed_resolver(self):
        pfd = (
            'PROCESS: Resolver Correlations\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: IDEAL\n'
            '\n'
            'COMPONENTS:\n'
            '    XCOR | X correlation test | MW=50.0, Tc=600.0, Pc=30.0, Tb=350.0\n'
            '\n'
            'PROPERTY_CORRELATIONS:\n'
            '    XCOR.Psat | equation=canonical_psat, Tmin_K=100.0, Tmax_K=600.0, '
            'Tc=600.0, Pc_bar=30.0, Tb=350.0, A=8.14444553485238, '
            'B=-2845.948891914135, C=0.0, D=0.0, E=0.0, F=0.0\n'
            '    XCOR.Hvap | equation=poly_x, Tmin_K=250.0, Tmax_K=500.0, A=40.0, B=-10.0\n'
            '    XCOR.Cpl | equation=poly_x, Tmin_K=250.0, Tmax_K=500.0, A=80.0, B=5.0\n'
            '    XCOR.Cpg | equation=shomate, Tmin_K=250.0, Tmax_K=500.0, A=30.0, B=2.0, C=3.0, D=4.0, E=5.0\n'
            '    XCOR.rhol | equation=poly_x, Tmin_K=250.0, Tmax_K=350.0, A=600.0, B=-50.0\n'
            '    XCOR.mul | equation=poly_tp, Tmin_K=250.0, Tmax_K=500.0, P_ref_bar=5.0, A=0.001, B=0.002, C=0.003, D=0.004, E=0.005, F=0.006\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = XCOR:1.0\n'
        )

        with patch('chemical_properties.OnlinePropertyFetcher.fetch_from_pubchem', return_value=None):
            sim = Simulator.from_string(pfd)
            result = sim.run()

        self.assertTrue(result.converged)
        known = sim.thermo._resolver_known_props['XCOR']
        resolver = PropertyResolver()
        x = (300.0 - 298.15) / 100.0
        psat = resolver.resolve_vapor_pressure('XCOR', 300.0, known)
        hvap = resolver.resolve_hvap('XCOR', known, T=300.0, allow_online=False, allow_estimation=False)
        cp = resolver.resolve_heat_capacity('XCOR', 300.0, phase='liquid', props=known)
        cp_gas = resolver.resolve_heat_capacity('XCOR', 300.0, phase='ideal_gas', props=known)
        density = resolver.resolve_liquid_molar_density('XCOR', 300.0, known)
        with patch.object(resolver, '_coolprop_viscosity', side_effect=AssertionError):
            viscosity = resolver.resolve_viscosity(
                'XCOR',
                300.0,
                phase='liquid',
                props=known,
                P=6.0,
            )
        self.assertEqual(psat.method, 'pfd_canonical_psat')
        self.assertAlmostEqual(
            psat.value,
            math.exp(8.14444553485238 - 2845.948891914135 / 300.0),
        )
        self.assertEqual(psat.quality, 1.0)
        self.assertEqual(hvap.method, 'provided_hvap_fit')
        self.assertAlmostEqual(hvap.value, 40.0 - 10.0 * x)
        self.assertEqual(hvap.quality, 1.0)
        self.assertEqual(cp.method, 'provided_heat_capacity_fit')
        self.assertAlmostEqual(cp.value, 80.0 + 5.0 * x)
        self.assertEqual(cp.quality, 1.0)
        t = 300.0 / 1000.0
        self.assertEqual(cp_gas.method, 'provided_heat_capacity_fit')
        self.assertAlmostEqual(cp_gas.value, 30.0 + 2.0 * t + 3.0 * t * t + 4.0 * t**3 + 5.0 / (t * t))
        self.assertEqual(cp_gas.quality, 1.0)
        self.assertEqual(density.method, 'provided_liquid_density_fit')
        self.assertAlmostEqual(density.value, (600.0 - 50.0 * x) / 50.0)
        self.assertEqual(density.quality, 1.0)
        p = 6.0 - 5.0
        self.assertEqual(viscosity.method, 'provided_viscosity_fit')
        self.assertAlmostEqual(
            viscosity.value,
            0.001 + 0.002 * x + 0.003 * p + 0.004 * x * x + 0.005 * x * p + 0.006 * p * p,
        )
        self.assertEqual(viscosity.quality, 1.0)

    def test_pfd_correlations_and_scalars_take_priority_over_lookup_sources(self):
        pfd = (
            'PROCESS: Priority Conflicts\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: IDEAL\n'
            '\n'
            'COMPONENTS:\n'
            '    AAC | acetic acid | MW=60.052, Tc=592.7, Pc=57.9, Tb=391.05, Hvap=99.0, rho=900.0, rho_T=298.15\n'
            '\n'
            'PROPERTY_CORRELATIONS:\n'
            '    AAC.Psat | equation=canonical_psat, Tmin_K=100.0, Tmax_K=592.7, '
            'Tc=592.7, Pc_bar=57.9, Tb=391.05, A=11.904063515788948, '
            'B=-4649.936651968166, C=0.0, D=0.0, E=0.0, F=0.0\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = AAC:1.0\n'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'liquid_volume_zra_cache.json'
            with patch.object(PropertyResolver, 'LIQUID_VOLUME_ZRA_CACHE_PATH', cache_path):
                sim = Simulator.from_string(pfd)
                result = sim.run()
                self.assertTrue(result.converged)
                known = sim.thermo._resolver_known_props['AAC']

                resolver = PropertyResolver()
                psat = resolver.resolve_vapor_pressure('AAC', 300.0, known)
                hvap = resolver.resolve_hvap('AAC', known, T=300.0, allow_online=False, allow_estimation=False)
                volume = resolver.resolve_liquid_molar_volume('AAC', 430.0, known)

        expected_hvap = 99.0 * ((1.0 - 300.0 / 592.7) / (1.0 - 391.05 / 592.7)) ** 0.38
        self.assertEqual(psat.method, 'pfd_canonical_psat')
        self.assertAlmostEqual(
            psat.value,
            math.exp(11.904063515788948 - 4649.936651968166 / 300.0),
        )
        self.assertEqual(psat.quality, 1.0)
        self.assertEqual(hvap.method, 'watson_hvap')
        self.assertAlmostEqual(hvap.value, expected_hvap)
        self.assertLess(hvap.quality, 1.0)
        self.assertEqual(volume.method, 'rackett_fitted_zra')
        self.assertIn('provided_rho_reference', volume.notes)
        self.assertLess(volume.quality, 1.0)

    def test_pfd_component_can_resolve_as_one_chemical_but_behave_like_another(self):
        pfd = (
            'PROCESS: Alias Component Pretends To Be Water\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: IDEAL\n'
            '\n'
            'COMPONENTS:\n'
            '    AQUA | ethanol | formula=H2O, MW=18.015, Tc=647.096, Pc=220.64, '
            'Vc=55.95, Zc=0.229, omega=0.344, Tb=373.15, '
            'Tt=273.16, Pt=0.00611655, Tm=273.15, Hvap=40.65, '
            'rho=997.0, rho_T=298.15\n'
            '\n'
            'PROPERTY_CORRELATIONS:\n'
            '    AQUA.Psat | equation=poly_x, Tmin_K=290.0, Tmax_K=395.0, '
            'A=0.031637262341485, B=0.188625912565902, '
            'C=0.499386650493013, D=0.666306019038771, '
            'E=0.713219031763572, F=0.221122902516808\n'
            '    AQUA.Cpl | equation=poly_x, Tmin_K=290.0, Tmax_K=395.0, A=75.3, B=0.0\n'
            '    AQUA.Cpg | equation=poly_x, Tmin_K=290.0, Tmax_K=395.0, A=33.6, B=0.0\n'
            '\n'
            'STREAM Feed : FEED -> P-1.in\n'
            '    T = 26.85 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = AQUA:1.0\n'
            '\n'
            'STREAM Pumped : P-1.out -> H-1.in\n'
            'STREAM Boiled : H-1.out -> PRODUCT\n'
            '\n'
            'UNIT P-1\n'
            '    TYPE: Pump\n'
            '    PORTS:\n'
            '        in  : inlet\n'
            '        out : outlet\n'
            '    PARAMS:\n'
            '        P_out = 1.2 [bar]\n'
            '        eta = 1.0\n'
            '\n'
            'UNIT H-1\n'
            '    TYPE: Heater\n'
            '    PORTS:\n'
            '        in  : inlet\n'
            '        out : outlet\n'
            '    PARAMS:\n'
            '        vapor_fraction = 1.0\n'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / 'liquid_volume_zra_cache.json'
            with patch.object(PropertyResolver, 'LIQUID_VOLUME_ZRA_CACHE_PATH', cache_path):
                sim = Simulator.from_string(pfd)
                result = sim.run()
                self.assertTrue(result.converged)
                props = sim.thermo.props['AQUA']
                known = sim.thermo._resolver_known_props['AQUA']
                psat = sim.thermo.Psat('AQUA', 300.0)
                psat_resolver = PropertyResolver()
                resolved_psat = psat_resolver.resolve_vapor_pressure(
                    'AQUA',
                    300.0,
                    known,
                )
                canonical_curve = next(iter(
                    psat_resolver._canonical_vapor_pressure_curves.values()
                )).curve
                cp_liquid = sim.thermo.Cp_liquid('AQUA', 300.0)
                cp_gas = sim.thermo.Cp_ideal_gas('AQUA', 300.0)
                hvap = sim.thermo.Hvap_at_T('AQUA', 300.0)
                volume = PropertyResolver().resolve_liquid_molar_volume('AQUA', 360.0, known)
                cmo_hvap = McCabeThieleDistillation('D-TEST', sim.thermo, {})._mixture_hvap({'AQUA': 1.0}, 300.0)

        expected_hvap = 40.65 * ((1.0 - 300.0 / 647.096) / (1.0 - 373.15 / 647.096)) ** 0.38
        self.assertEqual(props.name.lower(), 'ethanol')
        self.assertEqual(props.CAS, '64-17-5')
        self.assertEqual(props.formula, 'H2O')
        self.assertAlmostEqual(props.MW, 18.015)
        self.assertAlmostEqual(result.streams['Feed'].MW, 18.015)
        self.assertEqual(known['property_sources']['MW']['method'], 'pfd_component_override')
        self.assertEqual(known['property_sources']['Tc']['method'], 'pfd_component_override')
        self.assertEqual(known['property_sources']['Tt']['method'], 'pfd_component_override')
        self.assertEqual(known['property_sources']['Pt']['method'], 'pfd_component_override')
        self.assertEqual(known['property_sources']['Tm']['method'], 'pfd_component_override')
        self.assertEqual(known['property_sources']['property_correlations']['method'], 'pfd_property_correlations')
        self.assertEqual(resolved_psat.method, 'pfd_psat_poly_x')
        self.assertAlmostEqual(canonical_curve.T_min, 273.16, places=10)
        self.assertEqual(
            canonical_curve.metadata['domain_selection']['basis'],
            'triple_point',
        )
        pfd_fit = next(
            item
            for item in canonical_curve.metadata['slice_fit_diagnostics']
            if item['priority'] == 1000
        )
        self.assertLessEqual(pfd_fit['mard_percent'], 0.25)
        self.assertLessEqual(
            pfd_fit['max_absolute_relative_error_percent'],
            1.0,
        )
        x_psat = (300.0 - 298.15) / 100.0
        expected_psat = sum(
            coefficient * x_psat**power
            for power, coefficient in enumerate((
                0.031637262341485,
                0.188625912565902,
                0.499386650493013,
                0.666306019038771,
                0.713219031763572,
                0.221122902516808,
            ))
        )
        self.assertTrue(
            math.isclose(psat, expected_psat, rel_tol=1.0e-2),
            f"{psat!r} != {expected_psat!r}",
        )
        self.assertAlmostEqual(cp_liquid, 75.3, places=8)
        self.assertAlmostEqual(cp_gas, 33.6, places=8)
        self.assertAlmostEqual(hvap, expected_hvap, places=8)
        self.assertAlmostEqual(cmo_hvap, expected_hvap, places=8)
        self.assertEqual(volume.method, 'rackett_fitted_zra')
        self.assertIn('provided_rho_reference', volume.notes)
        self.assertTrue(math.isfinite(volume.value))
        self.assertGreater(volume.value, 0.0)
        self.assertAlmostEqual(result.streams['Pumped'].P, 1.2, places=8)
        self.assertEqual(result.units['P-1'].performance['liquid_volume_basis'], 'inlet_stream_density')
        self.assertGreater(result.units['P-1'].work, 0.0)
        self.assertAlmostEqual(result.streams['Boiled'].vapor_fraction, 1.0, places=8)
        self.assertAlmostEqual(
            result.streams['Boiled'].T,
            377.96085,
            delta=0.1,
        )
        self.assertGreater(result.units['H-1'].heat_duty, 0.0)

    def test_pfd_psat_override_does_not_inherit_lookup_phase_points(self):
        sim = Simulator.from_string(
            'PROCESS: PFD Psat Phase Isolation\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: IDEAL\n'
            'COMPONENTS:\n'
            '    AQUA | ethanol\n'
            'PROPERTY_CORRELATIONS:\n'
            '    AQUA.Psat | equation=poly_x, Tmin_K=290, Tmax_K=395, '
            'A=0.1, B=0.1\n'
        )

        result = sim.run()
        props = sim.thermo.props['AQUA']

        self.assertTrue(result.converged)
        self.assertIsNone(props.Tt)
        self.assertIsNone(props.Pt)
        self.assertIsNone(props.Tm)
        self.assertNotIn('Tt', props.property_sources)
        self.assertNotIn('Pt', props.property_sources)
        self.assertNotIn('Tm', props.property_sources)

    def test_pfd_overrides_apply_before_pr_eos_construction(self):
        pfd = (
            'PROCESS: Constructor Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: PR\n'
            '\n'
            'COMPONENTS:\n'
            '    AQUA | ethanol | CAS=7732-18-5, formula=H2O, MW=18.015, '
            'Tc=647.096, Pc=220.64, omega=0.344, Tb=373.15, '
            'Tt=273.16, Pt=0.00611657\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 300 [K]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = AQUA:1.0\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()

        self.assertTrue(result.converged)
        props = sim.thermo.props['AQUA']
        eos_props = sim.thermo.cubic.props['AQUA']
        eos_params = sim.thermo.cubic.params['AQUA']
        self.assertEqual(props.CAS, '7732-18-5')
        self.assertEqual(sim.thermo.cubic.component_cas['AQUA'], '7732-18-5')
        self.assertAlmostEqual(eos_props.Tc, 647.096)
        self.assertAlmostEqual(eos_props.Pc, 220.64)
        self.assertAlmostEqual(eos_params.Tc, 647.096)
        self.assertAlmostEqual(eos_params.Pc, 220.64)
        self.assertEqual(props.property_sources['Tc']['method'], 'pfd_component_override')
        self.assertAlmostEqual(props.Tt, 273.16)
        self.assertAlmostEqual(props.Pt, 0.00611657)
        self.assertEqual(props.property_sources['Tt']['method'], 'pfd_component_override')
        self.assertEqual(props.property_sources['Pt']['method'], 'pfd_component_override')
        self.assertEqual(props.property_sources['Tt']['quality'], 1.0)
        self.assertEqual(props.property_sources['Pt']['quality'], 1.0)

    def test_pfd_custom_alias_inherits_local_properties_by_cas_identifier(self):
        sim = Simulator.from_string(
            'PROCESS: Explicit CAS Alias Lookup\n'
            'VERSION: 1.0\n'
            'ONLINE_LOOKUP: false\n'
            'THERMO_METHOD: IDEAL\n'
            'COMPONENTS:\n'
            '    LLA | 79-33-4 | formula=C3H6O3, '
            'MW=90.078, SMILES=C[C@H](O)C(=O)O\n'
        )

        result = sim.run()
        props = sim.thermo.props['LLA']

        self.assertTrue(result.converged)
        self.assertEqual(props.symbol, 'LLA')
        self.assertEqual(props.name, '(S)-Lactic acid')
        self.assertEqual(props.CAS, '79-33-4')
        self.assertAlmostEqual(props.Hf, -615.9)
        self.assertEqual(
            props.property_sources['Hf']['method'],
            'derived_phase_conversion',
        )
        self.assertAlmostEqual(props.property_sources['Hf']['quality'], 0.96)

    def test_l_lactic_name_resolves_by_structure_to_bundled_s_card(self):
        from rdkit import Chem

        database = ChemicalDatabase(enable_online=False)
        for identifier in ('L-lactic acid', 'l-LACTIC ACID', '(S)-Lactic acid'):
            with self.subTest(identifier):
                props = database.get(identifier, fetch_online=False)
                self.assertIsNotNone(props)
                self.assertEqual(props.name, '(S)-Lactic acid')
                self.assertEqual(props.CAS, '79-33-4')
                self.assertEqual(props.smiles, 'C[C@H](O)C(=O)O')
                self.assertAlmostEqual(props.Hf, -615.9)
                molecule = Chem.MolFromSmiles(props.smiles)
                Chem.AssignStereochemistry(
                    molecule,
                    force=True,
                    cleanIt=True,
                )
                self.assertEqual(
                    Chem.FindMolChiralCenters(
                        molecule,
                        includeUnassigned=True,
                        includeCIP=True,
                    ),
                    [(1, 'S')],
                )
                self.assertEqual(
                    Chem.MolToInchiKey(molecule),
                    'JVTAAEKCZFNVCJ-REOHCLBHSA-N',
                )

        sim = Simulator.from_string(
            'PROCESS: L-Lactic Name Identifier\n'
            'VERSION: 1.0\n'
            'ONLINE_LOOKUP: false\n'
            'THERMO_METHOD: IDEAL\n'
            'COMPONENTS:\n'
            '    LLA | L-lactic acid\n'
        )
        result = sim.run()
        props = sim.thermo.props['LLA']
        self.assertTrue(result.converged)
        self.assertEqual(props.name, '(S)-Lactic acid')
        self.assertEqual(props.CAS, '79-33-4')
        self.assertAlmostEqual(props.Hf, -615.9)
        self.assertEqual(
            props.property_sources['Hf']['method'],
            'derived_phase_conversion',
        )

    def test_pfd_ambiguous_formula_symbol_uses_resolved_name_alias(self):
        sim = Simulator.from_string(
            'PROCESS: Ambiguous Formula Symbol Alias\n'
            'VERSION: 1.0\n'
            'ONLINE_LOOKUP: false\n'
            'THERMO_METHOD: IDEAL\n'
            'COMPONENTS:\n'
            '    C3H6O | Acetone\n'
        )

        result = sim.run()
        props = sim.thermo.props['C3H6O']

        self.assertTrue(result.converged)
        self.assertEqual(props.symbol, 'C3H6O')
        self.assertEqual(props.name, 'Acetone')
        self.assertEqual(props.CAS, '67-64-1')
        self.assertAlmostEqual(props.Hf, -215.7)

    def test_component_identifier_formula_policy_is_explicit(self):
        ambiguous = (
            'PROCESS: Ambiguous Formula Identifier\n'
            'VERSION: 1.0\n'
            'ONLINE_LOOKUP: false\n'
            'THERMO_METHOD: IDEAL\n'
            'COMPONENTS:\n'
            '    X | C3H6O | MW=58.08\n'
        )
        with self.assertRaisesRegex(
            SimulationError,
            'ambiguous molecular formula',
        ):
            Simulator.from_string(ambiguous).run()

        custom = Simulator.from_string(
            'PROCESS: Unambiguous Formula Identifier\n'
            'VERSION: 1.0\n'
            'ONLINE_LOOKUP: false\n'
            'THERMO_METHOD: IDEAL\n'
            'COMPONENTS:\n'
            '    X | CH4O\n'
        )
        result = custom.run()
        props = custom.thermo.props['X']

        self.assertTrue(result.converged)
        self.assertEqual(props.source, 'pfd_defined')
        self.assertEqual(props.formula, 'CH4O')
        self.assertAlmostEqual(props.MW, 32.042, places=3)
        self.assertTrue(any(
            "unambiguous formula identifier 'CH4O'" in warning
            for warning in result.warnings
        ))

        known = Simulator.from_string(
            'PROCESS: Known Formula Identifier\n'
            'VERSION: 1.0\n'
            'ONLINE_LOOKUP: false\n'
            'THERMO_METHOD: IDEAL\n'
            'COMPONENTS:\n'
            '    X | CO\n'
        )
        known_result = known.run()
        known_props = known.thermo.props['X']
        self.assertTrue(known_result.converged)
        self.assertEqual(known_props.name, 'Carbon Monoxide')
        self.assertEqual(known_props.CAS, '630-08-0')
        self.assertFalse(any(
            'formula identifier' in warning
            for warning in known_result.warnings
        ))

        overlap = Simulator.from_string(
            'PROCESS: Formula Beats SMILES\n'
            'VERSION: 1.0\n'
            'ONLINE_LOOKUP: false\n'
            'THERMO_METHOD: IDEAL\n'
            'COMPONENTS:\n'
            '    X | CCO\n'
        )
        overlap_result = overlap.run()
        overlap_props = overlap.thermo.props['X']
        self.assertTrue(overlap_result.converged)
        self.assertEqual(overlap_props.source, 'pfd_defined')
        self.assertEqual(overlap_props.formula, 'CCO')
        self.assertAlmostEqual(overlap_props.MW, 40.021, places=3)
        self.assertNotEqual(overlap_props.CAS, '64-17-5')

    def test_component_structural_identifiers_resolve_local_acetone(self):
        identifiers = (
            'CC(=O)C',
            'InChI=1S/C3H6O/c1-3(2)4/h1-2H3',
            'CSCPPACGZOOCGX-UHFFFAOYSA-N',
        )
        for identifier in identifiers:
            with self.subTest(identifier):
                sim = Simulator.from_string(
                    'PROCESS: Structural Identifier\n'
                    'VERSION: 1.0\n'
                    'ONLINE_LOOKUP: false\n'
                    'THERMO_METHOD: IDEAL\n'
                    'COMPONENTS:\n'
                    f'    X | {identifier}\n'
                )
                result = sim.run()
                props = sim.thermo.props['X']

                self.assertTrue(result.converged)
                self.assertEqual(props.name, 'Acetone')
                self.assertEqual(props.CAS, '67-64-1')
                self.assertAlmostEqual(props.Hf, -215.7)

    def test_identifier_prescreen_routes_only_names_to_opsin(self):
        expected = {
            '67-64-1': 'cas',
            'Acetone': 'name',
            '1-octen-3-one': 'name',
            'C3H6O': 'formula',
            'CCO': 'formula',
            'CC(=O)C': 'smiles',
            'InChI=1S/C3H6O/c1-3(2)4/h1-2H3': 'inchi',
            'CSCPPACGZOOCGX-UHFFFAOYSA-N': 'inchikey',
        }
        for identifier, kind in expected.items():
            with self.subTest(identifier):
                self.assertEqual(classify_identifier(identifier), kind)

        with tempfile.TemporaryDirectory() as tmpdir:
            database = ChemicalDatabase(enable_online=False)
            database._smiles_cache_path = Path(tmpdir) / 'smiles.sqlite'
            with patch.object(
                database,
                '_smiles_from_opsin',
                return_value=None,
            ) as opsin:
                for identifier in (
                    '67-64-1',
                    'CC(=O)C',
                    'InChI=1S/C3H6O/c1-3(2)4/h1-2H3',
                    'CSCPPACGZOOCGX-UHFFFAOYSA-N',
                ):
                    database.resolve_smiles_info(
                        identifier,
                        fetch_online=False,
                    )
                opsin.assert_not_called()

                database.resolve_smiles_info(
                    'unlisted systematic compound name',
                    fetch_online=False,
                )
                opsin.assert_called_once_with(
                    'unlisted systematic compound name'
                )

    def test_unknown_systematic_name_uses_opsin_structure_and_estimators(self):
        sim = Simulator.from_string(
            'PROCESS: OPSIN Structure Promotion\n'
            'VERSION: 1.0\n'
            'ONLINE_LOOKUP: false\n'
            'THERMO_METHOD: IDEAL\n'
            'COMPONENTS:\n'
            '    X | 2,2,3-triiodopropionic acid\n'
        )

        result = sim.run()
        props = sim.thermo.props['X']

        self.assertTrue(result.converged)
        self.assertEqual(props.formula, 'C3H3I3O2')
        self.assertEqual(props.smiles, 'IC(C(=O)O)(CI)I')
        self.assertAlmostEqual(props.MW, 451.767, places=3)
        self.assertEqual(props.property_sources['smiles']['source'], 'opsin')
        self.assertIn(
            props.property_sources['smiles']['method'],
            {'py2opsin', 'opsin_smiles_cache'},
        )
        self.assertTrue(any(
            'resolved to molecular structure' in warning
            for warning in result.warnings
        ))

    def test_unknown_inchi_converts_locally_but_unknown_inchikey_needs_lookup(self):
        inchi = 'InChI=1S/C3H3I3O2/c4-1-3(5,6)2(7)8/h1H2,(H,7,8)'
        sim = Simulator.from_string(
            'PROCESS: Direct InChI Structure Promotion\n'
            'VERSION: 1.0\n'
            'ONLINE_LOOKUP: false\n'
            'THERMO_METHOD: IDEAL\n'
            'COMPONENTS:\n'
            f'    X | {inchi}\n'
        )
        result = sim.run()
        props = sim.thermo.props['X']
        self.assertTrue(result.converged)
        self.assertEqual(props.formula, 'C3H3I3O2')
        self.assertEqual(props.property_sources['smiles']['source'], 'rdkit')
        self.assertIn(
            props.property_sources['smiles']['method'],
            {'rdkit_inchi_to_smiles', 'rdkit_smiles_cache'},
        )
        direct = ChemicalDatabase._smiles_from_inchi_identifier(inchi)
        self.assertIsNotNone(direct)
        self.assertEqual(direct.method, 'rdkit_inchi_to_smiles')

        with self.assertRaisesRegex(SimulationError, 'is not in the property database'):
            Simulator.from_string(
                'PROCESS: Unknown InChIKey Offline\n'
                'VERSION: 1.0\n'
                'ONLINE_LOOKUP: false\n'
                'THERMO_METHOD: IDEAL\n'
                'COMPONENTS:\n'
                '    X | RVDVRLLBJFYBEL-UHFFFAOYSA-N\n'
            ).run()

    def test_partial_pfd_coupled_overrides_rehydrate_before_eos_construction(self):
        pfd = (
            'PROCESS: Partial Coupled Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: PR\n'
            '\n'
            'COMPONENTS:\n'
            '    ETOH | ethanol | CAS=64-17-5, MW=46.06844, '
            'Tc=500.0, Pc=50.0, Tt=150.0\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 300 [K]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = ETOH:1.0\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()

        self.assertTrue(result.converged)
        props = sim.thermo.props['ETOH']
        expected_zc = (
            props.Pc * 100000.0 * props.Vc * 1.0e-6
            / (8.314462618 * props.Tc)
        )
        self.assertAlmostEqual(props.Zc, expected_zc)
        self.assertEqual(
            props.property_sources['Zc']['method'],
            'critical_volume_identity',
        )
        self.assertIsNone(props.Pt)
        self.assertNotIn('Pt', props.property_sources)

    def test_pfd_uniquac_rq_override_is_used_during_construction(self):
        pfd = (
            'PROCESS: UNIQUAC RQ Override\n'
            'VERSION: 1.0\n'
            'THERMO_METHOD: UNIQUAC\n'
            '\n'
            'COMPONENTS:\n'
            '    ETOH | ethanol | MW=46.07, uniquac_r=9.0, uniquac_q=8.0\n'
            '    H2O | water | MW=18.015\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 298.15 [K]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = ETOH:0.5, H2O:0.5\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()

        self.assertTrue(result.converged)
        self.assertAlmostEqual(sim.thermo.r['ETOH'], 9.0)
        self.assertAlmostEqual(sim.thermo.q['ETOH'], 8.0)
        self.assertEqual(
            sim.thermo.props['ETOH'].property_sources['uniquac_r']['method'],
            'pfd_component_override',
        )

    def test_pfd_uniquac_rq_override_is_ignored_for_non_uniquac_methods(self):
        pfd = (
            'PROCESS: Non-UNIQUAC Ignores RQ\n'
            'VERSION: 1.0\n'
            'ONLINE_LOOKUP: false\n'
            'THERMO_METHOD: PR\n'
            '\n'
            'COMPONENTS:\n'
            '    X | Imaginary | MW=40.0, Tc=400.0, Pc=40.0, omega=0.2, '
            'uniquac_r=9.0, uniquac_q=8.0, Cp_coeffs=[30.0,0.0,0.0,0.0]\n'
            '\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 350 [K]\n'
            '    P = 5 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = X:1.0\n'
        )

        sim = Simulator.from_string(pfd)
        result = sim.run()
        report = sim._generate_pfr()
        props = sim.thermo.props['X']

        self.assertTrue(result.converged)
        self.assertIsNone(props.uniquac_r)
        self.assertIsNone(props.uniquac_q)
        self.assertNotIn('uniquac_r', props.property_sources)
        self.assertTrue(any('Ignoring UNIQUAC pure-component parameters for X' in warning for warning in result.warnings))
        self.assertIn('Ignoring UNIQUAC pure-component parameters for X', report)

    def test_pfd_vdm_override_is_ignored_for_non_vdm_methods(self):
        pfd = (
            'PROCESS: Non-VDM Ignores Association\n'
            'VERSION: 1.0\n'
            'ONLINE_LOOKUP: false\n'
            'THERMO_METHOD: IDEAL\n\n'
            'COMPONENTS:\n'
            '    X | Methane | MW=16.04, Cp_coeffs=[30.0,0.0,0.0,0.0], '
            'VDM={delta_H:-50000,delta_S:-120}\n\n'
            '    Y | Ethane | MW=30.07, Cp_coeffs=[35.0,0.0,0.0,0.0], '
            'VDM={delta_H:-45000,delta_S:-110}\n\n'
            'INTERACTION_PARAMETERS:\n'
            '    X/Y | model=VDM, delta_H_residual=1000, '
            'delta_S_residual=-2.5\n\n'
            'STREAM Feed : FEED -> PRODUCT\n'
            '    T = 350 [K]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = X:0.5, Y:0.5\n'
        )

        simulator = Simulator.from_string(pfd)
        result = simulator.run()
        report = simulator._generate_pfr()
        props = simulator.thermo.props['X']

        self.assertTrue(result.converged)
        self.assertIsNone(props.vapor_dimerization)
        self.assertNotIn('vapor_dimerization', props.property_sources)
        self.assertTrue(any(
            'Ignoring vapor-dimerization parameters for X' in warning
            for warning in result.warnings
        ))
        self.assertIn('Ignoring vapor-dimerization parameters for X', report)
        self.assertTrue(any(
            'Ignoring VDM cross-interaction parameters for X/Y' in warning
            for warning in result.warnings
        ))
        self.assertIn('Ignoring VDM cross-interaction parameters for X/Y', report)


if __name__ == '__main__':
    unittest.main()
