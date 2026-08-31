import unittest

from pfd_parser import PFDParser, ParseError, ProcessFlowDiagram, validate_pfd
from simulator import Simulator


KINETIC_PARAMETERS = (
    'A=2, Ea=0, Ea_unit=J/mol, rate_basis=concentration, '
    'concentration_unit=kmol/m3, pressure_unit=bar, rate_unit=kmol/m3/h'
)


def catalog_pfd(unit_reaction='@kinetic_isomerization'):
    return f'''\
PROCESS: Reusable reaction catalog
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
COMPONENTS:
    C2H4O | Ethylene oxide | formula=C2H4O
    CH3CHO | Acetaldehyde | formula=C2H4O
REACTIONS:
    kinetic_isomerization : C2H4O -> CH3CHO | {KINETIC_PARAMETERS}
    packed_isomerization : C2H4O -> CH3CHO | A=2, Ea=0, Ea_unit=J/mol, rate_basis=concentration, concentration_unit=kmol/m3, pressure_unit=bar, rate_unit=kmol/kg_cat/h
    fixed_isomerization : C2H4O -> CH3CHO | conversion=0.25
    equilibrium_isomerization : C2H4O <=> CH3CHO
STREAM Feed : -> CSTR-1.in
    T = 500 [K]
    P = 2 [bar]
    F = 10 [kmol/h]
    x = C2H4O:1
STREAM Product : CSTR-1.out
UNIT CSTR-1 : CSTR
    volume = 5 [m3]
    T = 500 [K]
    phase = vapor
    REACTIONS:
        {unit_reaction}
'''


class ReactionCatalogParserTests(unittest.TestCase):
    def test_catalog_reference_parses_validates_and_preserves_reference_round_trip(self):
        pfd = PFDParser().parse(catalog_pfd())
        self.assertEqual(
            [definition.name for definition in pfd.reaction_definitions],
            [
                'kinetic_isomerization',
                'packed_isomerization',
                'fixed_isomerization',
                'equilibrium_isomerization',
            ],
        )
        self.assertEqual(pfd.units[0].reactions[0].reference, 'kinetic_isomerization')
        self.assertEqual(validate_pfd(pfd), ([], []))

        serialized = pfd.to_pfd()
        self.assertIn('kinetic_isomerization : C2H4O -> CH3CHO', serialized)
        self.assertIn('        @kinetic_isomerization', serialized)
        restored = PFDParser().parse(serialized)
        self.assertEqual(restored.units[0].reactions[0].reference, 'kinetic_isomerization')
        self.assertEqual(validate_pfd(restored), ([], []))

    def test_catalog_and_references_round_trip_through_dict(self):
        pfd = PFDParser().parse(catalog_pfd())
        restored = ProcessFlowDiagram.from_dict(pfd.to_dict())

        self.assertEqual(
            restored.reaction_definitions[0].reaction.parameters,
            pfd.reaction_definitions[0].reaction.parameters,
        )
        self.assertEqual(
            restored.units[0].reactions[0].reference,
            'kinetic_isomerization',
        )
        self.assertEqual(validate_pfd(restored), ([], []))

    def test_custom_net_catalog_reference_preserves_signed_model(self):
        text = catalog_pfd('@signed_net').replace(
            '    kinetic_isomerization :',
            "    signed_net : C2H4O -> CH3CHO | type=custom_net, A=1, "
            "Ea=0, Ea_unit=J/mol, concentration_unit=kmol/m3, "
            "pressure_unit=bar, rate_unit=kmol/m3/h, "
            "expression=k*(C['C2H4O']-C['CH3CHO'])\n"
            '    kinetic_isomerization :',
        )
        pfd = PFDParser().parse(text)
        self.assertEqual(validate_pfd(pfd), ([], []))
        self.assertEqual(
            pfd.get_reaction_definition('signed_net').reaction.parameters['type'],
            'custom_net',
        )

        restored = ProcessFlowDiagram.from_dict(pfd.to_dict())
        self.assertEqual(validate_pfd(restored), ([], []))
        self.assertEqual(
            restored.units[0].reactions[0].reference,
            'signed_net',
        )

    def test_unit_can_mix_catalog_references_and_inline_reactions(self):
        pfd = PFDParser().parse(catalog_pfd(
            '@kinetic_isomerization\n'
            '        C2H4O -> CH3CHO | '
            + KINETIC_PARAMETERS.replace('A=2', 'A=1')
        ))
        self.assertEqual(len(pfd.units[0].reactions), 2)
        self.assertEqual(pfd.units[0].reactions[0].reference, 'kinetic_isomerization')
        self.assertIsNone(pfd.units[0].reactions[1].reference)
        self.assertEqual(validate_pfd(pfd), ([], []))

    def test_catalog_references_validate_for_every_current_reactor_family(self):
        cases = (
            ('CSTR', '@kinetic_isomerization'),
            ('BatchReactor', '@kinetic_isomerization'),
            ('PFR', '@kinetic_isomerization'),
            ('PackedBedReactor', '@packed_isomerization'),
            ('Reactor', '@fixed_isomerization'),
            ('EquilibriumReactor', '@equilibrium_isomerization'),
        )
        for unit_type, reference in cases:
            with self.subTest(unit_type=unit_type):
                text = catalog_pfd(reference).replace(
                    'UNIT CSTR-1 : CSTR',
                    f'UNIT CSTR-1 : {unit_type}',
                )
                pfd = PFDParser().parse(text)
                self.assertEqual(validate_pfd(pfd), ([], []))

    def test_undefined_or_inapplicable_references_are_validation_errors(self):
        undefined = PFDParser().parse(catalog_pfd('@missing'))
        errors, _warnings = validate_pfd(undefined)
        self.assertTrue(any("Undefined reaction reference '@missing'" in error for error in errors))

        mixer_text = catalog_pfd().replace(
            'UNIT CSTR-1 : CSTR',
            'UNIT CSTR-1 : Mixer',
        )
        mixer = PFDParser().parse(mixer_text)
        errors, _warnings = validate_pfd(mixer)
        self.assertTrue(any('cannot load reaction reference' in error for error in errors))

    def test_duplicate_catalog_names_are_parse_errors(self):
        text = catalog_pfd().replace(
            '    fixed_isomerization :',
            '    kinetic_isomerization :',
        )
        with self.assertRaisesRegex(ParseError, 'Duplicate reaction definition'):
            PFDParser().parse(text)


class ReactionCatalogSimulationTests(unittest.TestCase):
    def test_catalog_reaction_is_resolved_when_creating_cstr(self):
        simulator = Simulator.from_string(catalog_pfd())
        result = simulator.run()
        self.assertTrue(result.converged, result.warnings)
        self.assertEqual(result.errors, [])
        performance = result.units['CSTR-1'].performance
        self.assertEqual(
            performance['reactions'][0]['equation'],
            'C2H4O -> CH3CHO',
        )
        self.assertGreater(performance['component_conversions']['C2H4O'], 0.0)
        report = simulator._property_quality_report(include_suppressed=True)
        hf_entries = {
            entry['component']: entry
            for entry in report
            if entry['property'] == 'Hf'
        }
        for component in ('C2H4O', 'CH3CHO'):
            self.assertTrue(any(
                context.get('unit_id') == 'CSTR-1'
                for context in hf_entries[component]['contexts']
            ))

    def test_catalog_cstr_recycle_is_deterministic_and_closes_balances(self):
        pfd = f'''\
PROCESS: Catalog kinetic CSTR recycle
VERSION: 1.0
ONLINE_LOOKUP: false
THERMO_METHOD: IDEAL
RECYCLE_METHOD: WEGSTEIN
COMPONENTS:
    C2H4O | Ethylene oxide | formula=C2H4O
    CH3CHO | Acetaldehyde | formula=C2H4O
REACTIONS:
    isomerization : C2H4O -> CH3CHO | {KINETIC_PARAMETERS}
STREAM Feed : FEED -> M.fresh
    T = 500 [K]
    P = 2 [bar]
    F = 10 [kmol/h]
    x = C2H4O:1
STREAM Recycle : SP.recycle -> M.recycle
STREAM Mixed : M.out -> CSTR.in
STREAM Reacted : CSTR.out -> SP.in
STREAM Product : SP.product -> PRODUCT
UNIT M : Mixer
    T_out = 500 [K]
    P = 2 [bar]
UNIT CSTR : CSTR
    volume = 5 [m3]
    T = 500 [K]
    phase = vapor
    REACTIONS:
        @isomerization
UNIT SP : Splitter
    outlets = product, recycle
    product_split_frac = 0.5
'''
        result = Simulator.from_string(pfd).run(max_iterations=100)

        self.assertTrue(result.converged, result.warnings)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.recycle_info['tear_streams'], ['Recycle'])
        self.assertLess(result.mass_balance_error, 1.0e-4)
        self.assertLess(result.energy_balance_error, 1.0e-4)
        self.assertLess(
            result.units['CSTR'].performance[
                'maximum_material_rate_residual_kmol_h'
            ],
            1.0e-8,
        )


if __name__ == '__main__':
    unittest.main()
