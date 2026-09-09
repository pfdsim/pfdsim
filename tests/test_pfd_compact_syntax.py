import unittest

from pfd_parser import ParseError, PortType, parse_pfd, validate_pfd
from simulator import Simulator
from unit_operations import UNIT_CLASSES
from unit_syntax import (
    NUMERIC_PORT_LAYOUTS,
    PORT_FAMILY_SCHEMAS,
    UNIT_PORT_FAMILIES,
    UNIT_TYPE_ALIASES,
    canonical_unit_type,
    port_schema_for_unit_type,
)


class CompactUnitSyntaxTests(unittest.TestCase):
    maxDiff = None

    def test_every_registered_unit_name_accepts_compact_header(self):
        self.assertEqual(set(UNIT_CLASSES), set(UNIT_TYPE_ALIASES))

        for public_name, canonical_name in UNIT_TYPE_ALIASES.items():
            with self.subTest(unit_type=public_name):
                pfd = parse_pfd(f'UNIT U : {public_name}\n')
                self.assertEqual(pfd.units[0].unit_type, canonical_name)
                self.assertEqual(canonical_unit_type(public_name), canonical_name)
                self.assertIsNotNone(port_schema_for_unit_type(public_name))
                self.assertIs(
                    UNIT_CLASSES[public_name],
                    UNIT_CLASSES[canonical_name],
                )

                restored = type(pfd).from_dict({
                    'units': [{
                        'id': 'U',
                        'unit_type': public_name,
                        'ports': [],
                        'params': [],
                    }],
                })
                self.assertEqual(restored.units[0].unit_type, canonical_name)

    def test_every_compact_port_name_and_shorthand_resolves(self):
        representative_types = {
            family: unit_type
            for unit_type, family in UNIT_PORT_FAMILIES.items()
        }
        for family, schema in PORT_FAMILY_SCHEMAS.items():
            unit_type = representative_types[family]
            for direction, aliases_key in (
                ('inlet', 'inlets'),
                ('outlet', 'outlets'),
            ):
                for alias, canonical in schema[aliases_key].items():
                    with self.subTest(
                        family=family,
                        direction=direction,
                        alias=alias,
                    ):
                        route = (
                            f'-> U.{alias}'
                            if direction == 'inlet'
                            else f'U.{alias}'
                        )
                        pfd = parse_pfd(
                            f'STREAM S : {route}\n'
                            f'UNIT U : {unit_type}\n'
                        )
                        stream = pfd.streams[0]
                        reference = (
                            stream.destination
                            if direction == 'inlet'
                            else stream.source
                        )
                        self.assertEqual(reference.port_id, canonical)
                        port = pfd.units[0].get_port(canonical)
                        self.assertIsNotNone(port)
                        expected_type = schema['types'].get(
                            canonical,
                            'inlet' if direction == 'inlet' else 'outlet',
                        )
                        self.assertEqual(port.port_type, PortType(expected_type))
                        self.assertEqual(validate_pfd(pfd), ([], []))

    def test_every_direction_exclusive_alias_rejects_reverse_use(self):
        representative_types = {
            family: unit_type
            for unit_type, family in UNIT_PORT_FAMILIES.items()
        }
        for family, schema in PORT_FAMILY_SCHEMAS.items():
            unit_type = representative_types[family]
            inlet_aliases = set(schema['inlets'])
            outlet_aliases = set(schema['outlets'])
            cases = [
                (alias, f'U.{alias}', 'inlet-only', 'source')
                for alias in sorted(inlet_aliases - outlet_aliases)
            ] + [
                (alias, f'-> U.{alias}', 'outlet-only', 'destination')
                for alias in sorted(outlet_aliases - inlet_aliases)
            ]
            for alias, route, restriction, endpoint_role in cases:
                with self.subTest(
                    family=family,
                    alias=alias,
                    endpoint_role=endpoint_role,
                ):
                    with self.assertRaises(ParseError) as caught:
                        parse_pfd(
                            f'STREAM S : {route}\n'
                            f'UNIT U : {unit_type}\n'
                        )
                    message = str(caught.exception)
                    self.assertIn(f"Port alias '{alias}'", message)
                    self.assertIn(restriction, message)
                    self.assertIn(f'stream {endpoint_role}', message)

    def test_every_fixed_clockwise_numeric_layout_position(self):
        representative_types = {
            family: unit_type
            for unit_type, family in UNIT_PORT_FAMILIES.items()
        }
        for family, layout in NUMERIC_PORT_LAYOUTS.items():
            unit_type = representative_types[family]
            for number, (direction, canonical) in enumerate(layout, start=1):
                with self.subTest(
                    family=family,
                    number=number,
                    direction=direction,
                ):
                    route = (
                        f'-> U.{number}'
                        if direction == 'inlet'
                        else f'U.{number}'
                    )
                    pfd = parse_pfd(
                        f'STREAM S : {route}\n'
                        f'UNIT U : {unit_type}\n'
                    )
                    reference = (
                        pfd.streams[0].destination
                        if direction == 'inlet'
                        else pfd.streams[0].source
                    )
                    self.assertEqual(reference.port_id, canonical)
                    self.assertIsNotNone(pfd.units[0].get_port(canonical))

    def test_clockwise_numeric_mixer_and_splitter_layouts(self):
        mixer = parse_pfd(
            'STREAM A : -> M.1\n'
            'STREAM B : -> M.2\n'
            'STREAM C : -> M.3\n'
            'STREAM Mixed : M.4\n'
            'UNIT M : Mixer\n'
        )
        self.assertEqual(
            [stream.destination.port_id for stream in mixer.streams[:3]],
            ['in1', 'in2', 'in3'],
        )
        self.assertEqual(mixer.streams[3].source.port_id, 'out')
        self.assertEqual(
            {port.id: port.port_type.value for port in mixer.units[0].ports},
            {'in1': 'inlet', 'in2': 'inlet', 'in3': 'inlet', 'out': 'outlet'},
        )

        splitter = parse_pfd(
            'STREAM Feed : -> S.1\n'
            'STREAM First : S.2\n'
            'STREAM Second : S.3\n'
            'UNIT S : Splitter\n'
        )
        self.assertEqual(splitter.streams[0].destination.port_id, 'in')
        self.assertEqual(
            [stream.source.port_id for stream in splitter.streams[1:]],
            ['out', 'out2'],
        )
        outlets = next(
            param for param in splitter.units[0].params
            if param.name == 'outlets'
        )
        self.assertEqual(outlets.value, 'out,out2')

    def test_clockwise_numeric_distillation_feeds_and_products(self):
        two_feeds = parse_pfd(
            'STREAM LowerFeed : -> D.1\n'
            'STREAM HigherFeed : -> D.2\n'
            'STREAM Distillate : D.3\n'
            'STREAM Bottoms : D.4\n'
            'UNIT D : RigorousDistillation\n'
            '    N_stages = 12\n'
        )
        self.assertEqual(
            [stream.destination.port_id for stream in two_feeds.streams[:2]],
            ['feed1', 'feed2'],
        )
        self.assertEqual(
            [stream.source.port_id for stream in two_feeds.streams[2:]],
            ['distillate', 'bottoms'],
        )
        feed_stages = next(
            param for param in two_feeds.units[0].params
            if param.name == 'feed_stages'
        )
        self.assertEqual(feed_stages.value, 'feed1:8,feed2:4')

        explicit_stages = parse_pfd(
            'STREAM LowerFeed : -> D.1\n'
            'STREAM HigherFeed : -> D.2\n'
            'STREAM Distillate : D.3\n'
            'STREAM Bottoms : D.4\n'
            'UNIT D : RigorousDistillation\n'
            '    feed_stages = 1:9,2:3\n'
        )
        feed_stages = next(
            param for param in explicit_stages.units[0].params
            if param.name == 'feed_stages'
        )
        self.assertEqual(feed_stages.value, 'feed1:9,feed2:3')

        mixed = parse_pfd(
            'STREAM Feed : -> D.1\n'
            'STREAM VaporDistillate : D.2\n'
            'STREAM LiquidDistillate : D.3\n'
            'STREAM Bottoms : D.4\n'
            'UNIT D : RigorousDistillation\n'
            '    condenser_type = mixed\n'
        )
        self.assertEqual(
            [stream.source.port_id for stream in mixed.streams[1:]],
            ['distillate_vapor', 'distillate_liquid', 'bottoms'],
        )

        with self.assertRaisesRegex(
            ParseError,
            r"require side_draws specifications with stage and flow or fraction",
        ):
            parse_pfd(
                'STREAM Feed : -> D.1\n'
                'STREAM Distillate : D.2\n'
                'STREAM SideDraw : D.3\n'
                'STREAM Bottoms : D.4\n'
                'UNIT D : RigorousDistillation\n'
            )

        side_draw = parse_pfd(
            'STREAM Feed : -> D.1\n'
            'STREAM Distillate : D.2\n'
            'STREAM SideDraw : D.3\n'
            'STREAM Bottoms : D.4\n'
            'UNIT D : RigorousDistillation\n'
            '    side_draws = stage:5,phase:liquid,fraction:0.1\n'
        )
        self.assertEqual(
            [stream.source.port_id for stream in side_draw.streams[1:]],
            ['distillate', 'side_draw1', 'bottoms'],
        )
        side_draws = next(
            param for param in side_draw.units[0].params
            if param.name == 'side_draws'
        )
        self.assertIn('port:side_draw1', side_draws.value)
        serialized = side_draw.to_pfd()
        self.assertIn('STREAM SideDraw : D.side_draw1 -> PRODUCT', serialized)
        self.assertIn('side_draws = stage:5,phase:liquid,fraction:0.1,port:side_draw1', serialized)
        reparsed = parse_pfd(serialized)
        self.assertEqual(
            reparsed.get_stream('SideDraw').source.port_id,
            'side_draw1',
        )

    def test_numeric_layout_rejects_nonclockwise_direction_order(self):
        with self.assertRaisesRegex(
            ParseError,
            r"must increase clockwise.*every numbered inlet must precede",
        ):
            parse_pfd(
                'STREAM Product : F.1\n'
                'STREAM Feed : -> F.2\n'
                'UNIT F : Flash\n'
            )

        with self.assertRaisesRegex(
            ParseError,
            r"must be contiguous from 1; found 1, 3",
        ):
            parse_pfd(
                'STREAM Feed : -> M.1\n'
                'STREAM Product : M.3\n'
                'UNIT M : Mixer\n'
            )

        with self.assertRaisesRegex(
            ParseError,
            r"Numeric port 1.*connected more than once",
        ):
            parse_pfd(
                'STREAM Feed1 : -> M.1\n'
                'STREAM Feed2 : -> M.1\n'
                'UNIT M : Mixer\n'
            )

    def test_single_path_units_accept_unfamiliar_directional_names(self):
        pfd = parse_pfd(
            'STREAM Feed : -> P.whatever_the_inlet_is_called\n'
            'STREAM Product : P.delivery\n'
            'UNIT P : Pump\n'
        )
        self.assertEqual(
            (
                pfd.streams[0].destination.port_id,
                pfd.streams[1].source.port_id,
            ),
            ('in', 'out'),
        )

    def test_requested_short_forms_and_one_ended_streams(self):
        pfd = parse_pfd(
            'STREAM FlashVapor : FlashA.vapor\n'
            'STREAM FlashVap : FlashB.vap\n'
            'STREAM FlashShort : FlashC.v\n'
            'STREAM FlashTrailing : FlashD.l ->\n'
            'STREAM Flash3Liquid2 : Flash3.l2\n'
            'STREAM ColumnFeed : -> Distillation.f\n'
            'STREAM Distillate : Distillation.d\n'
            'STREAM Bottoms : Distillation.b\n'
            'UNIT FlashA : Flash\n'
            'UNIT FlashB : Flash\n'
            'UNIT FlashC : Flash\n'
            'UNIT FlashD : Flash\n'
            'UNIT Flash3 : Flash3\n'
            'UNIT Distillation : ShortcutDistillation\n'
        )

        expected = {
            'FlashVapor': ('FlashA.vapor_out', 'PRODUCT'),
            'FlashVap': ('FlashB.vapor_out', 'PRODUCT'),
            'FlashShort': ('FlashC.vapor_out', 'PRODUCT'),
            'FlashTrailing': ('FlashD.liquid_out', 'PRODUCT'),
            'Flash3Liquid2': ('Flash3.liquid2_out', 'PRODUCT'),
            'ColumnFeed': ('FEED', 'Distillation.feed'),
            'Distillate': ('Distillation.distillate', 'PRODUCT'),
            'Bottoms': ('Distillation.bottoms', 'PRODUCT'),
        }
        self.assertEqual({
            stream.id: (
                stream.source.to_string(),
                stream.destination.to_string(),
            )
            for stream in pfd.streams
        }, expected)
        self.assertEqual(validate_pfd(pfd), ([], []))

        reparsed = parse_pfd(pfd.to_pfd())
        self.assertEqual({
            stream.id: (
                stream.source.to_string(),
                stream.destination.to_string(),
            )
            for stream in reparsed.streams
        }, expected)

    def test_compact_parameters_need_no_params_section(self):
        pfd = parse_pfd(
            'STREAM Feed : -> F.i\n'
            'STREAM Vapor : F.v\n'
            'STREAM Liquid : F.l\n'
            'UNIT F : Flash\n'
            '    T = 50 [C]\n'
            '    P = 1 [bar]\n'
        )

        self.assertEqual(
            [(param.name, param.value, param.unit) for param in pfd.units[0].params],
            [('T', '50', 'C'), ('P', '1', 'bar')],
        )
        self.assertEqual(
            [(port.id, port.port_type.value) for port in pfd.units[0].ports],
            [
                ('in', 'inlet'),
                ('vapor_out', 'vapor_outlet'),
                ('liquid_out', 'liquid_outlet'),
            ],
        )

    def test_dynamic_ports_remain_available_where_unit_models_allow_them(self):
        pfd = parse_pfd(
            'STREAM Fresh : -> M.fresh\n'
            'STREAM Recycle : -> M.recycle\n'
            'STREAM Mixed : M.o\n'
            'STREAM SplitFeed : -> S.i\n'
            'STREAM Warm : S.warm\n'
            'STREAM Cold : S.cold\n'
            'STREAM ColumnFeed2 : -> D.feed2\n'
            'STREAM ColumnSideDraw : D.side_liquid\n'
            'UNIT M : Mixer\n'
            'UNIT S : Splitter\n'
            '    outlets = warm,cold\n'
            'UNIT D : RigorousDistillation\n'
        )

        by_unit = {
            unit.id: {port.id: port.port_type.value for port in unit.ports}
            for unit in pfd.units
        }
        self.assertEqual(
            by_unit['M'],
            {'fresh': 'inlet', 'recycle': 'inlet', 'out': 'outlet'},
        )
        self.assertEqual(
            by_unit['S'],
            {'in': 'inlet', 'warm': 'outlet', 'cold': 'outlet'},
        )
        self.assertEqual(
            by_unit['D'],
            {'feed2': 'inlet', 'side_liquid': 'outlet'},
        )
        self.assertEqual(validate_pfd(pfd), ([], []))

        fixed_port_cases = (
            ('ShortcutDistillation', 'outlet', 'side_liquid'),
            ('Absorber', 'inlet', 'stage3_gas'),
            ('Stripper', 'inlet', 'stage3_liquid'),
        )
        for unit_type, direction, port_name in fixed_port_cases:
            with self.subTest(
                unit_type=unit_type,
                direction=direction,
                port_name=port_name,
            ):
                route = (
                    f'U.{port_name}'
                    if direction == 'outlet'
                    else f'-> U.{port_name}'
                )
                with self.assertRaisesRegex(
                    ParseError,
                    rf"Unknown {unit_type} {direction} port alias '{port_name}'",
                ):
                    parse_pfd(
                        f'STREAM S : {route}\n'
                        f'UNIT U : {unit_type}\n'
                    )

    def test_stream_shorthand_rejects_missing_or_reversed_external_endpoints(self):
        cases = {
            'STREAM Empty : ->\n': 'must include a source or destination',
            'STREAM BackwardProduct : PRODUCT -> U.i\n': (
                'PRODUCT cannot be a stream source'
            ),
            'STREAM BackwardFeed : U.o -> FEED\n': (
                'FEED cannot be a stream destination'
            ),
        }
        for source, expected in cases.items():
            with self.subTest(source=source.strip()):
                with self.assertRaisesRegex(ParseError, expected):
                    parse_pfd(source + 'UNIT U : Heater\n')

    def test_explicit_ports_accept_semantic_short_aliases(self):
        pfd = parse_pfd(
            'STREAM Feed : -> F.i\n'
            'STREAM Vapor : F.v\n'
            'STREAM Liquid : F.l\n'
            'UNIT F\n'
            '    TYPE: Flash\n'
            '    PORTS:\n'
            '        feed_port : inlet\n'
            '        overhead_stream : vapor_outlet\n'
            '        residue_stream : liquid_outlet\n'
            '    PARAMS:\n'
        )

        self.assertEqual(
            [
                (stream.source.to_string(), stream.destination.to_string())
                for stream in pfd.streams
            ],
            [
                ('FEED', 'F.feed_port'),
                ('F.overhead_stream', 'PRODUCT'),
                ('F.residue_stream', 'PRODUCT'),
            ],
        )
        self.assertEqual(validate_pfd(pfd), ([], []))

    def test_explicit_port_names_remain_authoritative_even_if_alias_like(self):
        pfd = parse_pfd(
            'STREAM Feed : FEED -> U.bottoms\n'
            'STREAM Product : U.feed -> PRODUCT\n'
            'UNIT U\n'
            '    TYPE: Heater\n'
            '    PORTS:\n'
            '        bottoms : inlet\n'
            '        feed : outlet\n'
            '    PARAMS:\n'
            '        T = 50 [C]\n'
        )
        self.assertEqual(
            (
                pfd.streams[0].destination.port_id,
                pfd.streams[1].source.port_id,
            ),
            ('bottoms', 'feed'),
        )
        self.assertEqual(validate_pfd(pfd), ([], []))

    def test_structural_validation_rejects_duplicate_material_connections(self):
        duplicate_inlet = parse_pfd(
            'STREAM A : FEED -> H.in\n'
            'STREAM B : FEED -> H.in\n'
            'UNIT H : Heater\n'
        )
        errors, _ = validate_pfd(duplicate_inlet)
        self.assertIn(
            'Streams A and B both enter material inlet H.in',
            '\n'.join(errors),
        )

        duplicate_outlet = parse_pfd(
            'STREAM A : H.out -> PRODUCT\n'
            'STREAM B : H.out -> PRODUCT\n'
            'UNIT H : Heater\n'
        )
        errors, _ = validate_pfd(duplicate_outlet)
        self.assertIn(
            'Streams A and B both leave material outlet H.out',
            '\n'.join(errors),
        )

    def test_wrong_direction_fuzzy_and_ambiguous_compact_names(self):
        with self.assertRaisesRegex(
            ParseError,
            r"Port alias 'bottoms'.*outlet-only.*stream destination",
        ):
            parse_pfd(
                'STREAM ImpossibleFeed : -> D.bottoms\n'
                'UNIT D : RigorousDistillation\n'
            )

        typo = parse_pfd(
            'STREAM Typo : F.vappr\n'
            'UNIT F : Flash\n'
        )
        self.assertEqual(typo.streams[0].source.to_string(), 'F.vapor_out')

        with self.assertRaisesRegex(
            ParseError,
            r"Ambiguous Flash outlet port name 'product'.*liquid_out.*vapor_out",
        ):
            parse_pfd(
                'STREAM Ambiguous : F.product\n'
                'UNIT F : Flash\n'
            )

        with self.assertRaisesRegex(
            ParseError,
            r"Unknown unit type 'Flsh'\. Did you mean 'Flash'\?",
        ):
            parse_pfd('UNIT F : Flsh\n')

        with self.assertRaisesRegex(
            ParseError,
            r"Unknown unit type 'Flsh'\. Did you mean 'Flash'\?",
        ):
            parse_pfd('UNIT F\n    TYPE: Flsh\n')

    def test_compact_heater_runs_end_to_end(self):
        simulator = Simulator.from_string(
            'PROCESS: Compact heater\n'
            'ONLINE_LOOKUP: false\n'
            'COMPONENTS:\n'
            '    H2O | Water | MW=18.015, Cp_coeffs=[33.6, 0, 0, 0]\n'
            'STREAM Feed : -> Heater.i\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = H2O:1\n'
            'STREAM Product : Heater.o\n'
            'UNIT Heater : Heater\n'
            '    T = 50 [C]\n'
        )

        result = simulator.run()

        self.assertTrue(result.converged, result.errors)
        self.assertAlmostEqual(result.streams['Product'].T, 323.15, places=10)

    def test_numeric_mixer_and_splitter_run_end_to_end(self):
        simulator = Simulator.from_string(
            'PROCESS: Numeric ports\n'
            'ONLINE_LOOKUP: false\n'
            'COMPONENTS:\n'
            '    H2O | Water | MW=18.015, Cp_coeffs=[33.6, 0, 0, 0]\n'
            'STREAM A : -> M.1\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 1 [kmol/h]\n'
            '    x = H2O:1\n'
            'STREAM B : -> M.2\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 2 [kmol/h]\n'
            '    x = H2O:1\n'
            'STREAM C : -> M.3\n'
            '    T = 25 [C]\n'
            '    P = 1 [bar]\n'
            '    F = 3 [kmol/h]\n'
            '    x = H2O:1\n'
            'STREAM Mixed : M.4 -> S.1\n'
            'STREAM First : S.2\n'
            'STREAM Second : S.3\n'
            'UNIT M : Mixer\n'
            'UNIT S : Splitter\n'
            '    split_frac = 0.25\n'
        )

        result = simulator.run()

        self.assertTrue(result.converged, result.errors)
        self.assertAlmostEqual(result.streams['Mixed'].F, 6.0)
        self.assertAlmostEqual(result.streams['First'].F, 1.5)
        self.assertAlmostEqual(result.streams['Second'].F, 4.5)

    def test_pfr_uses_canonical_unit_and_port_names(self):
        simulator = Simulator.from_string(
            'PROCESS: Canonical PFR names\n'
            'ONLINE_LOOKUP: false\n'
            'THERMO_METHOD: UNIFNIST\n'
            'COMPONENTS:\n'
            '    dichloromethane | Dichloromethane | MW=84.93258\n'
            '    water | Water | MW=18.01528\n'
            'STREAM Wet : -> MS.f\n'
            '    T = 38.8 [C]\n'
            '    P = 1.01325 [bar]\n'
            '    F = 11.948982338770843 [kmol/h]\n'
            '    x = dichloromethane:0.9814181882647248, '
            'water:0.018581811735275185\n'
            'STREAM Dry : MS.p\n'
            'STREAM Water : MS.a\n'
            'UNIT MS : Dryer\n'
            '    sieve_type = 3A\n'
            '    adsorbent_mass_flow = 30 [kg/h]\n'
        )
        result = simulator.run()
        report = simulator._generate_pfr()

        self.assertTrue(result.converged, result.errors)
        self.assertEqual(simulator.pfd.units[0].unit_type, 'MolecularSieveDryer')
        for expected in (
            'source = FEED',
            'destination = MS.feed',
            'source = MS.product',
            'source = MS.adsorbate',
            'type = MolecularSieveDryer',
        ):
            self.assertIn(expected, report)
        self.assertNotIn('type = Dryer', report)
        self.assertNotIn('MS.f\n', report)
        self.assertNotIn('MS.p\n', report)
        self.assertNotIn('MS.a\n', report)


if __name__ == '__main__':
    unittest.main()
