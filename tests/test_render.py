"""Topology and export checks for the standalone PFD renderer (no simulation)."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import render
from pfd_parser import Metadata, Port, PortReference, PortType, ProcessFlowDiagram, Stream, Unit, parse_pfd


def stream_routes(root):
    return {
        group.get('data-stream-id'): [tuple(map(float, point.split(',')))
                                      for point in group.get('data-route').split()]
        for group in root.iter(f'{{{render.SVG}}}g') if group.get('data-stream-id')
    }


@unittest.skipUnless(shutil.which('dot'), 'Graphviz is not installed')
class RenderTests(unittest.TestCase):
    def test_every_example_preserves_equipment_and_streams(self):
        for source in sorted((ROOT / 'examples').glob('*.pfd')):
            with self.subTest(example=source.name):
                pfd = parse_pfd(source.read_text())
                svg = render.render_svg(pfd)
                root = ET.fromstring(svg)
                groups = list(root.iter(f'{{{render.SVG}}}g'))
                self.assertCountEqual(
                    [group.get('data-unit-id') for group in groups if 'data-unit-id' in group.attrib],
                    [unit.id for unit in pfd.units],
                )
                self.assertCountEqual(
                    [group.get('data-stream-id') for group in groups if 'data-stream-id' in group.attrib],
                    [stream.id for stream in pfd.streams],
                )
                self.assertNotIn(render.BODY_COLOR, svg)
                self.assertNotIn(render.PORT_COLOR, svg)
                for name, points in stream_routes(root).items():
                    with self.subTest(stream=name):
                        self.assertTrue(all(a != b and (a[0] == b[0] or a[1] == b[1])
                                            for a, b in zip(points, points[1:])))
                for i, stream in enumerate(pfd.streams):
                    group = next(group for group in groups if group.get('id') == f'stream-{i}')
                    self.assertTrue(list(group.iter(f'{{{render.SVG}}}path')))
                    self.assertTrue(list(group.iter(f'{{{render.SVG}}}polygon')))  # arrowhead
                    self.assertIn(stream.id, ''.join(group.itertext()))

    def test_literal_labels_are_escaped_and_custom_unit_remains_visible(self):
        unit = Unit('U<&"\\N', 'Custom<&type', [Port('in', PortType.INLET)])
        stream = Stream('S<&"\\N', PortReference(None, None, is_feed=True),
                        PortReference(unit.id, 'in'))
        pfd = ProcessFlowDiagram(metadata=Metadata(process_name='A < B & C'),
                                 units=[unit], streams=[stream])
        root = ET.fromstring(render.render_svg(pfd, stream_table=True))
        text = ''.join(root.itertext())
        for value in (unit.id, unit.unit_type, stream.id, 'A < B & C', 'Unspecified'):
            self.assertIn(value, text)
        self.assertNotIn('script', {element.tag for element in root.iter()})

    def test_distinct_heat_exchanger_ports(self):
        pfd = parse_pfd('''PROCESS: Two circuits
UNIT HX : HeatExchanger
STREAM Hot : FEED -> HX.2
STREAM Cold : FEED -> HX.1
STREAM HotOut : HX.3 -> PRODUCT
STREAM ColdOut : HX.4 -> PRODUCT
''')
        svg = ET.fromstring(render.render_svg(pfd))
        equipment = next(g for g in svg.iter(f'{{{render.SVG}}}g') if g.get('id') == 'equipment-0')
        nozzle_group = equipment.find(f'{{{render.SVG}}}g')
        paths = nozzle_group.findall(f'{{{render.SVG}}}path')
        self.assertEqual(len(paths), 4)
        self.assertEqual(len({path.get('d') for path in paths}), 4)

    def test_empty_flowsheet(self):
        root = ET.fromstring(render.render_svg(ProcessFlowDiagram()))
        self.assertIn('Process flow diagram', ''.join(root.itertext()))

    def test_countercurrent_and_cocurrent_exchanger_connections(self):
        for pattern in ('countercurrent', 'cocurrent'):
            with self.subTest(pattern=pattern):
                pfd = parse_pfd(f'''UNIT HX : HeatExchanger
    flow_pattern = {pattern}
STREAM Hot : FEED -> HX.2
STREAM Cold : FEED -> HX.1
STREAM HotOut : HX.3 -> PRODUCT
STREAM ColdOut : HX.4 -> PRODUCT
''')
                routes = stream_routes(ET.fromstring(render.render_svg(pfd)))
                self.assertLess(routes['Hot'][-1][0], routes['HotOut'][0][0])
                self.assertLess(routes['Hot'][-1][1], routes['Cold'][-1][1])
                if pattern == 'countercurrent':
                    self.assertGreater(routes['Cold'][-1][0], routes['ColdOut'][0][0])
                else:
                    self.assertLess(routes['Cold'][-1][0], routes['ColdOut'][0][0])

    def test_countercurrent_contactor_terminals_ignore_declaration_order(self):
        for kind, first, second, upper, lower in (
            ('RigorousExtractor', 'solvent', 'feed', 'extract', 'raffinate'),
            ('Absorber', 'gas', 'liquid', 'gas_out', 'liquid_out'),
            ('RigorousStripper', 'gas', 'liquid', 'gas_out', 'liquid_out'),
        ):
            with self.subTest(kind=kind):
                pfd = parse_pfd(f'''UNIT COL : {kind}
STREAM LowerFeed : FEED -> COL.{first}
STREAM UpperFeed : FEED -> COL.{second}
STREAM Bottom : COL.{lower} -> PRODUCT
STREAM Top : COL.{upper} -> PRODUCT
''')
                routes = stream_routes(ET.fromstring(render.render_svg(pfd)))
                self.assertLess(routes['UpperFeed'][-1][1], routes['LowerFeed'][-1][1])
                self.assertLess(routes['Top'][0][1], routes['Bottom'][0][1])

    def test_liquefaction_routes_have_no_crossings_and_flow_back_through_cold_sides(self):
        pfd = parse_pfd((ROOT / 'examples/methane_claude_liquefaction_pr.pfd').read_text())
        root = ET.fromstring(render.render_svg(pfd))
        routes = stream_routes(root)
        # Independently count geometric crossings, including vertices on lines.
        names = list(routes)
        for index, name in enumerate(names):
            for other in names[:index]:
                for a, b in zip(routes[name], routes[name][1:]):
                    for c, d in zip(routes[other], routes[other][1:]):
                        if a[1] == b[1] and c[0] == d[0]:
                            self.assertFalse(min(a[0], b[0]) < c[0] < max(a[0], b[0])
                                             and min(c[1], d[1]) <= a[1] <= max(c[1], d[1]),
                                             (name, other))
                        if a[0] == b[0] and c[1] == d[1]:
                            self.assertFalse(min(c[0], d[0]) < a[0] < max(c[0], d[0])
                                             and min(a[1], b[1]) <= c[1] <= max(a[1], b[1]),
                                             (name, other))
        for name in ('Cold-Return-Mixed', 'Cold-Return-After-Cold-HX', 'Warm-Recycle'):
            self.assertLess(routes[name][-1][0], routes[name][0][0])

    def test_mirrored_return_preserves_arrow_and_readable_equipment_tag(self):
        pfd = parse_pfd((ROOT / 'examples/methane_claude_liquefaction_pr.pfd').read_text())
        raw = subprocess.run(['dot', '-Tsvg'], input=render._dot_source(pfd),
                             text=True, capture_output=True, check=True, timeout=30)
        root = ET.fromstring(raw.stdout)
        _, mirrored = render._route_layout(root, pfd, mirror_returns=True)
        self.assertGreater(mirrored, 0)
        render._replace_equipment(root, pfd)
        mixer = next(g for g in root.iter() if g.get('data-unit-id') == 'RETURN-MIX')
        self.assertEqual(mixer.get('data-mirrored'), 'true')
        self.assertIsNone(mixer.get('transform'))  # only the symbol is reflected
        routes = stream_routes(root)
        self.assertLess(routes['Cold-Return-Mixed'][1][0], routes['Cold-Return-Mixed'][0][0])
        self.assertLess(routes['JT-Vapor-Return'][-1][0], routes['JT-Vapor-Return'][-2][0])

    def test_file_exports_and_input_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / 'source.pfd'
            original = (ROOT / 'examples/simple_flash.pfd').read_bytes()
            source.write_bytes(original)
            output = render.render_file(source, stream_table=True)
            self.assertEqual(output, source.with_suffix('.svg'))
            text = ''.join(ET.parse(output).getroot().itertext())
            self.assertIn('T = 25 [C]', text)
            self.assertIn('not calculated', text)
            with self.assertRaisesRegex(render.RenderError, 'already exists'):
                render.render_file(source)
            self.assertEqual(render.render_file(source, force=True), output)
            with self.assertRaisesRegex(render.RenderError, 'input PFD'):
                render.render_file(source, source, force=True)
            link = directory / 'alias.svg'
            link.symlink_to(source)
            with self.assertRaisesRegex(render.RenderError, 'input PFD'):
                render.render_file(source, link, force=True)
            self.assertEqual(source.read_bytes(), original)

    def test_png_and_pdf_exports(self):
        try:
            import cairosvg  # noqa: F401
        except (ImportError, OSError):
            self.skipTest('CairoSVG/Cairo is not installed')
        with tempfile.TemporaryDirectory() as temporary:
            source = ROOT / 'examples/simple_flash.pfd'
            for suffix, signature in (('.pdf', b'%PDF-'), ('.png', b'\x89PNG\r\n\x1a\n')):
                output = Path(temporary) / f'export{suffix}'
                render.render_file(source, output)
                self.assertTrue(output.read_bytes().startswith(signature))

    def test_module_cli(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'cli.svg'
            completed = subprocess.run(
                [sys.executable, '-m', 'pfdsim.render',
                 str(ROOT / 'examples/simple_flash.pfd'), '-o', str(output)],
                cwd=ROOT.parent, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(output.is_file())


class RenderFailureTests(unittest.TestCase):
    def test_missing_graphviz_has_actionable_error(self):
        with patch.object(render.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(render.RenderError, 'Graphviz'):
                render.render_svg(ProcessFlowDiagram())

    def test_unknown_endpoint_and_reversed_ports(self):
        pfd = parse_pfd((ROOT / 'examples/simple_flash.pfd').read_text())
        pfd.streams[0].destination.unit_id = 'MISSING'
        with self.assertRaisesRegex(render.RenderError, 'invalid endpoint'):
            render.render_svg(pfd)
        pfd.streams[0].destination.unit_id = 'HEAT-1'
        pfd.streams[0].destination.port_id = 'missing'
        with self.assertRaisesRegex(render.RenderError, 'unknown port'):
            render.render_svg(pfd)
        pfd.streams[0].destination.port_id = 'out'
        with self.assertRaisesRegex(render.RenderError, 'wrong direction'):
            render.render_svg(pfd)

    def test_invalid_options_do_not_write_output(self):
        source = ROOT / 'examples/simple_flash.pfd'
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'diagram.svg'
            for dpi in (0, -1, float('nan'), float('inf')):
                with self.subTest(dpi=dpi), self.assertRaisesRegex(render.RenderError, 'DPI'):
                    render.render_file(source, output, dpi=dpi)
            with self.assertRaisesRegex(render.RenderError, 'extension'):
                render.render_file(source, output.with_suffix('.html'))
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
