"""Render a PFDsim flowsheet as an engineering-style SVG, PDF, or PNG.

Usage: python render.py examples/simple_flash.pfd [-o diagram.svg]
       python -m pfdsim.render flowsheet.pfd -o diagram.pdf --stream-table

Graphviz's ``dot`` executable is required. PDF/PNG additionally require the
``render`` extra (CairoSVG). Rendering parses the source without simulating it.
See docs/rendering.md for symbol conventions and scope.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
import heapq
from html import escape
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Sequence
import xml.etree.ElementTree as ET

if __package__ and __package__.split('.', 1)[0] == 'pfdsim':
    from .pfd_parser import ParseError, ProcessFlowDiagram, Unit, parse_pfd
    from .unit_syntax import port_family_for_unit_type, port_schema_for_unit_type
else:
    from pfd_parser import ParseError, ProcessFlowDiagram, Unit, parse_pfd
    from unit_syntax import port_family_for_unit_type, port_schema_for_unit_type


SVG = 'http://www.w3.org/2000/svg'
# These temporary cell colors identify geometry in Graphviz's SVG. They are
# removed when equipment symbols and nozzle lines replace the layout cells.
BODY_COLOR = '#f0e1d2'
PORT_COLOR = '#d2e1f0'


class RenderError(Exception):
    """A flowsheet cannot be rendered or an optional renderer is unavailable."""


def _element(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    return ET.SubElement(parent, f'{{{SVG}}}{tag}', {
        key.replace('_', '-'): str(value) for key, value in attrs.items()
    })


def _quote(value: str) -> str:
    """Quote literal DOT text, including Graphviz's backslash substitutions."""
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') + '"'


def _connections(pfd: ProcessFlowDiagram):
    """Validate drawing topology without imposing simulation requirements."""
    units = {unit.id: unit for unit in pfd.units}
    if len(units) != len(pfd.units):
        raise RenderError('Duplicate equipment identifiers.')
    if len({stream.id for stream in pfd.streams}) != len(pfd.streams):
        raise RenderError('Duplicate stream identifiers.')
    connected = defaultdict(lambda: {'in': [], 'out': []})
    directions = {}
    for unit in pfd.units:
        if len({port.id for port in unit.ports}) != len(unit.ports):
            raise RenderError(f'Duplicate port identifiers on {unit.id}.')
    for stream in pfd.streams:
        for ref, side in ((stream.source, 'out'), (stream.destination, 'in')):
            if (side == 'out' and ref.is_feed) or (side == 'in' and ref.is_product):
                if ref.unit_id is not None or ref.port_id is not None:
                    raise RenderError(f'Invalid boundary on stream {stream.id}.')
                continue
            unit = units.get(ref.unit_id)
            if ref.is_feed or ref.is_product or unit is None:
                raise RenderError(f'Stream {stream.id}: invalid endpoint {ref.to_string()}.')
            port = unit.get_port(ref.port_id)
            if port is None:
                raise RenderError(f'Stream {stream.id}: unknown port {ref.to_string()}.')
            expected = 'in' if port.port_type.value.endswith('inlet') else 'out'
            if side != expected:
                raise RenderError(f'Stream {stream.id}: wrong direction at {ref.to_string()}.')
            key = (unit.id, port.id)
            if key in directions:
                raise RenderError(f'Multiple streams connected to {ref.to_string()}.')
            directions[key] = side
            connected[unit.id][side].append(port.id)
    # Preserve declaration order, including the parser's normalized numeric ports.
    for unit in pfd.units:
        for side in ('in', 'out'):
            connected[unit.id][side].sort(
                key=lambda name: next(i for i, port in enumerate(unit.ports) if port.id == name)
            )
    return connected


def _port_role(unit: Unit, name: str) -> str:
    schema = port_schema_for_unit_type(unit.unit_type) or {}
    port = unit.get_port(name)
    direction = 'inlets' if port.port_type.value.endswith('inlet') else 'outlets'
    normalized = name.lower().replace('-', '_').replace(' ', '_')
    return schema.get(direction, {}).get(normalized, port.port_type.value)


def _port_level(unit: Unit, name: str) -> float:
    """Schematic terminal elevations, with stage 1 at the top of contactors."""
    family = port_family_for_unit_type(unit.unit_type)
    role = _port_role(unit, name)
    if family == 'heat_exchanger':
        return 0.3 if role.startswith(('hot_', 'shell_')) else 0.7
    if family == 'extractor':
        return {'feed': 0.2, 'extract': 0.2, 'extract_outlet': 0.2,
                'solvent': 0.8, 'solvent_inlet': 0.8,
                'raffinate': 0.8, 'raffinate_outlet': 0.8}.get(role, 0.5)
    if family in {'absorber', 'stripper', 'distillation', 'flash', 'flash3', 'decanter'}:
        return {'liquid': 0.2, 'solvent_inlet': 0.2,
                'gas_out': 0.2, 'vapor_out': 0.2, 'vapor_outlet': 0.2,
                'distillate': 0.2, 'distillate_vapor': 0.2, 'distillate_liquid': 0.2,
                'light': 0.3, 'liquid1_out': 0.65, 'liquid2_out': 0.8,
                'gas': 0.8, 'gas_inlet': 0.8, 'liquid_out': 0.8,
                'liquid_outlet': 0.8, 'bottoms': 0.8, 'heavy': 0.8}.get(role, 0.5)
    return 0.5


def _port_banks(pfd: ProcessFlowDiagram) -> dict:
    """Separate port direction from its physical side on the drawing."""
    connected = _connections(pfd)
    banks = {}
    for unit in pfd.units:
        banks[unit.id] = {'w': list(connected[unit.id]['in']),
                          'e': list(connected[unit.id]['out'])}
        if port_family_for_unit_type(unit.unit_type) != 'heat_exchanger':
            continue
        params = {param.name: str(param.value).strip().lower().replace('-', '_')
                  for param in unit.params}
        pattern = params.get('flow_pattern', params.get('type', 'countercurrent'))
        if pattern in {'cocurrent', 'co_current', 'parallel', 'parallel_flow'}:
            continue
        for direction, old_side, new_side in (('inlets', 'w', 'e'), ('outlets', 'e', 'w')):
            for name in connected[unit.id]['in' if direction == 'inlets' else 'out']:
                canonical = _port_role(unit, name)
                if canonical.startswith(('cold_', 'tube_')):
                    banks[unit.id][old_side].remove(name)
                    banks[unit.id][new_side].append(name)
    for unit in pfd.units:
        for names in banks[unit.id].values():
            names.sort(key=lambda name: _port_level(unit, name))
    return banks


def _unit_label(unit: Unit, ports: dict, port_ids: dict) -> str:
    family = port_family_for_unit_type(unit.unit_type)
    tall = family in {'distillation', 'absorber', 'stripper', 'extractor', 'dryer'}
    height = max(150 if tall else 100, 28 * max(map(len, ports.values()), default=0) + 40)

    def bank(side: str) -> str:
        names = ports[side]
        if not names:
            return '<TD WIDTH="14"></TD>'
        levels = [_port_level(unit, name) for name in names]
        rows, previous = [], 0
        for i, (name, level) in enumerate(zip(names, levels)):
            same_level = [j for j, value in enumerate(levels) if value == level]
            offset = (same_level.index(i) - (len(same_level)-1)/2) * 24
            y = level * height + offset
            gap = max(1, round(y - 3 - previous))
            rows.append(f'<TR><TD HEIGHT="{gap}"></TD></TR>')
            rows.append(
                f'<TR><TD PORT="{port_ids[name]}" COLOR="{PORT_COLOR}" BORDER="1" '
                'WIDTH="14" HEIGHT="6" FIXEDSIZE="TRUE"><FONT POINT-SIZE="1"> </FONT></TD></TR>'
            )
            previous = y + 3
        rows.append(f'<TR><TD HEIGHT="{max(1, round(height-previous))}"></TD></TR>')
        return (
            '<TD><TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" '
            f'CELLPADDING="0">{"".join(rows)}</TABLE></TD>'
        )

    return (
        '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        f'<TR><TD COLSPAN="3"><B>{escape(unit.id)}</B></TD></TR>'
        f'<TR><TD COLSPAN="3"><FONT POINT-SIZE="10">{escape(unit.unit_type)}</FONT></TD></TR>'
        '<TR>' + bank('w')
        + f'<TD BGCOLOR="{BODY_COLOR}" WIDTH="100" HEIGHT="{height}" FIXEDSIZE="TRUE"> </TD>'
        + bank('e') + '</TR></TABLE>'
    )


def _dot_source(pfd: ProcessFlowDiagram) -> str:
    connected = _port_banks(pfd)
    lines = [
        'digraph flowsheet {',
        'graph [rankdir=LR, splines=polyline, nodesep=0.65, ranksep=0.9, '
        'pad=0.3, bgcolor="white", outputorder=edgesfirst];',
        'node [shape=plain, fontname="DejaVu Sans", fontsize=12];',
        'edge [fontname="DejaVu Sans", fontsize=10, penwidth=1.5, arrowsize=0.7];',
    ]
    node_ids = {unit.id: f'u{i}' for i, unit in enumerate(pfd.units)}
    port_ids = {
        unit.id: {port.id: f'p{i}' for i, port in enumerate(unit.ports)}
        for unit in pfd.units
    }
    for i, unit in enumerate(pfd.units):
        label = _unit_label(unit, connected[unit.id], port_ids[unit.id])
        lines.append(f'u{i} [id="equipment-{i}", label=<{label}>];')
    for i, stream in enumerate(pfd.streams):
        endpoints = []
        source_side = ('w' if stream.source.unit_id and stream.source.port_id
                       in connected[stream.source.unit_id]['w'] else 'e')
        destination_side = ('e' if stream.destination.unit_id and stream.destination.port_id
                            in connected[stream.destination.unit_id]['e'] else 'w')
        backward = source_side == 'w' or destination_side == 'e'
        for ref, side in ((stream.source, source_side), (stream.destination, destination_side)):
            if ref.is_feed or ref.is_product:
                kind = 'feed' if ref.is_feed else 'product'
                name = f'{kind}{i}'
                lines.append(
                    f'{name} [id="{kind}-{i}", shape={"larrow" if backward else "rarrow"}, '
                    f'label="{kind.upper()}", fontsize=9, width=0.7, height=0.32];'
                )
                boundary_side = ('w' if backward else 'e') if ref.is_feed else ('e' if backward else 'w')
                endpoints.append(f'{name}:{boundary_side}')
            else:
                endpoints.append(f'{node_ids[ref.unit_id]}:{port_ids[ref.unit_id][ref.port_id]}:{side}')
        tooltip = f'{stream.id}: {stream.source.to_string()} -> {stream.destination.to_string()}'
        attributes = ''
        if backward:
            # Rank cold-side/return connections in their geometric direction;
            # dir=back preserves the actual material-flow arrow.
            endpoints.reverse()
            attributes = ', dir=back'
            if stream.source.unit_id and stream.destination.unit_id:
                attributes += ', constraint=false'
        lines.append(
            f'{endpoints[0]} -> {endpoints[1]} [id="stream-{i}", '
            f'label={_quote(stream.id)}, tooltip={_quote(tooltip)}{attributes}];'
        )
    lines.append('}')
    return '\n'.join(lines)


def _bounds(polygon: ET.Element) -> tuple[float, float, float, float]:
    points = [tuple(map(float, point.split(','))) for point in polygon.attrib['points'].split()]
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


def _intersects(a, b, box) -> bool:
    """Whether an axis-aligned segment crosses a rectangle's interior."""
    x0, y0, x1, y1 = box
    if a[1] == b[1]:
        return y0 < a[1] < y1 and max(a[0], b[0]) > x0 and min(a[0], b[0]) < x1
    return x0 < a[0] < x1 and max(a[1], b[1]) > y0 and min(a[1], b[1]) < y1


def _simplify(points):
    result = []
    for point in points:
        if result and point == result[-1]:
            continue
        if len(result) >= 2 and (
            result[-2][0] == result[-1][0] == point[0]
            or result[-2][1] == result[-1][1] == point[1]
        ):
            result.pop()
        result.append(point)
    return result


def _route_layout(root: ET.Element, pfd: ProcessFlowDiagram, *, mirror_returns: bool) -> tuple[float, int]:
    """Route on a rectilinear visibility grid, independently of dot's splines.

    Graphviz places equipment and reserves label space. Routing uses its exact
    port-cell coordinates, with equipment/labels as obstacles. Distance, bends,
    crossings and shared segments are penalized, so returns prefer clear lanes.
    """
    groups = {group.get('id'): group for group in root.iter(f'{{{SVG}}}g')}
    banks = _port_banks(pfd)
    centers = {}
    for i, unit in enumerate(pfd.units):
        body = next(p for p in groups[f'equipment-{i}'].findall(f'{{{SVG}}}polygon')
                    if p.get('fill') == BODY_COLOR)
        x0, _, x1, _ = _bounds(body)
        centers[unit.id] = (x0+x1)/2
    mirror_families = {'mixer', 'splitter', 'pump', 'compressor', 'expander',
                       'valve', 'pipe', 'heater', 'cooler', 'reactor', 'batch_reactor', 'filter'}
    mirrored_count = 0
    for i, unit in enumerate(pfd.units):
        outgoing = [s for s in pfd.streams if s.source.unit_id == unit.id]
        if not mirror_returns or port_family_for_unit_type(unit.unit_type) not in mirror_families or not outgoing:
            continue
        if not all(s.destination.unit_id is not None
                   and centers[s.destination.unit_id] < centers[unit.id] for s in outgoing):
            continue
        group = groups[f'equipment-{i}']
        group.set('data-mirrored', 'true')
        mirrored_count += 1
        banks[unit.id]['w'], banks[unit.id]['e'] = banks[unit.id]['e'], banks[unit.id]['w']
        for polygon in group.findall(f'{{{SVG}}}polygon'):
            if polygon.get('stroke') == PORT_COLOR:
                reflected = [(2*centers[unit.id]-float(x), float(y))
                             for x, y in (point.split(',') for point in polygon.get('points').split())]
                polygon.set('points', ' '.join(f'{x},{y}' for x, y in reflected))
    anchors, boxes = {}, {}
    for i, unit in enumerate(pfd.units):
        group = groups[f'equipment-{i}']
        polygons = list(group.findall(f'{{{SVG}}}polygon'))
        bounds = [_bounds(polygon) for polygon in polygons]
        box = (min(b[0] for b in bounds), min(b[1] for b in bounds),
               max(b[2] for b in bounds), max(b[3] for b in bounds))
        # The HTML label spans the node width and sits above the symbol.
        ys = [float(text.get('y')) - float(text.get('font-size', '12'))
              for text in group.findall(f'{{{SVG}}}text')]
        boxes[unit.id] = (box[0], min([box[1], *ys]) - 4, box[2], box[3])
        cells = iter(sorted((p for p in polygons if p.get('stroke') == PORT_COLOR),
                            key=lambda p: (_bounds(p)[0], _bounds(p)[1])))
        for side in ('w', 'e'):
            for name in banks[unit.id][side]:
                x0, y0, x1, y1 = _bounds(next(cells))
                anchors[(unit.id, name)] = ((x0 if side == 'w' else x1, (y0+y1)/2), side, unit.id)
    for i, stream in enumerate(pfd.streams):
        for ref, kind in ((stream.source, 'feed'), (stream.destination, 'product')):
            if ref.unit_id is not None:
                continue
            key = f'{kind}-{i}'
            polygon = groups[key].find(f'{{{SVG}}}polygon')
            box = _bounds(polygon)
            boxes[key] = box
            other = stream.destination if kind == 'feed' else stream.source
            other_side = anchors[(other.unit_id, other.port_id)][1] if other.unit_id else ('w' if kind == 'feed' else 'e')
            side = ('e' if other_side == 'w' else 'w')
            anchors[(key, None)] = ((box[0] if side == 'w' else box[2], (box[1]+box[3])/2), side, key)

    def endpoint(ref, boundary):
        point, side, key = anchors[(ref.unit_id or boundary, ref.port_id)]
        box = boxes[key]
        stub = (box[0]-18 if side == 'w' else box[2]+18, point[1])
        return point, stub

    ends = [(endpoint(s.source, f'feed-{i}'), endpoint(s.destination, f'product-{i}'))
            for i, s in enumerate(pfd.streams)]
    if not ends:
        return 0, mirrored_count
    xs, ys = set(), set()
    for box in boxes.values():
        xs.update((box[0]-18, box[2]+18))
        ys.update((box[1]-24, box[3]+24))
    for pair in ends:
        for _, (x, y) in pair:
            xs.add(x)
            ys.update((y, y-20, y+20))
    xs.update((min(xs)-40, max(xs)+40))
    ys.update((min(ys)-40, max(ys)+40))
    xs, ys = sorted(xs), sorted(ys)
    xindices, yindices = {x: i for i, x in enumerate(xs)}, {y: i for i, y in enumerate(ys)}
    obstacles = [(a-6, b-6, c+6, d+6) for a, b, c, d in boxes.values()]
    routes, occupied = {}, []

    @lru_cache(maxsize=None)
    def neighbors(ix, iy):
        choices = []
        for nx, ny in ((ix-1, iy), (ix+1, iy), (ix, iy-1), (ix, iy+1)):
            if 0 <= nx < len(xs) and 0 <= ny < len(ys):
                a, b = (xs[ix], ys[iy]), (xs[nx], ys[ny])
                if not any(_intersects(a, b, box) for box in obstacles):
                    choices.append((nx, ny, int(ny != iy), abs(a[0]-b[0])+abs(a[1]-b[1])))
        return choices

    def route(start, finish):
        initial = (xindices[start[0]], yindices[start[1]], 0)
        target = (xindices[finish[0]], yindices[finish[1]])
        queue = [(0, 0, initial)]
        costs, previous = {initial: 0}, {}
        while queue:
            _, cost, state = heapq.heappop(queue)
            if cost != costs[state]:
                continue
            ix, iy, direction = state
            if (ix, iy) == target:
                points = []
                while state is not None:
                    points.append((xs[state[0]], ys[state[1]]))
                    state = previous.get(state)
                return list(reversed(points))
            for nx, ny, new_direction, distance in neighbors(ix, iy):
                a, b = (xs[ix], ys[iy]), (xs[nx], ys[ny])
                penalty = 18 * (direction != new_direction)
                for c, d in occupied:
                    if a[1] == b[1] == c[1] == d[1]:
                        penalty += 20 * max(0, min(max(a[0], b[0]), max(c[0], d[0]))
                                            - max(min(a[0], b[0]), min(c[0], d[0])))
                    elif a[0] == b[0] == c[0] == d[0]:
                        penalty += 20 * max(0, min(max(a[1], b[1]), max(c[1], d[1]))
                                            - max(min(a[1], b[1]), min(c[1], d[1])))
                    elif _intersects(a, b, (min(c[0], d[0])-1, min(c[1], d[1])-1,
                                            max(c[0], d[0])+1, max(c[1], d[1])+1)):
                        penalty += 90
                next_state = (nx, ny, new_direction)
                next_cost = cost + distance + penalty
                if next_cost < costs.get(next_state, math.inf):
                    costs[next_state] = next_cost
                    previous[next_state] = (ix, iy, direction)
                    heuristic = abs(b[0]-finish[0]) + abs(b[1]-finish[1])
                    heapq.heappush(queue, (next_cost+heuristic, next_cost, next_state))
        raise RenderError('No clear route between equipment ports.')

    # Reserve short local connections first; long returns take the outer lanes.
    order = sorted(range(len(ends)), key=lambda i: sum(
        abs(a-b) for a, b in zip(ends[i][0][1], ends[i][1][1])))
    for i in order:
        (source, start), (target, finish) = ends[i]
        points = _simplify([source, *route(start, finish), target])
        routes[i] = points
        occupied.extend(zip(points, points[1:]))

    label_boxes = []
    label_penalty = 0
    for i, stream in enumerate(pfd.streams):
        group = groups[f'stream-{i}']
        for child in list(group):
            group.remove(child)
        _element(group, 'title').text = f'{stream.id}: {stream.source.to_string()} → {stream.destination.to_string()}'
        points = routes[i]
        group.set('data-route', ' '.join(f'{x:g},{y:g}' for x, y in points))
        d = 'M ' + ' L '.join(f'{x:g},{y:g}' for x, y in points)
        # A small white gap at unavoidable crossings keeps streams distinct.
        _element(group, 'path', d=d, stroke='white', stroke_width=5, fill='none')
        _element(group, 'path', d=d, stroke='black', stroke_width=1.5, fill='none')
        x, y = points[-1]
        px, py = points[-2]
        dx, dy = (x-px), (y-py)
        length = math.hypot(dx, dy)
        dx, dy = dx/length, dy/length
        _element(group, 'polygon', points=f'{x},{y} {x-8*dx+3*dy},{y-8*dy-3*dx} {x-8*dx-3*dy},{y-8*dy+3*dx}', fill='black')
        width = max(24, len(stream.id)*6.2)
        candidates = []
        for a, b in zip(points, points[1:]):
            if a[1] != b[1]:
                continue
            for fraction in (0.5, 0.25, 0.75):
                cx = a[0] + (b[0]-a[0])*fraction
                for offset in (-8, 18, -24, 34):
                    cy = a[1] + offset
                    box = (cx-width/2-3, cy-11, cx+width/2+3, cy+3)
                    collisions = sum(box[0] < c and box[2] > a0 and box[1] < d0 and box[3] > b0
                                     for a0, b0, c, d0 in [*obstacles, *label_boxes])
                    crossings = sum(_intersects(c, d0, box) for c, d0 in occupied)
                    score = collisions*10000 + crossings*1000 + abs(offset) + max(0, width-abs(b[0]-a[0]))
                    candidates.append((score, cx, cy, box))
        label_score, cx, cy, box = min(candidates)
        label_penalty += label_score
        label_boxes.append(box)
        _element(group, 'text', x=cx, y=cy, text_anchor='middle', font_family='DejaVu Sans',
                 font_size=10, fill='black').text = stream.id

    all_boxes = [*boxes.values(), *label_boxes]
    all_points = [point for points in routes.values() for point in points]
    xmin = min([b[0] for b in all_boxes] + [p[0] for p in all_points])-24
    ymin = min([b[1] for b in all_boxes] + [p[1] for p in all_points])-24
    xmax = max([b[2] for b in all_boxes] + [p[0] for p in all_points])+24
    ymax = max([b[3] for b in all_boxes] + [p[1] for p in all_points])+24
    graph = next(g for g in groups.values() if g.get('class') == 'graph')
    for polygon in list(graph.findall(f'{{{SVG}}}polygon')):
        graph.remove(polygon)
    graph.set('transform', f'translate({-xmin} {-ymin})')
    root.set('viewBox', f'0 0 {xmax-xmin} {ymax-ymin}')
    crossings, overlap, length = 0, 0, 0
    for index, (a, b) in enumerate(occupied):
        length += abs(a[0]-b[0]) + abs(a[1]-b[1])
        for c, d in occupied[:index]:
            if a[1] == b[1] == c[1] == d[1]:
                overlap += max(0, min(max(a[0], b[0]), max(c[0], d[0]))
                               - max(min(a[0], b[0]), min(c[0], d[0])))
            elif a[0] == b[0] == c[0] == d[0]:
                overlap += max(0, min(max(a[1], b[1]), max(c[1], d[1]))
                               - max(min(a[1], b[1]), min(c[1], d[1])))
            elif a[1] == b[1] and c[0] == d[0]:
                crossings += min(a[0], b[0]) < c[0] < max(a[0], b[0]) and min(c[1], d[1]) < a[1] < max(c[1], d[1])
            elif a[0] == b[0] and c[1] == d[1]:
                crossings += min(c[0], d[0]) < a[0] < max(c[0], d[0]) and min(a[1], b[1]) < c[1] < max(a[1], b[1])
    quality = crossings*1000 + overlap*20 + length*0.05 + len(occupied)*4 + label_penalty
    root.set('data-route-crossings', str(crossings))
    return quality, mirrored_count


def _route_streams(root: ET.Element, pfd: ProcessFlowDiagram) -> ET.Element:
    """Keep a mirrored return layout only when it improves drawing quality."""
    candidate = deepcopy(root)
    mirrored_quality, mirrored_count = _route_layout(candidate, pfd, mirror_returns=True)
    if not mirrored_count:
        return candidate
    normal_quality, _ = _route_layout(root, pfd, mirror_returns=False)
    return candidate if mirrored_quality < normal_quality else root


def _symbol(parent: ET.Element, unit: Unit) -> None:
    """Draw in a 100 x 100 coordinate system; nozzle lines sit underneath."""
    family = port_family_for_unit_type(unit.unit_type)

    def path(d: str, fill: str = 'none'):
        return _element(parent, 'path', d=d, fill=fill)

    def vessel(x=25, y=8, width=50, height=84):
        _element(parent, 'rect', x=x, y=y, width=width, height=height,
                 rx=width / 2, ry=10, fill='white')

    if family in {'mixer', 'splitter'}:
        _element(parent, 'circle', cx=50, cy=50, r=3, fill='black')
    elif family == 'valve':
        path('M 15,32 L 85,68 L 85,32 L 15,68 Z', 'white')
    elif family == 'pump':
        _element(parent, 'circle', cx=50, cy=50, r=32, fill='white')
        path('M 30,25 L 77,50 L 30,75 Z', 'white')
        path('M 32,77 L 25,88 L 75,88 L 68,77')
    elif family in {'compressor', 'expander'}:
        d = ('M 15,20 L 85,35 L 85,65 L 15,80 Z' if family == 'compressor'
             else 'M 15,35 L 85,20 L 85,80 L 15,65 Z')
        path(d, 'white')
    elif family in {'heater', 'cooler', 'heat_exchanger'}:
        _element(parent, 'circle', cx=50, cy=50, r=40, fill='white')
        path('M 10,50 L 28,50 L 37,30 L 47,70 L 57,30 L 67,70 L 76,50 L 90,50')
        if family in {'heater', 'cooler'}:
            label = _element(parent, 'text', x=50, y=21, text_anchor='middle',
                             font_size=12, stroke='none', fill='black')
            label.text = '+' if family == 'heater' else '−'
    elif family == 'pipe':
        path('M 0,46 L 100,46 M 0,54 L 100,54 M 5,42 L 5,58 M 95,42 L 95,58')
    elif family == 'filter':
        path('M 15,20 L 85,20 L 85,80 L 15,80 Z', 'white')
        path('M 15,65 L 85,35 M 15,69 L 85,39')
    elif family == 'decanter':
        vessel(8, 22, 84, 56)
        path('M 10,52 L 90,52')
    elif family in {'distillation', 'absorber', 'stripper', 'extractor', 'dryer'}:
        vessel()
        if family == 'dryer':
            path('M 26,25 L 74,25 L 74,75 L 26,75 Z')
            for y in range(30, 76, 10):
                path(f'M 27,{y} L 73,{y - 5}')
        else:
            for y in range(25, 81, 11):
                path(f'M 27,{y} L 73,{y}')
    elif family in {'flash', 'flash3'}:
        vessel()
        path('M 26,62 L 74,62')
        if family == 'flash3':
            path('M 26,74 L 74,74')
    elif unit.unit_type in {'PFR', 'PackedBedReactor'}:
        vessel(12, 22, 76, 56)
        if unit.unit_type == 'PackedBedReactor':
            for x in range(25, 81, 10):
                path(f'M {x},26 L {x - 10},74')
        else:
            path('M 17,42 L 83,42 M 17,58 L 83,58')
    elif family in {'reactor', 'batch_reactor', 'crystallizer', 'layer_crystallizer'}:
        vessel(20, 12, 60, 76)
        if unit.unit_type in {'CSTR', 'BatchReactor'} or family == 'crystallizer':
            path('M 50,3 L 50,68 M 35,62 L 65,72 M 35,72 L 65,62')
        elif family == 'layer_crystallizer':
            path('M 32,25 L 32,76 M 40,25 L 40,76 M 60,25 L 60,76 M 68,25 L 68,76')
    else:
        # Future/custom equipment remains visible and explicitly type-labelled.
        _element(parent, 'rect', x=12, y=12, width=76, height=76, fill='white')


def _replace_equipment(root: ET.Element, pfd: ProcessFlowDiagram) -> None:
    groups = {group.get('id'): group for group in root.iter(f'{{{SVG}}}g')}
    for i, unit in enumerate(pfd.units):
        group = groups[f'equipment-{i}']
        group.set('data-unit-id', unit.id)
        group.set('data-unit-type', unit.unit_type)
        polygons = list(group.findall(f'{{{SVG}}}polygon'))
        body = next(p for p in polygons if p.get('fill') == BODY_COLOR)
        x0, y0, x1, y1 = _bounds(body)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        nozzles = _element(group, 'g', stroke='black', stroke_width=1.5, fill='none')
        junction = port_family_for_unit_type(unit.unit_type) in {'mixer', 'splitter'}
        for polygon in polygons:
            if polygon.get('stroke') != PORT_COLOR:
                continue
            px0, py0, px1, py1 = _bounds(polygon)
            px = px0 if (px0 + px1) / 2 < cx else px1
            py = (py0 + py1) / 2
            _element(nozzles, 'path', d=f'M {px},{py} L {cx},{cy if junction else py}')
            group.remove(polygon)
        group.remove(body)
        mirrored = group.get('data-mirrored') == 'true'
        scale_x = (x1-x0)/100 * (-1 if mirrored else 1)
        symbol = _element(group, 'g', transform=f'translate({x1 if mirrored else x0} {y0}) scale({scale_x} {(y1-y0)/100})',
                          stroke='black', stroke_width=1.5, stroke_linejoin='round', fill='none')
        _symbol(symbol, unit)
        title = group.find(f'{{{SVG}}}title')
        if title is not None:
            title.text = f'{unit.id} ({unit.unit_type})'
    for i, stream in enumerate(pfd.streams):
        groups[f'stream-{i}'].set('data-stream-id', stream.id)


def _sheet(diagram: ET.Element, pfd: ProcessFlowDiagram, stream_table: bool) -> ET.Element:
    """Add a drawing border, title block, and optional specified-stream table."""
    _, _, graph_width, graph_height = map(float, diagram.get('viewBox').split())
    rows = []
    if stream_table:
        for stream in pfd.streams:
            conditions = '; '.join(prop.to_pfd().strip() for prop in stream.properties)
            if stream.composition is not None:
                conditions += ('; ' if conditions else '') + stream.composition.to_pfd().strip()
            rows.append((stream.id, stream.source.to_string(), stream.destination.to_string(),
                         conditions or 'Unspecified'))
    headers = ('Stream', 'From', 'To', 'Specified conditions (input; not calculated)')
    widths = [max(len(row[col]) for row in [headers, *rows]) * 6.5 + 24 for col in range(4)]
    title = pfd.metadata.process_name or 'Process flow diagram'
    width = max(graph_width + 40, sum(widths) + 40 if rows else 640, len(title) * 10 + 40)
    table_height = (len(rows) + 2) * 25 + 12 if rows else 0
    height = graph_height + 120 + table_height
    root = ET.Element(f'{{{SVG}}}svg', {
        'version': '1.1', 'viewBox': f'0 0 {width:g} {height:g}',
        'width': f'{width:g}pt', 'height': f'{height:g}pt', 'role': 'img',
        'aria-label': title,
    })
    _element(root, 'title').text = title
    _element(root, 'desc').text = (
        'Process flow diagram. Solid directed lines are material streams; '
        'crossings without a junction dot are not connections. '
        'Equipment and connections come from the PFD input; no simulation was run.'
    )
    _element(root, 'rect', width=width, height=height, fill='white')
    _element(root, 'rect', x=10, y=10, width=width-20, height=height-20,
             fill='none', stroke='black', stroke_width=1)
    text_group = _element(root, 'g', font_family='DejaVu Sans, sans-serif', fill='black')
    _element(text_group, 'text', x=24, y=38, font_size=18, font_weight='bold').text = title
    _element(text_group, 'text', x=24, y=58, font_size=10).text = (
        f'PROCESS FLOW DIAGRAM  |  Rev {pfd.metadata.version}  |  {pfd.metadata.thermo_method}'
    )
    diagram.set('x', str((width - graph_width) / 2))
    diagram.set('y', '70')
    diagram.set('width', str(graph_width))
    diagram.set('height', str(graph_height))
    root.append(diagram)
    table_y = graph_height + 88
    if rows:
        for index, row in enumerate([headers, *rows]):
            y = table_y + index * 25
            _element(root, 'path', d=f'M 24,{y + 6} H {width - 24}', stroke='#aaaaaa', stroke_width=0.5)
            x = 24
            for col, value in enumerate(row):
                _element(text_group, 'text', x=x, y=y, font_size=10,
                         font_weight='bold' if index == 0 else 'normal').text = value
                x += widths[col]
    _element(text_group, 'text', x=24, y=height-22, font_size=9).text = (
        'Material streams →  |  Crossings are unconnected unless marked with a dot  |  '
        'Schematic · not to scale'
    )
    return root


def render_svg(pfd: ProcessFlowDiagram, *, stream_table: bool = False) -> str:
    """Return a standalone SVG using the parsed PFD's exact stream topology."""
    source = _dot_source(pfd)
    executable = shutil.which('dot')
    if executable is None:
        raise RenderError('Graphviz is required: install Graphviz and put dot on PATH.')
    try:
        result = subprocess.run([executable, '-Tsvg'], input=source, text=True,
                                capture_output=True, check=True, timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise RenderError('Graphviz layout exceeded 30 seconds.') from exc
    except subprocess.CalledProcessError as exc:
        raise RenderError(f'Graphviz failed: {exc.stderr.strip()}') from exc
    try:
        diagram = ET.fromstring(result.stdout)
        diagram = _route_streams(diagram, pfd)
        _replace_equipment(diagram, pfd)
        root = _sheet(diagram, pfd, stream_table)
    except (ET.ParseError, KeyError, StopIteration, ValueError) as exc:
        raise RenderError('Graphviz returned an unsupported or invalid SVG layout.') from exc
    ET.register_namespace('', SVG)
    return ET.tostring(root, encoding='unicode', xml_declaration=True)


def render_file(source: str | Path, output: str | Path | None = None, *,
                stream_table: bool = False, dpi: float = 150, force: bool = False) -> Path:
    """Render a .pfd file, defaulting to a sibling .svg; never run the solver."""
    source = Path(source).expanduser()
    output = Path(output).expanduser() if output is not None else source.with_suffix('.svg')
    if output.resolve() == source.resolve() or (output.exists() and source.samefile(output)):
        raise RenderError('Output must not overwrite the input PFD file.')
    if output.suffix.lower() not in {'.svg', '.png', '.pdf'}:
        raise RenderError('Output extension must be .svg, .png, or .pdf.')
    if not math.isfinite(dpi) or dpi <= 0:
        raise RenderError('DPI must be a positive finite number.')
    if output.exists() and not force:
        raise RenderError(f'Output already exists: {output}; use --force to replace it.')
    svg = render_svg(parse_pfd(source.read_text(encoding='utf-8')), stream_table=stream_table)
    content = svg.encode('utf-8')
    if output.suffix.lower() != '.svg':
        try:
            import cairosvg
        except (ImportError, OSError) as exc:
            raise RenderError('PDF/PNG requires CairoSVG and Cairo: install pfdsim[render].') from exc
        convert = cairosvg.svg2png if output.suffix.lower() == '.png' else cairosvg.svg2pdf
        content = convert(bytestring=content, dpi=dpi)
    # Exclusive creation also protects a file created during layout/conversion.
    with output.open('wb' if force else 'xb') as handle:
        handle.write(content)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('file', type=Path, help='input .pfd file')
    parser.add_argument('-o', '--output', type=Path, help='output .svg (default), .pdf, or .png')
    parser.add_argument('--stream-table', action='store_true', help='include source stream specifications')
    parser.add_argument('--dpi', type=float, default=150, help='PNG resolution (default: 150)')
    parser.add_argument('--force', action='store_true', help='replace an existing output file')
    args = parser.parse_args(argv)
    try:
        output = render_file(args.file, args.output, stream_table=args.stream_table,
                             dpi=args.dpi, force=args.force)
    except (RenderError, ParseError, OSError, ValueError) as exc:
        print(f'render: {exc}', file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
