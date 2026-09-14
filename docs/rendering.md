# Rendering process flow diagrams

From the project directory:

```bash
python render.py examples/simple_flash.pfd
python render.py examples/methanol_synthesis.pfd -o methanol.pdf
python render.py examples/simple_flash.pfd -o flash.png --dpi 200 --stream-table
```

The installed package also supports `python -m pfdsim.render`. Output defaults
to an SVG beside the input. Existing output files require `--force`; the input
file cannot be overwritten. SVG is scalable and can be viewed in a browser.

Install the Graphviz system package (`dot` must be on `PATH`). For PDF and PNG,
install the optional Python dependencies with `pip install '.[render]'` from
the repository, or `pip install 'pfdsim[render]'` for an installed distribution.
CairoSVG also requires the Cairo system library on platforms without it.

The Python API is `render_file(source, output=None, stream_table=False,
dpi=150, force=False)` (options after `output` are keyword-only), or
`render_svg(parsed_pfd, stream_table=False)` to obtain SVG text in memory.

## Drawing conventions

The drawing uses a white sheet, equipment tags and types, conventional vessel,
column, pump, compressor, valve, heat exchanger, reactor, and filter symbols,
labelled material streams with arrowheads, and separate feed/product boundaries.
Mixer and splitter junctions have dots; crossings without dots are unconnected.
Every connected equipment port is routed separately, including both heat
exchanger circuits. Unknown/custom equipment uses a labelled rectangular symbol.
Column internals and packed beds are schematic, not literal counts of stages.

The parser is the same one used by the simulator, including aliases and compact
numeric port syntax. Rendering never initializes thermodynamics or runs a
simulation. The optional stream table reports only specifications in the input;
it does not present unknown values as calculated results. Internal condenser,
reboiler, utility and control connections are not invented when absent from the
source flowsheet.

The relevant engineering standards are
[ISO 10628-1 (diagram content)](https://www.iso.org/standard/51840.html) and
[ISO 10628-2 (graphical symbols)](https://www.iso.org/standard/51841.html).
This is a conventional engineering schematic renderer, not a certified ISO
symbol library or a detailed piping and instrumentation diagram (P&ID).

Graphviz arranges equipment predominantly left to right. A separate rectilinear
router uses the exact port locations, avoids equipment and equipment labels,
and penalizes bends, crossings, and shared line segments. Stream labels are
placed alongside the routed lines. Small gaps distinguish unavoidable crossings.
This avoids Graphviz's native orthogonal router's
[limitations with ports and edge labels](https://graphviz.org/docs/attrs/splines/).

Countercurrent exchangers put their secondary-fluid inlet on the right and
outlet on the left; cocurrent exchangers retain parallel connections. Extractors,
absorbers, strippers, and distillation columns place terminal feeds and products
at their corresponding ends. Extractors use schematic stage order (feed/extract
at stage 1, solvent/raffinate at the opposite end), without inferring phase
densities. Intermediate feeds and unspecified port roles remain schematic.

For return paths, the renderer also tries horizontally mirrored equipment and
keeps that arrangement only if it improves the routing score (crossings, shared
segments, length, bends, and label conflicts). Equipment tags stay readable and
material-flow arrows retain their original direction.

Large flowsheets can produce wide drawings and may need manual drafting for a
particular paper size. Existing editor coordinates are not used.
