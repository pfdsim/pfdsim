import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from thermodynamics import create_thermodynamics
from unit_operations import RigorousLiquidLiquidExtractor

components = ["diethyl ether", "n-hexane", "acrylic acid", "water"]
thermo = create_thermodynamics(components, "UNIFAC")

def stream_from_mass(masses_kg_h):
    molar = {
        c: masses_kg_h.get(c, 0.0) / thermo.props[c].MW
        for c in components
    }
    total = sum(molar.values())
    composition = {c: n / total for c, n in molar.items() if n > 0.0}
    return thermo.calculate_state(
        298.15, 1.0, total, composition, phase="liquid", flash=False
    )

def component_kg_h(stream, comp):
    return stream.F * stream.composition.get(comp, 0.0) * thermo.props[comp].MW

feed = stream_from_mass({
    "acrylic acid": 200.0,
    "water": 1500.0,
})

solvent = stream_from_mass({
    "diethyl ether": 1000.0,
    "n-hexane": 1000.0,
})

result = RigorousLiquidLiquidExtractor(
    "EX-100",
    thermo,
    {"N_stages": 20, "T": 25, "max_iterations": 250},
).solve({"feed": feed, "solvent": solvent})

for name, stream in result.outlet_streams.items():
    print(name, "total kg/h:", stream.mass_flow())
    print({
        comp: component_kg_h(stream, comp)
        for comp in components
    })

print(result.performance)