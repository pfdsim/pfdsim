"""Profile high-level costs in the complete Haber-Bosch example.

Historical performance reference::

    | Metric                     | Original | Compiled ``k_ij`` | Compact direct PH + secant | Temperature-only PFR PH |
    |----------------------------|---------:|-------------------:|----------------------------:|------------------------:|
    | Full state constructions   |  178,010 |            178,010 |                      23,200 |               **6,408** |
    | Mixture enthalpy calls     |  179,693 |            179,693 |                      24,883 |               **8,091** |
    | Cpig enthalpy calls        |  898,407 |            898,407 |                     124,465 |              **40,505** |
    | Departure enthalpy calls   |  305,784 |            305,784 |                      28,198 |              **11,406** |
    | PFR time                   |   54.86 s |            40.65 s |                     13.23 s |              **9.01 s** |
    | Total time                 |   59.22 s |            45.06 s |                     17.60 s |             **13.40 s** |
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from cubic_eos import CubicEOS
from equilibrium_stage_column import EquilibriumStageColumnMixin
from flowsheet_solver import FlowsheetSolver
from simulator import Simulator
from thermodynamics_models.base import IdealThermodynamics
from thermodynamics_models.eos import CubicEOSThermodynamics
from unit_operations_basic import Compressor, Expander
from unit_operations_distillation import RigorousDistillation
from unit_operations_reactors import KineticsPFR


def main() -> None:
    stats = {}

    def instrument(cls, name):
        original = getattr(cls, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                key = f"{cls.__name__}.{name}"
                calls, elapsed = stats.get(key, (0, 0.0))
                stats[key] = calls + 1, elapsed + time.perf_counter() - start

        setattr(cls, name, wrapped)

    for owner, method in (
        (FlowsheetSolver, "solve"),
        (RigorousDistillation, "solve"),
        (EquilibriumStageColumnMixin, "_sparse_newton_solve"),
        (KineticsPFR, "solve"),
        (Compressor, "solve"),
        (Expander, "solve"),
        (CubicEOSThermodynamics, "calculate_state"),
        (CubicEOSThermodynamics, "mixture_enthalpy"),
        (CubicEOSThermodynamics, "mixture_entropy"),
        (CubicEOSThermodynamics, "departure_enthalpy"),
        (CubicEOSThermodynamics, "departure_entropy"),
        (IdealThermodynamics, "enthalpy_ideal_gas"),
        (CubicEOS, "phi_phi_K_values"),
        (CubicEOS, "fugacity_coefficients"),
        (CubicEOS, "departure_enthalpy"),
        (CubicEOS, "departure_entropy"),
        (CubicEOS, "_compiled_kij_values"),
        (CubicEOS, "_pair_kij_values"),
    ):
        instrument(owner, method)

    simulator = Simulator.from_file(str(ROOT / "examples" / "haber_bosch_full.pfd"))
    simulator.initialize()
    stats.clear()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = simulator.run()
    payload = {
        "wall_seconds": time.perf_counter() - wall_start,
        "cpu_seconds": time.process_time() - cpu_start,
        "converged": result.converged,
        "errors": result.errors,
        "mass_balance_error": result.mass_balance_error,
        "compiled_backend": type(simulator.thermo.cubic._compiled_backend).__name__,
        "timings": {
            name: {"calls": calls, "inclusive_seconds": elapsed}
            for name, (calls, elapsed) in sorted(
                stats.items(), key=lambda item: item[1][1], reverse=True
            )
        },
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
