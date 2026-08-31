# Archived kinetics engines

`kinetics.py` and `kinetics_reactor.py` are retained here as historical source
only. They are not imported, packaged as runtime modules, or supported APIs.

They were retired in August 2026 because they implemented two incompatible
reaction schemas, guessed kinetic and activation-energy units, approximated
liquid volumes, clipped negative inventories, and used fixed-point/fixed-step
reactor solvers.

Active implementations are:

- `reaction_models.py` for canonical stoichiometry and equilibrium extents;
- `kinetic_models.py` for explicit-unit kinetic reactions and homogeneous rate
  states;
- `pellet_models.py` for exact first-order, generalized power-law, and rigorous
  one-species spherical catalyst effectiveness factors;
- `pfr_models.py` for shared adaptive axial PFR/packed-bed integration; and
- `unit_operations_reactors.py` for registered conversion, equilibrium, CSTR,
  batch/semi-batch, PFR, and fixed-catalyst packed-bed unit operations.
