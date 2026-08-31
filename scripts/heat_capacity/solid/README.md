# Solid heat-capacity and volume property foundation

This directory contains the offline canonical solid-property compilers and the
retained estimator audit. Runtime code consumes only the generated SQLite
artifacts; it does not import these scripts.

## Canonical artifacts

```text
data/solid_heat_capacity.sqlite
data/solid_volume.sqlite
```

The heat-capacity database contains multiple admitted native records per CAS
when temperature coverage or material form differs. It preserves crystalline,
glass/amorphous, allotrope/polymorph, hydrate, and solvate identity rather than
collapsing all solids to one curve.

Native solid-Cp kernels are:

```text
NIST WebBook Shomate
JANAF exact piecewise-linear tables
CRC exact piecewise-linear temperature tables
CRC narrow 298.15 K points
Perry 2-151 analytic correlations
```

Real solid-solid transitions are stored explicitly. Sensible enthalpy or
entropy integration across a transition fails until its transition enthalpy is
available; no smooth fit is allowed to erase the boundary.

Absolute-zero continuation is intentionally absent. JANAF's formal `(0 K,
0 Cp)` anchors are removed before constructing positive-temperature kernels,
Lastovka begins at 100 K, and other sources retain their native lower bounds.
Queries farther than 5 K below a source fail. A future Debye/Einstein model can
add the third-law limit when defensible characteristic temperatures exist.

The volume database contains 1,872 CRC constant solid molar volumes. Runtime
preserves native molar volume and derives density through molecular weight.
No universal inorganic thermal expansion coefficient is assumed; volume stays
constant and source quality decreases with distance from 298.15 K.

## Identity audit

CAS validation alone is insufficient. Perry 2-151 contains valid-CAS collisions
such as tungsten under L-tryptophan's CAS and uranium under uracil's CAS. The
compiler cross-checks source and resolved formulas and quarantines conflicts.
The CRC temperature-table graphite row is locally remapped from methane's CAS
to graphite CAS 7782-42-5 because its complete curve exactly matches JANAF
graphite. Diamond and graphite remain distinct identities.

## Estimators

Modified Kopp is valid only as a narrow 298.15 K estimate. Retained benchmarks:

```text
Strict organic CRC points:
    n=102, mean 9.27%, median 6.75%, P90 22.14%

Vetted broader Perry-formula/CRC subset:
    n=127, mean 8.37%, median 6.10%, P90 18.27%, within 10% 71.7%
```

It receives a lower quality tier for inorganic use. The crude universal
`3 R`-per-atom form was rejected because its inorganic median error was 19.4%
and P90 was 67.5%.

Lastovka-Fulem-Becerra-Shaw is admitted only inside its published MW and
elemental mass-fraction domain and only when a trustworthy `Tt` or `Tm` limits
the upper range. On 38 identity-confirmed CRC points it produced mean 6.63%,
median 6.38%, and P90 13.11% error. It remains one temperature-dependent curve;
scaling it to the modified-Kopp point worsened all three available organic
temperature-curve cases.

Reproduce the point benchmarks with:

```bash
python scripts/heat_capacity/solid/benchmark_solid_cp_estimators.py
```

## Runtime order

```text
1. Explicit .pfd Cps correlation
2. Explicit .pfd Cp_solid constant
3. Canonical solid_heat_capacity.sqlite
4. Non-.pfd Cp_solid reference point
5. Persisted/online NIST solid kernel
6. Strict-domain Lastovka estimator
7. Modified Kopp 298.15 K estimator
8. Failure
```

The separate permanent-solid process layer now consumes these artifacts for
explicitly declared nonmelting, nondissolving components. This directory still
contains only pure-component property builders; stream routing and unit
capability rules live in the thermodynamics and unit-operation layers.
