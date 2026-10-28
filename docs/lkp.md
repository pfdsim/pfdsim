# Lee-Kesler-Plocker (LKP)

Select the model with `THERMO_METHOD: LKP` (or `LEE-KESLER-PLOCKER`). `LKP`
also works in `THERMO_SCOPES`, the Python factory, and the web method picker.
The implementation provides liquid/vapor fugacities, compressibility, EOS
volumes, residual Gibbs energy, enthalpy, entropy, heat capacity, and phi-phi
TP flashes and bubble/dew calculations through the existing shared solvers.
`examples/methane_ethane_lkp.pfd` is a complete two-phase flash example.

```text
THERMO_METHOD: LKP

COMPONENTS:
    CH4 | Methane
    C2H6 | Ethane

INTERACTION_PARAMETERS:
    CH4/C2H6 | model=LKP, kij=1.052
```

The parameter is symmetric and uses `Tc_ij = kij * sqrt(Tc_i * Tc_j)`.
Its neutral value is **1**, unlike a cubic EOS's usual zero. Overrides must
be positive and finite. They support the normal scope/inheritance rules;
temperature-dependent user parameter laws and parameter fitting are not
part of this implementation. `k_ij` is an accepted spelling of `kij`.

Pure-component `Tc` (K), `Pc` (bar), `Vc` (cm3/mol), and `omega` come from
pfdsim's existing property resolution, including explicit `.pfd` overrides.
Unavailable or invalid critical data produce a descriptive error. The
Plocker mixture pressure is calculated from mixture critical volume and
`Zcm = 0.2905 - 0.085*omega_m`; it is not an average of component pressures.

## Data and parameter priority

1. Explicit LKP interaction override.
2. CAS-keyed DWSIM table: **142 distinct pairs**.
3. DWSIM's applicable missing-pair correlation.
4. Neutral `kij = 1`, with a warning identifying the unsupported pair.

Estimated pairs also generate warnings, and
`LKPThermodynamics.interaction_parameter_sources` identifies the value and
source for each component pair. Hydrocarbons are identified by their resolved
C/H-only molecular formulas; there is no petroleum-fraction characterizer.

The raw source is `data/source/lkp_ip.dat.txt`, pinned to DWSIM commit
`1abf72d1b6b41d3e9a8cc770d3cc4e8fc76e5766`. Run
`python3 scripts/build_lkp_parameters.py` to rebuild the runtime table.
The builder records the source URL, SHA-256, and **13 corrections** checked
visually against Table I, p. 327 of the original paper. These include mislabeled
pairs, missing decimal points, and incorrect numerical transcriptions; the
unchanged raw file and explicit errata preserve the audit trail.

The estimation correlations follow DWSIM's online technical guide:

- Hydrogen with a missing partner:
  `k = 0.2036 + 0.002246*Tc + 0.8364*omega + 1.2111*Vc**(1/3)`.
- A heavier hydrocarbon and a lighter hydrocarbon, nitrogen, or carbon monoxide,
  when `Tc2/Tc1 >= 1.3`:
  `k = 1 + 0.2802*ln(Tc2/Tc1)*ln(Vc2/Vc1) - 0.0561*ln(Tc2/Tc1)`.
- CO2 with a hydrocarbon having `Tc >= 304.21 K`:
  `k = 0.9276 + 0.10719*X + 0.04505*X**2`, with
  `X = ln(Tc/304.21)*ln(Vc/0.094)`. H2S uses this value minus `0.027`.

Here `Vc` is in **m3/kmol**, after conversion from the property database.
Correlations use nominal component properties and never replace table values
or explicit overrides.

## Equations and numerical choices

The simple and reference fluids use the standard Lee-Kesler BWR coefficients
from DWSIM. Their properties interpolate with `omega_m/0.3978`. Spinodal
bracketing locates mechanically stable volume roots; liquid selects the
smallest and vapor the largest. Both labels use the same root when only one
exists. No cubic-EOS approximation is substituted for the LKP EOS.
The interpolated mixture must also have a positive volume and positive
isothermal compressibility. Acentric extrapolation can violate these conditions
even when both fluid roots are stable; such states fail explicitly rather than
entering flashes or property reports.

Component log fugacities are composition derivatives of the same molar residual
Gibbs potential used for caloric properties. Hydrogen uses the Gunn-Chueh-
Prausnitz temperature-dependent effective criticals and `omega=0`, with its
temperature derivatives included in residual enthalpy. Entropy follows
`Sdep = (Hdep - Gdep)/T`. Heat capacity uses the shared constant-pressure
enthalpy derivative. These choices preserve Gibbs-Duhem, Gibbs-Helmholtz,
and the pressure/volume identity, including hydrogen mixtures.

For reported liquid density, this package follows pfdsim's existing EOS
convention of using resolver-backed liquid volumes. `molar_volume(...,
phase="liquid")` exposes the LKP EOS volume explicitly. Vapor density uses
the LKP EOS volume.

This first implementation uses Python/NumPy/SciPy with bounded state/root
caches; it has no compiled LKP kernel. It uses the shared phi-phi equilibrium
solver. The shared substitution, stability, and boundary-residual arithmetic
has one array implementation: cubic, PSRK, and MHV2 backends compile it with
their EOS callbacks, while LKP and custom callbacks interpret the same source.
Persistent native caches use fixed model entry points with numerical arguments.
The shared higher-order algorithms are inlined into those entry points, so
cache indexes do not retain weak dispatcher references or depend on a callback
retention limit. Fresh-process reload and signature growth are regression-tested.
Boundary finite-difference Jacobians batch composition perturbations; their
temperature column re-prepares temperature-dependent interaction parameters.
Initialization loads native specializations by signature without evaluating
an arbitrary fluid state. Final physical audits still use the property model's
public methods. K-value estimates outside coexistence can contain Wilson single-root
extrapolations for equipment initialization. Bubble/dew temperatures and
pressures solve component fugacity equality directly, require distinct
admissible phases, and reject extrapolated or homogeneous false boundaries.
Spinodal-informed seeds resolve narrow near-critical root intervals.
Wilson values only initialize TP flashes and phase-boundary solves. The flash updates the actual liquid
and vapor fugacity ratios and checks the final split's material and fugacity
balances. EOS feeds must contain only known components with finite,
nonnegative amounts and a positive total; invalid feeds fail explicitly.
Finite component amounts are normalized without overflowing their sum.
Validation and scaled summation avoid allocating NumPy temporaries for each
stream-state request.
It distinguishes physical liquid/vapor root branches, searches for
an admissible opposite phase when a trial collapses to the wrong branch, and
uses nonlinear refinement for slow composition iterations.
The shared solver reuses exact state evaluations, starts with full substitution
steps, and damps growing or slowly oscillating iterations. It retries an
invalid full step with damping and performs stability analysis directly for
endpoint initial guesses.
Pure fluids select the valid lower-Gibbs root without composition iteration.
Bounded flash-result caches retain independent composition dictionaries and
recompute when warning context is cleared. Successful pure-fluid boundary
results are reusable across initial guesses; mixture boundaries retain their
guess-specific searches. These optimizations retain all final physical audits.
A continuous single supercritical root uses the vapor inventory and EOS volume convention,
including compressed supercritical fluids; its phase label does not depend on
the Wilson seed or a numerical Gibbs-energy tie. Invalid trial branches are
discarded while other phases are searched. Unconverged calculations or states
with no valid feed root raise
`ThermodynamicsValidityError`; stream calculations do not substitute guessed
phases for these failures. Explicitly requesting a mechanically unstable phase
also raises that error, even when an unforced flash can select a valid liquid.
It is intended primarily for nonpolar/light-gas and hydrocarbon systems;
the presence of a tabulated polar pair does not establish broad accuracy for
polar mixtures. EOS flashes remain VLE-constrained: they do not solve LLE/VLLE.
When the constrained single liquid has detected negative Gibbs curvature, the
solver reports a warning that liquid instability has been retained because
EOS LLE/VLLE is unsupported. A negative sampled curvature proves local
instability; a nonnegative result does not establish global liquid stability.

## Sources

- [DWSIM parameter table (pinned)](https://github.com/DanWBR/dwsim/blob/1abf72d1b6b41d3e9a8cc770d3cc4e8fc76e5766/DWSIM.Thermodynamics/Assets/lkp_ip.dat).
- [DWSIM Lee-Kesler implementation (pinned)](https://github.com/DanWBR/dwsim/blob/1abf72d1b6b41d3e9a8cc770d3cc4e8fc76e5766/DWSIM.Thermodynamics/PropertyPackages/Models/LeeKeslerPlocker.vb).
- [DWSIM technical guide and estimation correlations](https://dwsim.org/help/main/70-lee-kesler-plocker.html).
- [Plocker, Knapp & Prausnitz (1978)](https://doi.org/10.1021/i260067a020),
  *Ind. Eng. Chem. Process Des. Dev.* **17**, 324–332. A reference PDF is
  retained in `data/reference/thermodynamic-models/`.

DWSIM source attribution: Daniel Wagner and DWSIM contributors, GPL v3 or
later. pfdsim remains under its existing AGPL-3.0-only license.
