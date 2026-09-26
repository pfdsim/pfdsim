# Molecular sieve adsorption

`MolecularSieveDryer` (alias `Dryer`) represents one isothermal equilibrium
contact with fresh or regenerated sieve. All feed components with valid
isotherms participate automatically. There is no water-only removal list.
The existing 3A water GSTA coefficients and pure-component curve are retained.

Specify **exactly one** of:

- `adsorbent_mass_flow` in kg/h (the existing sieve-flow aliases also work).
  Explicit `kg/s`, `g/h`, and `lb/h` units use the shared mass-flow conversion.
- `target_mole_fraction` with `target_component`.
- `removal_fraction` with `target_component`, defined on that component's
  inlet molar flow, not its mole fraction.
- `target_water_mole_fraction`, which selects water automatically.

`target_component` defaults to the water component when present. A bare sieve
type no longer supplies an implicit water target. Sizing includes every
coadsorbate in both the capacity balance and the product mole-fraction
denominator. An already-satisfied target needs zero sieve. Exactly complete
removal at finite sieve flow is rejected.

```text
UNIT MS : MolecularSieveDryer
    sieve_type = 13X
    adsorbent_mass_flow = 100 [kg/h]
```

See [the three-gas example](../examples/competitive_13x_adsorption.pfd).
The [4A hydrocarbon example](../examples/propane_propylene_4a_adsorption.pfd)
contacts propane/propylene vapor containing CO2, methane and nitrogen with
fresh sieve. It illustrates impurity removal together with hydrocarbon
coadsorption, and explicitly identifies extrapolation of the C3 isotherms.

## Isotherms and competition

Pure isotherms use fugacity `f` in **bar**, temperature `T` in **K**, and loading
`q` in **mol/kg of dry sieve**. For a numerical value, mol/kg equals mmol/g.
The thermodynamic model supplies fugacities for vapor and liquid mixtures,
including its activity, EOS and liquid reference-state corrections. Two-phase
VLE feeds are flashed during the adsorption balance; their equilibrium phase
compositions supply fugacity. Separate liquid-liquid feeds before this unit;
formation of a second liquid phase during contact is also rejected.
Single-phase feeds retain their specified phase constraint.

Ideal adsorbed solution theory (IAST) solves a common spreading potential:

```text
Pi = integral_0^f0_i q_i(f,T) d(ln f)
x_i = f_i / f0_i
sum_i x_i = 1
1/q_total = sum_i x_i/q_i(f0_i,T)
q_mix_i = x_i*q_total
F_in_i - F_product_i = sieve_kg_h*(q_mix_i - q_initial_i)/1000
```

This satisfies the Gibbs adsorption equation and handles different pure
saturation capacities. Independent pure-species removal calculations or an
extended Langmuir denominator with unequal capacities do not generally have
this property. IAST assumes an ideal adsorbed mixture on a common accessible
adsorbent; it does not model nonideal adsorbed interactions, pore blocking,
reaction, hysteresis, diffusion, breakthrough or regeneration cycles.

References: [Myers and Prausnitz IAST, as implemented and described by the
pyIAST authors](https://doi.org/10.1016/j.cpc.2015.11.016), and
[limitations of IAST in microporous crystals](https://pmc.ncbi.nlm.nih.gov/articles/PMC8210411/).

## Built-in data and warnings

The offline catalogue is `data/molecular_sieve_isotherms.json`. It includes
33 selected NIST curves for 3A, 4A, 5A and 13X, with source identifiers, temperature
and fugacity ranges, fit errors and explicit exclusion reasons. Component
symbols, chemical names and CAS identities resolve to the same default.
The retained 3A water GSTA curve and a primary-literature 3A ammonia curve
bring the total to **35 sieve/substance pairs**.

| Sieve | Built-in substances |
|---|---|
| 3A (3) | Water (retained GSTA), neon (NIST, 77 K), ammonia (primary literature, 303.15 K) |
| 4A (7) | CO2, methane, nitrogen, H2S, carbonyl sulfide, propane, propylene |
| 5A (11) | CO2, methane, nitrogen, oxygen, CO, argon, helium, ethane, ethylene, propane, propylene |
| 13X (14) | CO2, methane, nitrogen, oxygen, water, argon, acetylene, ethane, ethylene, propane, propylene, isobutane, 1-butene, sulfur hexafluoride |

The NIST fits use 1,184 observations from the 90 retained source records; the
mean absolute relative fit errors range from 0.34% to 12.67%. These are fitting
errors against pure-component data, not validation errors for mixture
predictions. The inventory is not claimed to cover all of NIST. In particular,
the available 4A water record uses mol/m3 and lacks the required adsorbent
density conversion in its record; it is not assigned a guessed gravimetric
capacity. The reviewed inventory contained no pure 5A water curve under the
exact `Zeolite 5A` label. Missing water curves generate the same warning as
any other missing adsorbate curve.

The unmodified source records are saved in
[`data/source/molecular_sieve_nist.json`](../data/source/molecular_sieve_nist.json),
with a pinned commit of the [official NIST mirror](https://github.com/NIST-ISODB/isodb-library).
Rebuild the fitted catalogue offline with:

```bash
python scripts/build_molecular_sieve_isotherms.py
```

The builder uses pure experimental observations at pressures up to 1 bar,
converts pressure to pure PR fugacity, and compares Langmuir, dual-site
Langmuir, Toth and Sips curves. Every series with multiple temperatures is
fitted jointly with van't Hoff affinity dependence; fitted capacity and shape
parameters are temperature independent. Form selection uses interleaved
three-fold held-out predictions within each source curve, with leave-one-out
validation for the smallest datasets. Underidentified validation fits are
ineligible. Similar prediction errors (within 1% relative) prefer fewer
parameters, then a Henry-consistent form. Six deterministic affinity starts
include both the original builder and comparison starts, reducing local-minimum
bias. Candidate errors and the chosen validation method are retained per pair.
Records with absolute adsorption are used as such; otherwise the
reported low-pressure loading approximates absolute loading. That approximation
is warned about at runtime. No universal high-pressure accuracy is implied.
STP volume units use 273.15 K and 1 atm; wt% uses mass of adsorbate per dry
adsorbent mass. Raw records are not rewritten to hide their original units.

Different publications may describe different batches, binders and activation
conditions even under the same sieve name. Defaults are estimates for that
material class, not universal material constants. Source curves with missing
conversion information, unavailable pure-component properties, or poor fits
are not silently converted into defaults.
The catalogue also records rejected nonmonotonic formaldehyde curves and
suspect propane digitizations. Neon on 3A uses added local neon properties
sourced from [CoolProp](https://coolprop.org/fluid_properties/fluids/Neon.html).
The reviewed 4A/5A neon series disagree strongly and fail the fit-quality
threshold. The retained 3A helium record lies outside the fitting domain
(NIST pressure values 194–4,826 bar), with unspecified absolute/excess basis;
it cannot supply a defensible low-pressure default. ASMS-3A and carbon
molecular sieves containing "3A" in their names are different materials
and are not mapped to zeolite 3A.

The ammonia default uses five approximately digitized points from the Z1
filled-circle branch in Figure 6 of
[Sakata et al. (1994)](https://doi.org/10.11450/seitaikogaku1989.6.2_1).
The original PDF and digitized data are retained in `data/source/` as
`sakata_1994_3a_ammonia.pdf` and `molecular_sieve_ammonia.json`.
This branch describes repeat adsorption after 90-minute vacuum regeneration;
it excludes irreversible first-cycle uptake. Approximate reading uncertainties
are ±25 Torr and ±4 cm3(STP)/g, larger than the 1.05% fitting residual. The
Henry region and temperature dependence are not resolved. These limitations
are reported as runtime warnings. No H2/N2 capacity is inferred from the
same figure's unresolved near-zero baseline.

### Audited methane temperature correction and external check

The NIST methane/4A record `10.1007s1045001698416.Isotherm20` says 298 K.
Matching pressures and absolute loadings in the original paper's
`Online resource 9` establish that this curve is at **−0.160 °C = 272.990 K**.
The other two curves are at **297.991 K** and **322.989 K**. The raw NIST
records remain unchanged; `data/source/molecular_sieve_corrections.json`
records the interpretation, and `wynnyk_2017_supplement.docx` preserves the
original evidence. Both the builder and comparison script apply the audit.

The resulting temperature-dependent dual-site methane fit has **0.34% MARE**
and **0.61% held-out MARE**, versus the former 20.37% fitting error caused by
the conflicting temperature labels. At fugacity 0.5 bar, its differential
isosteric heat is 19.25, 18.50 and 17.90 kJ/mol at the three temperatures.

An independent, newer [2020 study](https://doi.org/10.1007/s10450-020-00206-7)
is retained in `data/source/methane_4a_external_validation.json` and is never
used for fitting:

| T (K) | Pressure (bar) | Observed (mol/kg) | Predicted (mol/kg) | Relative error |
|---|---:|---:|---:|---:|
| 273 | 0.99265 | 1.25666 | 1.40531 | +11.83% |
| 313 | 1.01433 | 0.58537 | 0.58188 | −0.60% |
| 363 | 1.05882 | 0.26367 | 0.25357 | −3.83% |

These near-1-bar checks slightly extrapolate the fitting fugacity range;
363 K additionally extrapolates temperature. Different batches and the
external study's excess-loading convention limit exact agreement. The verified
2017 source is retained for fitting instead of replacing it solely because
the other publication is newer.

Reproduce the fit comparison and catalogue audit with
`scripts/compare_adsorption_fits.py` and `scripts/audit_molecular_sieve_catalogue.py`.
The latter accepts `--baseline` to compare a previous catalogue and `--output`
for the report. The selected-form audit is retained under `docs/investigations/`.

A single-temperature default is used only within 1 K of its recorded
temperature (allowing rounded experimental temperatures). At other temperatures,
it is unavailable until a temperature-dependent custom curve is provided.
For temperature-dependent defaults, extrapolation outside the measured range
is warned about. IAST fictitious pure fugacities can exceed the feed partial
fugacities substantially; extrapolation diagnostics check these fictitious
fugacities too.
Selected Sips curves improve predictions within the measured region but do
not have a finite nonzero Henry limit. Extrapolation warnings identify that
additional limitation explicitly. Smaller loading residuals are not independent
calorimetric validation of the temperature-derived adsorption enthalpies.

If a curve is missing and the molecule fits the nominal pore, the unit warns
and passes that component through. If its molecular size or the custom pore
size is unknown, the unit warns that adsorption cannot be assessed. The
`kinetic_diameters` map and `pore_diameter`, both in angstrom, supply missing
screening information. Known nominal pores are 3, 4, 5 and 7.4 angstrom for
3A, 4A, 5A and 13X respectively. A valid measured or custom isotherm takes
precedence over this approximate size screen.

Kinetic diameters are approximate screening values, not adsorption capacities.
Sources include [steam adsorption on 3A](https://doi.org/10.1007/s10450-020-00283-8),
[methanol exposure of molecular sieves](https://pmc.ncbi.nlm.nih.gov/articles/PMC8892488/),
and [molecular sizes in NaA dehydration](https://pmc.ncbi.nlm.nih.gov/articles/PMC8150887/).

## Custom settings

An `isotherms` map overrides or supplements the selected sieve's defaults.
Keys are feed component symbols or catalogue component identities. Custom
sieve names are allowed; provide a pore diameter to enable size screening.

| `model` | Pure loading equation | Required parameters |
|---|---|---|
| `langmuir` | `qmax*b*f/(1+b*f)` | `qmax`, `b` |
| `dual_site_langmuir` | Sum of two Langmuir terms | `qmax`, `b`, `qmax2`, `b2` |
| `sips` | `qmax*(b*f)^n/(1+(b*f)^n)` | `qmax`, `b`, `n` |
| `toth` | `qmax*b*f/(1+(b*f)^n)^(1/n)` | `qmax`, `b`, `n` |
| `henry` | `k*f` | `k` |
| `freundlich` | `k*f^n` | `k`, `n` |
| `custom` | User expression | `expression`, optional `parameters` |

For standard models, optional `heat` in J/mol and `T_ref` in K change the
affinity via `b(T)=b(T_ref)*exp(heat/R*(1/T-1/T_ref))`, with positive heat
denoting exothermic adsorption. The second Langmuir site uses `heat2`.
Henry/Freundlich use the same scaling for `k`. Without heat, affinity is
temperature independent. `R` is 8.31446261815324 J/(mol K).

```text
UNIT MS : MolecularSieveDryer
    sieve_type = custom
    pore_diameter = 5
    target_component = CO2
    target_mole_fraction = 0.02
    isotherms = {CO2: {model: langmuir, qmax: 4, b: 8, heat: 25000, T_ref: 298.15}, N2: {model: custom, expression: "cap*b*f/(1+b*f)", parameters: {cap: 4, b: 0.3}}}
```

The Python API accepts the same nested dictionaries. Custom equations may use
`f`, `T`, `R`, numeric parameters, arithmetic, powers (`**`), and `exp`, `log`,
`log10`, `sqrt`, `abs`, `min`, `max`. Attribute access, imports, arbitrary calls
and Python statements are prohibited by the existing restricted expression
evaluator. Equations must be nonnegative, monotonic, zero at zero fugacity,
and have a finite spreading integral. Custom curves are checked numerically
over the operating and IAST extrapolation ranges. These sampled checks cannot
prove that an arbitrary equation is physical everywhere; the supplied equation
must satisfy these conditions throughout its domain. Custom integration and
inversion are numerical and slower than the analytic standard forms.

`initial_loadings = {CO2: 0.01, N2: 0}` specifies kg adsorbate/kg dry sieve.
The legacy scalar `initial_loading` / `initial_loading_kg_per_kg` applies to
water. Regeneration is required if the supplied loading would cause net
desorption of any component; the unit rejects that case instead of clipping
a negative uptake and claiming equilibrium.
This check uses the solved outlet equilibrium: a component initially loaded
above its inlet-equilibrium value can still adsorb as competitors are removed.

## Results and adsorption enthalpy

`product` is the remaining fluid; `adsorbate` is the net material removed.
Per-component removed flows, equilibrium and final sieve loadings, source
records, fugacities, IAST spreading potential and material-balance residual
are available in `performance` and PFR reports.

Temperature-dependent isotherms determine adsorbed enthalpy relative to ideal
gas at the contact temperature, neglecting adsorbed-phase pV:

```text
H_ex_pure(f,T) = R*T^2 * (partial Pi / partial T)_f          [J/kg]
H_ex_mix = sum_i q_mix_i * H_ex_pure_i(f0_i,T)/q_pure_i(f0_i,T)
Qst_i = R*T^2 * (partial ln f_i / partial T)_(all q_j fixed) [J/mol]
```

Integral enthalpy is analytic for standard forms and GSTA; custom
temperature-dependent expressions use a checked numerical derivative.
Differential isosteric heats use inverse IAST at fixed **entire adsorbed
inventory**, including competition. Positive `Qst` means exothermic adsorption
relative to ideal gas. Actual fluid enthalpies supply the nonideal vapor or
liquid reference correction.

Finite adsorption heat is the difference between final and initial inventory
enthalpies, including preloading, rather than final differential heat times
uptake. The dry sieve enters and leaves at the same contact temperature, so
its unchanged sensible enthalpy cancels. `heat_duty` is the physical isothermal
bed duty when these enthalpies are identifiable (negative means cooling).

`adsorbate` remains a **fluid material-accounting stream**. The explicit
`unrepresented_enthalpy_change` [kJ/h] replaces that accounting stream's
enthalpy with the adsorbed inventory enthalpy change. The flowsheet audit uses:

```text
H_out_fluid + unrepresented_enthalpy_change = H_in_fluid + Q + W
```

This preserves mass accounting without pretending the removed material is
physically a free fluid or subtracting latent heat twice. Regeneration of
the loaded sieve remains a separate process requiring its own energy duty.

Performance includes `bed_heat_duty_kJ_h`, `stream_enthalpy_duty_kJ_h`,
`isosteric_heats_kJ_per_mol`, and initial/final excess enthalpies.
`adsorption_heat_release_kJ_h` is the finite inventory heat release relative
to ideal gases; bed duty also includes actual fluid enthalpies.

The catalogue has 19 temperature-dependent fitted pairs plus GSTA water with
identifiable enthalpy models. A single-temperature default has **unknown heat,
not zero heat**. Custom standard forms require explicit `heat` (and `heat2`
for dual-site); custom expressions must explicitly depend on `T`. Explicit
zero heat is allowed. If any active adsorbate lacks this information,
`bed_heat_duty_kJ_h` is null, a warning identifies missing components, and
`heat_duty` is labeled fluid-accounting duty only. Zero sieve flow or no
adsorption needs no unknown adsorption-heat contribution.
