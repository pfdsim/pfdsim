# Property Resolution Design

This document describes the current property lookup and fallback design for
`pfdsim`. The main rule is simple: normal property requests should go through
`PropertyResolver`, either directly or through `ChemicalProperties` and
`thermodynamics.py`.

The resolver is intentionally metadata-heavy. Every resolved value should carry
the value, source, method, quality, and notes needed to understand why a
simulation used that number.

## Responsibilities

| Module | Responsibility |
| --- | --- |
| `compound_identity.py` | Canonicalizes names, aliases, CAS numbers, formulas, and common symbols. Formula aliases are conservative because isomers share formulas. |
| `chemical_properties.py` | Owns `ChemicalProperties` and `ChemicalDatabase`. Loads `chemicals.json`, resolves aliases, and hydrates missing scalar properties once per cached chemical through `PropertyResolver`. |
| `property_resolver.py` | Owns source order, fallback policy, units, quality propagation, online cache usage, and temperature-range checks. |
| `property_resolution/coolprop.py` | Owns the bundled static CoolProp identity map, strict pure-fluid matching, backend selection, and shared saturation/property calls. Unknown identities never trigger dynamic CoolProp indexing. |
| `property_resolution/vapor_pressure_canonical.py` | Provider-neutral segment assembly, gap completion, junction checks, and constrained canonical Psat fitting. |
| `property_resolution/vapor_pressure_adapter.py` | Converts component/PFD/CoolProp data into canonical Psat segments and selects the lower canonical domain boundary. |
| `perry_properties.py` | Loads and evaluates extracted Perry correlations. It does not decide global priority. |
| `textbook_properties.py` | Loads the Smith textbook table. It is a secondary local source after `chemicals.json` and Perry where applicable. |
| `vapor_pressure_tables.py` | Provides direct tabulated vapor-pressure interpolation for the current pointwise resolver and future exceptional canonical source segments. |
| `thermodynamics.py` | Uses resolver-backed component properties for phase and mixture calculations. |

## Request Paths

| Request style | Path |
| --- | --- |
| `ChemicalDatabase.get(identifier)` | `chemicals.json` direct/case-insensitive lookup -> identity resolver -> local aliases/SMILES/CAS -> local Smith textbook -> local Perry -> PubChem/NIST only if enabled. Bare ambiguous formulas are not guessed from non-CAS tables. |
| `PropertyResolver.resolve_*(identifier, props=None)` | If `props` is omitted, usually calls `ChemicalDatabase.get()` for hydrated local props before applying that property chain. `Psat(T)`, `Tb`, and `Tm` avoid online hydration until deterministic local sources have had their chance. For supported pure fluids, CoolProp is a deterministic local source. Explicit PFD-tagged props are authoritative overrides; ordinary supplied props may still be replaced by the global CoolProp bundle. |
| `ChemicalProperties.Psat/Cp/Hvap_at_T/density` | Calls `PropertyResolver` with the object's props. Stored Antoine/Cp constants are no longer a separate path. |
| `thermodynamics.py` activity methods | `Psat`, ideal-gas Cp, liquid Cp, Hvap, and liquid molar volume use resolver-backed paths. Activity mixture Cp blends phase-specific component Cp with vapor fraction. |
| EOS methods | EOS constructors read hydrated `ChemicalProperties` fields (`Tc`, `Pc`, `Vc`, `omega`, etc.). They do not call the resolver directly. |

## Quality Semantics

`quality` is a 0-1 scalar describing trust in the resolved value. It is not a
probability; it is a monotonic ranking for diagnostics, warnings, and choosing
between alternatives.

When a calculation combines several inputs, use:

```text
quality = min(input qualities) * method_factor
```

Exact formulas use source `exact` and factor `1.0` only when every input is
real/source-backed. If any input is soft, estimated, or missing, the result
stays `calculated` or `estimated` and carries the appropriate method factor.

Softness is quality-gated. Values below the hard threshold are treated as soft
even if the source string says `exact`.

Source metadata may explicitly set `replaceable: false`. This separates
authority from uncertainty: the numeric `quality` remains the honest confidence
score used by downstream calculations, while provider arbitration must not
substitute a different source merely because its score is higher. PFD overrides
remain authoritative through their existing mechanism. The explicit flag is
used for deliberately selected coupled bundles such as glycerol's EOS-effective
`Tc/Pc/Vc/Zc/omega`, where replacing one member would destroy internal
consistency even though the bundle carries sub-0.90 uncertainty labels.

General anchors:

| Quality | Meaning |
| --- | --- |
| `1.00` | User-specified authoritative value, exact definitions, exact math. |
| `0.99+` | Direct, validated experimental value under matching conditions. |
| `0.97+` | High-quality curated database or critically evaluated correlation. |
| `0.95+` | Textbooks, trusted local databases, `chemicals.json`. |
| `0.90+` | Lower-priority local databases, direct reputable online sources, very standard correlations, near-exact estimates. |
| `0.85+` | Good fitted correlation, minor extrapolation, strong corresponding-states estimate. |
| `0.80+` | Medium-quality estimate/source, ordinary corresponding-states estimate, validated group contribution. |
| `0.70+` | Unverified online value, moderate extrapolation, weaker fitted correlation, decent group contribution. |
| `0.60+` | Motivated rough estimate, broad group-contribution method. |
| `0.40+` | Heuristic, fallback correlation outside its comfort zone. |
| `0.20+` | Assumed typical/default value. |
| `0.00` | Unknown, placeholder, deliberately unsafe/unresolved. |

## Hydration Policy

`ChemicalDatabase` hydrates scalar fields:

```text
Tc, Pc, Vc, Zc, omega, Tb, Tt, Pt, Tm, Hvap, Hfus, Hf, Gf, S, Hcomb
```

Temperature-dependent properties are not flattened into `ChemicalProperties`.
They remain resolver calls so valid ranges, source priority, and equation forms
are evaluated at the requested temperature.

Hydration fills gas-standard formation values when they are available or
derivable. It does not hydrate large temperature-dependent tables into
`ChemicalProperties`.

## Online Caching Policy

Online lookups remain enabled by default for missing properties when the caller
allows online resolution. Successful values and negative lookups are cached in
the resolver cache.

Vapor pressure uses temperature-band negative caching for online Antoine misses
so a solver does not repeatedly query absent data while walking a temperature
interval. PubChem liquid-density lookups also cache successful parsed clusters
and misses.

Local Perry and Smith lookups are not online. They remain available even when
`ChemicalDatabase(enable_online=False)` or `fetch_online=False` is used.

PFD component definitions are an explicit user override layer. Values and
correlations supplied in a `.pfd` are tagged with `pfd_component_override` or
`pfd_property_correlations`, assigned source quality `1.0`, and take precedence
over lookup sources for the same property. Derived methods such as Watson
`Hvap(T)` or Rackett liquid-volume fitting still apply their method penalties.

After PFD overrides are applied, the component is rehydrated locally. This
fills any still-missing primitive fields and recomputes coupled derived fields
from the final selected inputs. In particular, a non-PFD `Zc` is discarded
when PFD overrides change `Tc`, `Pc`, or `Vc`, then recomputed from the final
triplet. A one-sided PFD `Tt` or `Pt` override does not retain the other
provider's triple-point counterpart.

## CoolProp Policy

CoolProp is a global deterministic local source for supported pure fluids:

- PFD overrides remain first and retain quality `1.0`.
- CoolProp `Tb`, `Tt`, `Pt`, `Tc`, `Pc`, `Vc`, `omega`, and Psat source
  segments have quality `0.995`.
- Water deliberately uses `IF97`; other accepted pure fluids use `HEOS`.
- Runtime identity lookup uses `data/coolprop_fluid_aliases.json`. The map is
  generated offline, omits ambiguous normalized aliases, and prevents unknown
  components from triggering dynamic CoolProp fluid indexing.
- The installed CoolProp dependency is pinned to `7.2.0`, the same version
  recorded by the bundled identity map.
- CoolProp replaces ordinary non-PFD hydrated/provided anchors as one preferred
  source family. This is intentional even when the prior value was otherwise
  hard or curated, because downstream EOS and saturation work benefits from a
  single internally consistent model bundle.
- `Zc` is not selected as an independent CoolProp field. It is calculated only
  after the final `Tc/Pc/Vc` fields have been selected. An all-CoolProp triplet
  retains method `coolprop_critical_identity`; mixed PFD/CoolProp or other
  triplets use `critical_volume_identity`.
- One-sided PFD triple-point overrides are not mixed with CoolProp or hydrated
  counterparts. The unspecified member remains missing unless the PFD supplies
  it too.
- CoolProp viscosity is a separate property path with quality `0.96`; its rung
  depends on whether pressure was explicitly supplied.

Canonical Psat domain selection consumes the already resolved `Tt/Pt` pair,
then `Tm`, then a completion-supplied `Tsat(P_floor)`. The default floor is
`0.001 bar` (`100 Pa`) and may be overridden. `Tt` or `Tm` is accepted only
when its saturation pressure is at least the selected floor. A missing `Pt` is
replaced by a generic `Psat(Tt)` resolution, while `Psat(Tm)` is always resolved
through the same chain. Canonical curves end at `Tc`. A direct full-domain PFD
curve bypasses completion and fitting by design. Fitted curves retain source
quality separately from numerical fit diagnostics. The base A-F form ends in
`F*T^5`; an A-G retry adds `G*T^3`, and an A-H retry adds one inverse tail term
with power `-3`, `-5`, or `-7`. Retry and rejection thresholds depend on source
quality and whether one provider covers the complete domain.

## Property Matrix

| Property | Resolution order | Current simulator use |
| --- | --- | --- |
| Current pointwise `Psat(T)` | PFD-provided `Psat`/Antoine when present -> direct vapor-pressure table in range -> provided portable `Psat` correlation in range -> provided Antoine in range -> Perry 2-8 vapor-pressure correlation in range -> Perry 2-10 tabulated vapor-pressure interpolation in range -> Smith/`antoine.txt` Antoine in range -> validated online Antoine row containing `Tb` -> Ambrose-Walton -> Lee-Kesler -> any other in-range cached/online Antoine -> near local Antoine extrapolation -> Clausius-Clapeyron from `Tb/Hvap` -> online `Hvap` plus Clausius-Clapeyron -> far local Antoine extrapolation -> error. This remains the live runtime path until canonical initialization is wired. | VLE, activity models, flashes, distillation, `ChemicalProperties.Psat()`. |
| Canonical Psat construction | PFD canonical override or pinned PFD correlation -> CoolProp pure-fluid saturation -> validated exceptional local tables -> local `chemicals.json` Antoine -> Perry 2-8 -> Perry 2-10 -> validated external Antoine. Missing ranges will be supplied by external boundary-conditioned AW, frozen-Hvap Clapeyron, then Nannoolal relations. The assembled curve is first fitted to `ln(P/bar)=A+B/T+C*ln(T)+D*T+E*T^2+F*T^5`, exactly constrained at qualified `Tb/1 atm` and `Tc/Pc`; difficult curves may add `G*T^3`, then an inverse `H` tail. Only PFD and CoolProp input methods are currently implemented. | Future initialization-time replacement for pointwise runtime switching and compiled Psat/Jacobian kernels. |
| `Tc`, `Pc`, `Vc`, `Zc`, `omega` | PFD field override -> CoolProp pure-fluid bundle -> ordinary provided hard value -> ACS JCED critical review -> Perry -> Smith textbook -> effective critical with quality at least `0.95` -> unstarred PSRK-2005 `Tc`/`Pc` fallback -> other qualifying effective critical -> strict NIST/PubChem online result when allowed -> lower-grade effective critical -> exact `Zc=Pc*Vc/(R*Tc)` when selected inputs qualify -> Nannoolal and empirical estimates when enabled -> missing. `Zc` is recomputed from the final selected triplet when CoolProp supplies a primitive or PFD overrides `Tc`, `Pc`, or `Vc`, unless PFD explicitly overrides `Zc`; otherwise an independent hard `Zc` remains eligible. | EOS construction, corresponding-states `Psat`, vapor-pressure fallbacks, liquid-volume fallbacks. |
| `Tb` | PFD override -> CoolProp saturation temperature at 1 atm -> ordinary provided value -> Smith textbook -> Perry 2-8 vapor-pressure root at 1 atm -> Perry 2-10 tabulated 760 mmHg point -> hydrated local value -> strict online phase-change result -> formula/HBD estimate -> weak MW estimate -> missing. | Phase-at-STP classification, Watson `Hvap_at_T`, Clausius-Clapeyron fallback, Lee-Kesler omega estimate. |
| `Tt`, `Pt` | PFD pair or individual override -> CoolProp pure-fluid triple-point pair -> hydrated component values -> missing. If only one member is PFD-overridden, the other provider's member is deliberately not mixed in. | Canonical Psat lower-domain selection and triple-point provenance. |
| `Tm` | Explicit provided/PFD value -> curated `chemicals.json` `Tm` override -> Perry 2-68 fusion-row melting point -> Perry 2-10 melting point -> hydrated local value -> PubChem -> NIST phase-change data -> missing. | Phase-at-STP classification and metadata. |
| `Hvap` | Provided portable `Hvap(T)` correlation when target `T` is supplied -> PFD scalar `Hvap(Tb)` as Watson reference -> Perry `Hvap(T)` correlation -> accepted NIST multi-point Watson fit -> fixed-exponent Watson from a source-backed `Hvap(Tb)` reference -> direct source-backed scalar at the requested/reference condition -> Smith textbook -> Trouton estimate when enabled -> missing. Online measurements at temperatures other than `Tb` stay temperature-tagged records and are not silently stored as scalar `Hvap`. | `Hvap_at_T`, frozen-source deep-vacuum Clausius-Clapeyron construction, latent heat. |
| `Hfus` | Provided scalar -> Perry fusion data -> PubChem -> NIST phase-change data/median fusion enthalpy -> missing. | Exposed through resolver/database; not currently used by unit operations. |
| Liquid Cp `Cp_l(T)` | Provided portable `Cpl` correlation in range -> provided constant `Cp_liquid` -> Perry liquid Cp in range -> up to 20 K bounded extrapolation of provided/Perry liquid Cp when no constant exists, then clamp to that boundary -> online NIST tabulated liquid Cp averaged to a constant -> provided ideal-gas polynomial multiplied by 1.3 -> MW estimate -> error. | Activity-model stream Cp, heaters/coolers, enthalpy. |
| Ideal-gas/vapor Cp `Cp_ig(T)` | Provided portable `Cpg` in range -> provided ideal-gas polynomial -> Perry ideal-gas Cp in range -> up to 20 K bounded extrapolation of provided/Perry gas Cp, then clamp -> online NIST tabulated gas Cp: 10+ points fit Shomate-style with outlier exclusion, 4-9 points fit linear, 1-3 points interpolate/clamp -> MW estimate -> error. | Gas/vapor stream Cp, reaction enthalpy sensible corrections, `ChemicalProperties.Cp()`. |
| Liquid molar density `rho_l(T)` | Provided portable mass-density correlation in range, converted with MW -> PFD scalar `rho` via liquid-volume/Rackett path when supplied -> Perry liquid density in range -> liquid-volume chain converted back to density. | Exposed directly; liquid molar volume is used by activity-model density. |
| Liquid molar volume `V_l(T)` | Provided density correlation in range -> PFD scalar `rho` as a Rackett fit source -> Perry density in range -> up to 20 K density-correlation extrapolation -> Rackett with cached `Z_RA` fitted from provided/Perry density -> PubChem liquid-density cluster: fitted Rackett with hard `Tc/Pc`, otherwise thermal expansion from nearest point -> PTV if `Zc` is hard -> Rackett/Yamada-Gunn -> plain PR -> formula atom-increment -> MW heuristic -> error. | Activity-model mixture density, stream density estimates, Poynting corrections. |
| Vapor viscosity `mu_v(T,P)` | Provided portable viscosity correlation in range -> CoolProp when pressure is explicit -> Perry in range -> bounded Perry extrapolation -> CoolProp at a phase-consistent atmospheric/saturation bound when pressure is omitted -> Yoon-Thodos -> error. | Exposed for hydraulics/pressure-drop work. |
| Liquid viscosity `mu_l(T,P)` | Provided portable viscosity correlation in range -> CoolProp when pressure is explicit -> Perry in range -> bounded Perry extrapolation -> CoolProp at a phase-consistent atmospheric/saturation bound when pressure is omitted -> Hsu group contribution -> error. | Exposed for hydraulics/pressure-drop work. |
| Surface tension `sigma(T)` | Provided/PFD `sigma` or `surface_tension` correlation in range -> provided scalar -> CAS-keyed local correlations in priority order: Mulero/REFPROP, revised Somayajulu, VDI/DIPPR EQ106, Jasper/Lange -> Knotts/Perry group-contribution Parachor estimate below `Tr=0.90` when `Tc` is available -> error. The Parachor path resolves saturated liquid density normally. Saturated vapor density uses PTV when `Zc` is hard (`quality >= 0.90`), otherwise PR; both EOS paths require `Tc/Pc` quality `>= 0.80` and `omega` quality `>= 0.70`, then fall back to `rhoV=0` when inputs, `Psat`, or a vapor root are unavailable. | Pure-component input for on-demand mixture calculations in `interfacial_properties.py`. |
| Molecular radii of gyration `R_g`, `R'` | PFD/property override -> conventional mass-weighted `R_g` and Thompson linear/nonlinear principal-moment `R'` from the shared cached GFN2-xTB gas geometry. A missing geometry may be generated and is persisted before either radius is evaluated. | `R'` is an input to the Hayden-O'Connell `-BV` provider; `R_g` is exposed for future molecular corresponding-states methods. |
| `Hf`, `Gf`, `S`, `Hcomb` | Provided -> Perry gas-standard formation table -> online NIST gas/liquid thermochemistry rows -> derive missing gas `Hf` from liquid `Hf + Hvap(298.15 K)` -> derive gas `Gf` or absolute gas `S` from the other using elemental standard entropies -> derive gas `Gf` from `Gf_liquid - RT ln(Psat/1 bar)` -> derive `Gf_liquid` from `S_liquid` and liquid `Hf`, then gas `Gf/S` -> final Domalski-Hearing gas `Hf/S` estimation (`Hf` only inside its structural applicability domain) -> convert gross `Hcomb` to net -> calculate net `Hcomb` from formula and reactant STP phase -> missing per field. | Reaction enthalpy and thermochemistry metadata. |

## Source Notes

`chemicals.json` is authoritative for curated local identity and scalar values.
It no longer stores duplicate Antoine vapor-pressure coefficients or Cp
constants for ordinary core species, so normal `Psat` and Cp calls reach direct
tables and Perry correlations. For curated compounds outside Perry coverage,
`chemicals.json` may store explicit `property_correlations` with portable
equation forms and valid temperature ranges.

Curated `Tm` corrections in `chemicals.json` are reserved for known
Perry-table disagreements and carry `curated_tm_override` metadata with quality
`0.98`. They take precedence over Perry Table 2-68 and Table 2-10. Ordinary
hydrated `Tm` values without that metadata are fallback values, so the two
Perry melting-point tables resolve at equal quality before them.

Portable correlation forms:

| Key | Equation |
| --- | --- |
| `Psat` | `ln(Psat/Pc) = (A*tau + B*tau^1.5 + C*tau^3 + D*tau^6)/Tr`, with `tau=1-Tr`, `Tr=T/Tc`; result converted from Pa to bar. |
| `Psat` with `canonical_psat` or `canonical_psat_af` | A-F uses `ln(Psat/bar)=A+B/T+C*ln(T)+D*T+E*T^2+F*T^5`. `canonical_psat_ag` adds `G*T^3`; `canonical_psat_ah` additionally adds `H*((T/Tc)^inverse_power-1)` for `inverse_power` in `{-3,-5,-7}`. A full-domain PFD curve is a direct quality-`1.0` canonical override; a narrower curve is a pinned segment over its declared range. |
| `Hvap` | `ln(Hvap_kJmol) = A + B*ln(tau) + C*tau + D*tau^2`, with `tau=1-T/Tc`. |
| `Cpl`, `Cpg`, `rhol`, `mug` | Polynomial in `x=(T-298.15)/100`. Vapor viscosity `mug` is stored in microPa*s and converted to Pa*s. |
| `mul` | Exponential polynomial in `x=(T-298.15)/100`; result is Pa*s. |
| `mug` or `mul` with `poly_tp` | `mu = A + B*x + C*p + D*x^2 + E*x*p + F*p^2`, with `x=(T-298.15)/100` and `p=P_bar-P_ref_bar`. Optional `Pmin_bar`/`Pmax_bar` bound the pressure range. This is a pressure-specific fit and is not pressure-corrected again. |
| `mug` or `mul` with `viscosity_exp_rhor` | `mu = exp(A+B/T+C*ln(T)+D*T) + exp(E+F/T)*(rho_r+x*rho_r^2+y*rho_r^3)`. `mug` output is microPa*s; `mul` output is Pa*s. Explicit pressure requires supplied molar density; without pressure only the dilute-density term is used. |
| `sigma`, `surface_tension` | Existing portable forms such as `poly_x`/`exp_poly_x`, or DIPPR EQ106: `sigma=A*(1-Tr)^(B+C*Tr+D*Tr^2+E*Tr^3)` with `Tr=T/Tc`; result is N/m. |

The Knotts Parachor fallback uses Perry Table 2-177 contributions with an
RDKit heavy-atom fragmentation and requires at least one carbon atom, preventing
organic functional-group labels from being applied to elemental or wholly
inorganic species. Molecules with two or more halogens bonded to the same atom
are rejected because the table does not define that substitution topology. It
detects carbon hybridization, branch count,
ring size, fused aliphatic/aromatic atoms, benzene ortho/meta/para substitution,
substituted naphthalene, and the listed oxygen, nitrogen, sulfur, halogen,
silicon, phosphorus, boron, and aluminum groups. Fragmentations are cached by
SMILES. Degree-three and degree-four acyclic carbon branch points
are classified as secondary and tertiary branching, respectively; every bond
between two branch points receives the corresponding sec-sec, sec-tert, or
tert-tert adjacency correction in addition to the per-branch corrections.
When the saturated-vapor-density EOS inputs do not meet their quality gates,
`Psat` cannot be resolved, or neither EOS yields a vapor root, the Parachor path
uses `rhoV=0`. Parachor result quality starts at `0.80`, subtracts
`max(0, 0.92 - quality_rhoL)`, and then subtracts `0.02` for PR vapor density or
`0.10` for the zero-vapor fallback. A result without `Tc` receives a flat
quality of `0.50`. Accepted branching, adjacency, fused-ring, and polyfunctional
fragmentations do not receive separate quality penalties.

## Mixture Surface Tension

`interfacial_properties.py` calculates liquid-vapor mixture surface tension on
demand rather than adding it to every stream state. Butler calculations use a
separately supplied UNIFNIST `activity_coefficients(T, composition)` object;
this does not change the flowsheet's selected VLE/LLE thermodynamic model. The
same activity object is retained so its UNIFAC fragmentation and compiled
backend remain available, while the interfacial calculator adds bounded
activity caches and nearby-state Butler warm starts.

Butler molar surface areas first use the CAS-keyed values in
`data/mixture_surface_tension_areas_cas.json` when every active component is
covered. If any active component is missing, Goldsack-White estimates are used
for every active component so tabulated and estimated area scales are not mixed.
The tabulated 2-propanol area is `10.6948685e4 m^2/mol`, fitted to six aqueous
surface-tension points from 5 to 30 wt% at 25 C. Binary and multicomponent
systems use the same constrained Butler equations. Damped Newton is preferred
when its finite-difference Jacobian is well-conditioned, with SciPy least
squares and damped fixed point as numerical fallbacks. WSD remains an explicit
method fallback and always emits a warning because it omits surface segregation.

Extracted Perry data currently includes:

| Perry table | Stored data |
| --- | --- |
| 2-8 | Vapor-pressure correlations. |
| 2-10 | Organic vapor-pressure table up to 1 atm. Stored separately by resolved CAS and used for bounded log-PCHIP `Psat(T)`, normal boiling point at 760 mmHg, and melting-point fallback. |
| 2-32 | Liquid molar-density correlations and endpoint molar volumes. |
| 2-68 | Organic heat of fusion and melting point. Stored separately because the table has no CAS column; rows are resolved only when formula and name agree through Perry 2-10 or `chemicals`. |
| 2-69 | Heat-of-vaporization correlations. |
| 2-72 | Liquid heat-capacity correlations. |
| 2-74, 2-75 | Ideal-gas heat-capacity correlations. |
| 2-95 | Ideal-gas formation properties and standard entropy. |
| 2-106 | Critical constants and acentric factors. |
| 2-138, 2-139 | Vapor and liquid viscosity correlations. |

Not yet extracted but useful future Perry data:

| Perry pages/tables | Potential use |
| --- | --- |
| PDF page 143 / printed page 2-108, Table 2-58 | Direct aqueous ethanol density table versus mass percent and temperature. |
| PDF pages 146-147 / printed pages 2-111 to 2-112, Table 2-63 | Aqueous organic solution density polynomial correlations from International Critical Tables. |
| PDF pages 385-386 / printed pages 2-350 to 2-351 | Spencer-Danner-Li modified Rackett mixture volume prediction. |

## Heat Capacity and Enthalpy

Gas, ordinary-liquid, and solid Cp are phase-specific. Each resolved kernel
owns scalar Cp plus analytic sensible enthalpy and entropy primitives. Liquid
Cp uses liquid data before predictive and scaled-gas fallbacks.

Solid Cp uses a form-aware multi-record canonical database. Its runtime order
is provided `Cps`, provided `Cp_solid`, bundled native solid curves, a non-PFD
298.15 K point, persisted/online NIST, strict-domain Lastovka, modified Kopp
near 298.15 K, then failure. It has no gas-scaled or universal Dulong-Petit
fallback.

The bundled sources are JANAF tables, NIST WebBook solid Shomate, CRC
temperature tables and standard points, and Perry 2-151. JANAF/WebBook share a
lineage and do not corroborate each other. Native tabular, Shomate, and Perry
kernels are retained instead of smoothing all records into one curve.

Solid-solid boundaries are explicit. Cp can be evaluated on either side, but
`delta_h` and `delta_s` reject an interval crossing a transition until its
transition enthalpy is represented. Glass, crystalline, hydrate, solvate,
allotrope, and named-polymorph records are not silently merged.

Provided ordinary-liquid and solid correlations retain every explicit range
endpoint. For a provided `Cpl` correlation, a missing lower endpoint defaults
to resolved `Tm` and a missing upper endpoint defaults to resolved `Tb`; the
respective fallbacks for unavailable or interval-incompatible phase points are
273.15 K and 1500 K. For a provided `Cps` correlation, a missing upper endpoint
defaults to resolved `Tm` and a missing lower endpoint defaults to
`min(100 K, 0.8*Tm)`. If `Tm` is unavailable, the solid fallbacks are 100 K and
1500 K. The effective inferred range and its phase-point provenance are
retained in the kernel notes and fingerprint.

NIST Cp behavior:

- Native liquid and solid Shomate ranges are retained as executable kernels.
- A single liquid/solid point becomes a narrow constant reference; sparse
  liquid tables use a linear kernel and larger tables use robust Shomate or
  portable Chebyshev fits.
- Gas tables with 10+ points use a Shomate-style polynomial fit with outlier
  exclusion.
- Gas tables with 4-9 points use a linear fit.
- Gas tables with 1-3 points use interpolation/clamping.
- Every normalized liquid/solid kernel and the Shomate/linear gas fits have
  analytic Cp integration hooks for enthalpy speed.
- Interpolation/clamp tables remain pointwise and do not have analytic
  integration.
- Solid tabular tables retain exact piecewise-linear analytic primitives and
  preserve duplicate-temperature transition boundaries.

All three heat-capacity phases use one bounded range policy. Up to 10 K outside
the valid range, the correlation is evaluated at the requested temperature and
its quality is reduced by `0.02`. Farther outside, evaluation clamps to the
10 K continuation boundary; the quality penalty then grows by `0.02` per
additional 5 K and is capped at `0.40`.

Solid Cp retains an additional cryogenic guard: below 100 K, extrapolation and
clamping are forbidden, but a correlation whose effective range covers the
requested temperature remains valid. That range may be source-declared or the
`0.8*Tm` lower range inferred for a provided `Cps` correlation. Solid-solid
transition boundaries are still enforced independently of this outer-range
policy.

## Solid Volume and Density

`data/solid_volume.sqlite` contains 1,872 CRC constant solid molar volumes.
The resolver preserves native `Vm_solid`, then derives mass density through
the selected molecular weight. Explicit `Vm_solid`, `rho_solid`, and `rhos`
PFD data take priority, followed by the bundled CRC source, form-qualified
PubChem observations, and the existing strict organic density heuristic.

The organic no-observation fallback uses the temperature-general correlation
`rho_s(T) = (1.28 - 0.16*T/T_transition)*rho_l(T_transition)` from
*J. Chem. Eng. Data* (2004) 49 (6): 1512–1514, which reports 5.6% MAPE across
the evaluated temperatures. It is admitted from `0.3*T_transition` through the
selected trustworthy triple or melting point, provided no known solid-Cp
transition boundary lies between the requested temperature and that point.
Neutral, single-fragment molecular organics are eligible regardless of size or
whether their transition is below room temperature. Inorganic, metallic,
formally charged, hydrate, and solvate identities remain excluded. An empirical
packing-risk guard also excludes molecules with at least 10 chirality-aware
heavy-atom graph automorphisms; this guard reflects local validation rather
than a claim made by the cited correlation source.

The CRC source has no temperature correlation. Inorganic thermal expansion is
therefore zero rather than guessed; quality decreases by 0.01 per complete
25 K away from 298.15 K, capped at a 0.30 penalty. Neutral molecular-organic
PubChem observations retain the documented `1.7e-4 K^-1` average expansion
model. Hydrates, solvates, and polymorph-only observations remain form-safe.

## Heat of Vaporization

`PropertyResolver` owns Watson scaling. The old thermodynamics-level Watson
path and obsolete local `HVAP_DATABASE` path were removed.

When a target temperature is supplied, the resolver first tries direct
temperature-dependent `Hvap` correlations. If no direct value exists, it builds
a Watson curve around a normal-boiling reference chosen from:

1. `Hvap(Tb)` from provided/Perry/NIST temperature-dependent fits.
2. PFD or other source-backed scalar `Hvap(Tb)`.
3. Smith textbook normal `Hvap`.
4. A temperature-tagged online measurement converted from its real reference
   temperature when the source and critical inputs qualify.

If Watson construction fails, scalar `Hvap` is returned only as a scalar
fallback.

Scalar `ChemicalProperties.Hvap` strictly means `Hvap(Tb)`. PubChem and NIST
measurements at 25 C or any other explicit temperature stay in
`Hvap_records`; records with unknown temperature are provenance-only and are
not eligible for Watson or Clausius-Clapeyron calculations. NIST multi-point
fits are preferred over distant scalar conversion, and one selected Hvap model
must be frozen before a canonical deep-vacuum integration begins.

## Liquid Volume Details

The pure-component liquid volume chain is layered from experimental/correlated
data to increasingly predictive estimates.

Quality anchors:

Vapor-pressure source anchors:

| Method | Quality behavior |
| --- | --- |
| PFD canonical or pinned Psat override | `1.0`. |
| CoolProp pure-fluid saturation segment | `0.995`. |
| Direct curated vapor-pressure table | `0.995`. |
| Perry 2-8 vapor-pressure correlation | `0.98`. |
| Perry 2-10 tabulated Psat, older local Antoine, validated online Antoine | `0.95`. |
| Ambrose-Walton | Critical-property input minimum times `0.91`. |
| Lee-Kesler | Critical-property input minimum times `0.89`. |
| Other online Antoine | `0.87`. |
| Near local Antoine extrapolation | `0.85`. |
| Clausius-Clapeyron from `Tb/Hvap` | Input minimum times `0.75`. |
| Online-Hvap Clausius-Clapeyron | Input minimum times `0.70`. |
| Far local Antoine extrapolation | `0.60`. |

Scalar phase-change source anchors:

| Method | Quality behavior |
| --- | --- |
| PFD scalar override | `1.0`. |
| CoolProp `Tb`, `Tt`, or `Pt` | `0.995`. |
| Smith textbook `Tb` | `0.98`. |
| Perry 2-8 `Tb` from vapor-pressure root | `0.97`. |
| Perry 2-10 `Tb` from 760 mmHg point | `0.94`. |
| Curated `chemicals.json` `Tm` override | `0.98`. |
| Perry 2-68 `Tm` and Perry 2-10 `Tm` | `0.95`. |

Critical-property source anchors:

| Method | Quality behavior |
| --- | --- |
| PFD scalar override | `1.0`. |
| CoolProp `Tc`, `Pc`, `Vc`, or omega | `0.995`. |
| ACS JCED/IUPAC reviewed critical value | `0.99`. |
| Perry or Smith critical value | `0.98`. |
| Unstarred PSRK-2005 `Tc` or `Pc` | `0.95`; offered only after modern local reviews and never exposes starred estimates, `Vc`, or omega. |
| Effective critical at quality `>=0.95` | Retains its stored score and precedes the PSRK fallback because this band denotes experimental criticals that coincide with the effective parameterization. |
| NIST AVG critical row | `0.96`. |
| Multiple consistent NIST rows | `0.95`. |
| Single NIST row | `0.94`. |
| Multiple consistent PubChem rows | `0.89`. |
| Single unit-qualified PubChem row | `0.88`. |
| Lee-Kesler omega | Input minimum times `0.95` for ordinary compounds; input minimum times `0.80` when HBD > 1, HBA > 2, or structure classification fails. |

Online critical parsing requires explicit recognized units, rejects predictive
or calculated rows, and rejects mutually inconsistent report clusters. NIST
replaces PubChem per critical field when both are available. Effective-critical
values count as physical/source-backed for canonical AW eligibility only when
that individual field has quality at least `0.93`.

| Method | Quality behavior |
| --- | --- |
| In-range provided/Perry density | About `0.98`. |
| Small provided/Perry density extrapolation | Input quality times `0.96`. |
| Rackett with `Z_RA` fitted to provided/Perry density | Fitted `Z_RA` uses critical/input minimum times `0.94`; Rackett evaluation is exact from those inputs. |
| PubChem density points | `0.80-0.92` depending on format, reference, peer-review marker, temperature availability, and cluster consistency. |
| PubChem-fitted Rackett | `min(PubChem cluster, Tc, Pc) * 0.98`, only with hard `Tc/Pc`. |
| PubChem thermal expansion | Cluster quality times `1.00` within 25 K, `0.96` within 100 K, `0.90` within 200 K, `0.84` farther away. |
| PTV liquid volume | Hard-input minimum times `0.84`. |
| Rackett/Yamada-Gunn | Input minimum times `0.78`. |
| Plain PR liquid root | Input minimum times `0.75`. |
| Formula atom-increment liquid volume | `0.60`. |
| MW-only heuristic | `0.40`. |

PubChem density parser rules:

- Accept explicit density units, relative density/specific gravity, ranges, and
  simple bare values only from PubChem `Density` rows.
- Reject vapor/gas/air/saturated-air rows, aqueous/commercial/mixture rows,
  solids/ice/molten rows, bulk/powder density, obvious estimates, and invalid
  temperatures.
- Cluster accepted points after a small 298 K normalization and discard
  inconsistent outliers.
- Cache successful clusters and misses.
- If PubChem succeeds, terminate the pure liquid-volume chain before predictive
  EOS/corresponding-states methods.

The thermal-expansion fallback is:

```text
V(T) = V_ref * exp(9e-4 * (T - T_ref))
```

with the expansion factor clamped to `0.70-1.50`.

EOS liquid-volume fallback pressures are floored at 1 Pa for numerical
stability when used by the liquid-volume resolver. The EOS modules themselves
are not globally pressure-clamped.

## Pressure Standards

pfdsim keeps two pressure standards deliberately separate:

- Thermochemical standard-state properties use 1 bar. This includes `Hf`,
  `Gf`, absolute standard entropies, entropy pressure corrections, and
  fugacity-standard terms such as `-RT ln(Psat/1 bar)`.
- Normal boiling points use 1 atm = 1.01325 bar. This includes explicit `Tb`
  values, Perry normal-point rows at 760 mmHg, vapor-pressure roots used to
  infer `Tb`, Clausius-Clapeyron fallbacks anchored at `Tb`, and Lee-Kesler
  acentric-factor estimates from `Tb`.

The named constants live in `pressure_standards.py`. `thermodynamics.P_REF`
aliases the 1 bar thermochemical standard for compatibility with existing
callers; normal-boiling code imports `NORMAL_BOILING_PRESSURE_BAR` instead of
using `P_REF`.

## Formation Properties

NIST thermochemistry rows are parsed for gas and liquid phase formation data.
The resolver treats gas entropy `S` and liquid entropy `S_liquid` as absolute
standard molar entropies, not formation entropies.

Legacy populated formation fields in the bundled `chemicals.json` predate its
per-property provenance map. They are retained as curated local values with
`source=local`, `method=chemicals_json`, and quality `0.98`; missing fields
remain missing and are never manufactured by this default.

Important derivations:

- Missing gas `Hf` can be derived from `Hf_liquid + Hvap(298.15 K)`.
- Missing gas `Gf` or gas absolute `S` can be derived from the other gas value
  using elemental standard entropies from `data/S_table_elements.txt`.
- Missing gas `Gf` can be derived from `Gf_liquid - RT ln(Psat/1 bar)` at
  298.15 K.
- If only `S_liquid` is available, derive `Gf_liquid` from liquid `Hf` and
  absolute liquid entropy, then derive gas `Gf` and gas `S`.
- As the final gas `Hf/S` rung, Domalski-Hearing group additivity supplies
  missing `Hf` at quality `0.85` and missing absolute `S` at quality `0.89`.
  Its 1-atm entropy is converted to the 1-bar thermochemical standard. `Hf`
  is refused for excluded ring/bond/correction topologies; S-S compounds keep
  `S` at reduced quality `0.75`.
- If `Hcomb_gross` exists, convert it to net `Hcomb` by adding the
  vaporization enthalpy of produced water.
- If `Hcomb` is still missing, calculate net combustion from formula
  stoichiometry and reactant phase at 298.15 K.

Combustion reactant state policy:

- Gas reactants use gas `Hf`.
- Liquid reactants prefer `Hf_liquid`; if unavailable, use gas `Hf -
  Hvap(298.15 K)`.
- Solid reactants use `Hf_solid` only; no solid derivations are attempted.

The homogeneous `EquilibriumReactor` uses gas-standard `Hf`, absolute `S`, and
ideal-gas heat-capacity integration to construct component standard chemical
potentials and reaction `ln(K)` at temperature. It requires `Gf` independently
for a 298.15 K consistency check. Vapor reaction activities use fugacity
coefficients; liquid activities use the selected activity/EOS model and its
pure-liquid reference fugacity. All of these property sources are marked as
result-affecting in the quality report.

Supported combustion products are `CO2(g)`, `H2O(g)`, `SO2(g)`, `SiO2(s)`,
`N2(g)`, and elemental halogen standard states. Metal/oxide and phosphorus
combustion conventions are not inferred yet.

## Simulation Wiring

- `thermodynamics.Psat()` still uses the pointwise resolver. Canonical Psat
  construction exists but is not yet invoked during component initialization.
- Canonical Psat curves end at `Tc`; any critical-tangent continuation above
  `Tc` belongs to thermo/K-value initialization and is not vapor pressure.
- `Cp_ideal_gas()` and `Cp_liquid()` route PFD-provided Cp correlations through
  the resolver before using prebound scalar/Perry shortcuts.
- Liquid enthalpy integrates liquid Cp where possible.
- `Hvap_at_T()` delegates temperature-dependent Hvap, including Watson scaling,
  to `PropertyResolver`.
- Activity-model liquid density uses resolver liquid molar volume where
  available.
- Poynting corrections use resolver/prebound liquid molar volume, with endpoint
  clamping when needed.
- EOS setup uses hydrated scalar properties from `ChemicalDatabase`.
- Supported pure-fluid EOS anchors are globally hydrated from one CoolProp
  backend unless a PFD override is present. Coupled identities are recomputed
  after PFD application before EOS construction.
- `.pfr` output includes a quality report section flagging low-quality property
  resolutions.

### Permanent-Solid Process Assembly

`phase_behavior=permanent_solid` is an explicit process assertion and is never
inferred from a component's ordinary phase at STP. Thermodynamic construction
retains three component contracts:

```text
process_components          all declared material-balance components
components                  conventional components compiled into fluid backends
permanent_solid_components  components assigned directly to the solid inventory
```

EOS, activity-coefficient, and VLLE matrices are compiled only from
`components`. A stream calculation converts total composition to component
flows, assigns permanent solids directly, flashes the normalized fluid
subtotal, scales fluid phases back to the total-stream basis, and attaches the
solid property contributions afterward. Dry-solid states skip fluid
equilibrium entirely.

Bulk sensible properties use total-stream mole weighting. Solid enthalpy and
entropy use the retained pure-solid kernels and reference properties without
an ideal solid mixing term. Bulk molar volume is additive across vapor,
liquid-1, liquid-2, and pure solid-component volumes. This is a thermodynamic
volume estimate, not powder bulk density: interparticle voidage, slip, settling,
and slurry rheology require later transport/material-form models.

## Future Work: Mixture Liquid Density and Excess Volume

Pure-component liquid volume is now strong enough that mixture excess volume is
the next real limitation for density-sensitive liquid calculations.

Activity models should not be used for `V^E` unless pressure-dependent
parameters are available:

```text
V^E = (dG^E/dP)_T,n
```

Ordinary NRTL/UNIQUAC/Wilson/UNIFAC parameter sets are temperature-dependent,
not pressure-dependent, so they do not provide trustworthy excess volumes.

Suggested architecture:

1. Store native Perry binary solution-density data first.
   Direct tables should stay as reconstructed density grids, and Table 2-63
   rows should stay as native polynomial density correlations. Use native
   evaluators directly only when the liquid phase is exactly the supported
   binary, such as water + ethanol.

2. Derive binary excess-volume models from native density data/correlations:

   ```text
   V_mix = MW_mix / rho_mix
   V^E = V_mix - x1*V1 - x2*V2
   V^E_ij = x_i*x_j*sum_k A_k,ij*(x_i - x_j)^k
   ```

   Keep the native source as truth; Redlich-Kister is the composable runtime
   representation.

3. Use binary `V^E` models for multicomponent mixtures:

   ```text
   V_mix = sum_i x_i*V_i + sum_pairs V^E_ij
   ```

   Pairwise `V^E` ignores true ternary terms, but it composes cleanly and is
   better than trying to combine direct density tables.

4. Use Spencer-Danner-Li modified Rackett as the predictive fallback for
   organic-organic mixtures and uncovered pairs. Perry reports about 7% average
   uncertainty.

   ```text
   phi_i = x_i*Vc_i / sum_j x_j*Vc_j
   Tc_ij = sqrt(Tc_i*Tc_j)
   Tc_mix = sum_i sum_j phi_i*phi_j*Tc_ij
   Tr = T / Tc_mix
   q = 1 + (1 - Tr)^(2/7)
   Z_RA,m = sum_i x_i*Z_RA,i
   V = R*sum_i(x_i*Tc_i/Pc_i)*Z_RA,m^q
   ```

   Use fitted pure-component `Z_RA` values when available:
   fitted from local/Perry/provided density, then PubChem-fitted with hard
   criticals, then Yamada-Gunn from omega.

5. Hybrid water-organic handling:

   ```text
   V_mix = sum_i x_i*V_i + V^E_org_predictive + sum_j V^E_water,j
   V^E_org_predictive = x_org*(V_SDL(y_org) - sum_org y_i*V_i)
   V^E_water,j = (x_w + x_j)*V^E_binary(y_w, y_j)
   y_w = x_w/(x_w + x_j)
   y_j = x_j/(x_w + x_j)
   ```

   Water tends to break corresponding-states predictions, but Perry has good
   real water-organic density data. Use real water-organic binary terms where
   available and SDL for the organic submixture.

Suggested mixture-volume quality ranking:

| Method | Suggested quality |
| --- | --- |
| Direct Perry solution-density table, in range | About `0.995`. |
| Perry/ICT polynomial solution-density correlation, in range | About `0.97-0.98`. |
| Redlich-Kister fit derived from native Perry density, in range | Source quality minus about `0.01-0.02`, depending on residual. |
| Mild Redlich-Kister extrapolation | About `0.90-0.94`, with warning. |
| Spencer-Danner-Li with hard criticals and fitted `Z_RA` anchors | About `0.84-0.86`. |
| Spencer-Danner-Li with hard criticals and Yamada-Gunn `Z_RA` | About `0.80-0.82`. |
| Spencer-Danner-Li with soft criticals | Demote by input quality propagation. |
| Ideal pure-volume mixing | Fallback only; warn for aqueous, polar, associating, electrolyte, and salt systems. |
