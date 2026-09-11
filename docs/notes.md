Current property-system notes

Still-valid architectural reminders:

- Normal property requests should continue to go through `PropertyResolver`, either
  directly or through `ChemicalProperties`/`thermodynamics.py`.
- Phase-specific heat capacity is the right direction. Liquid Cp should be liquid
  data first, gas Cp should be gas/ideal-gas data first, and ideal-gas Cp should
  only be a lower-quality fallback for liquids.
- Cp improvements should also improve enthalpy. Liquid enthalpy uses liquid-Cp
  integration where available, Hvap can be temperature-dependent, and NIST Cp
  constant/fit forms have analytic integral hooks.
- Activity-model mixture Cp should respect vapor fraction and phase
  compositions. The current code does this through `mixture_Cp()` and
  `phase_weighted_mixture_Cp()`.
- Plain cubic EOS liquid volumes should still be treated cautiously. Prefer
  measured/correlated liquid density or molar volume when possible; use cubic
  liquid volumes as a pressure-aware fallback with sanity checks.
- Property results should continue carrying source, quality, range, and notes
  internally. That metadata is the only sane way to debug surprising simulation
  results later.
- Quality propagation rule: when a calculation combines several input
  properties, start from the minimum input quality and multiply by the method's
  own quality factor. Exact formulas use source `exact` and factor 1.0 only
  when every input is real/source-backed; otherwise they stay calculated and
  carry the non-exact method factor.

Remaining gaps / watch points:

- Viscosity resolution exists, including provided correlations and Perry tables,
  but unit operations do not appear to consume viscosity yet.
- Online Cp lookup now has a NIST tabulated-data path. It intentionally ignores
  Shomate coefficient tables and does not try PubChem free-text Cp snippets.
- Some non-activity state-density paths still have crude liquid fallbacks, such
  as `800/MW`, when EOS or resolver density is unavailable.
- Group-contribution property estimation remains mostly a future/fallback idea,
  not a substitute for real local data.


Updated lookup system

Identity first:

- Most resolver paths call `_identifier_candidates(identifier, props)`, which
  uses the centralized identity resolver with the input identifier plus any
  provided symbol, name, formula, CAS, and identifier metadata.
- Formula aliases are conservative; ambiguous formulas should not be blindly
  accepted.
- `ChemicalDatabase.get()` tries direct/local lookup first, then identity and
  local aliases/SMILES/CAS, then local Smith textbook and Perry fetches, and
  online PubChem/NIST only when enabled.

Hydration policy:

- `ChemicalDatabase` hydrates missing scalar fields once per cached chemical:
  `Tc`, `Pc`, `Vc`, `Zc`, `omega`, `Tb`, `Tm`, `Hvap`, `Hfus`, `Hf`, `Gf`, `S`,
  and `Hcomb`.
- Temperature-dependent properties are not flattened into `ChemicalProperties`.
  They stay as resolver calls so valid ranges and equation forms are checked at
  the requested temperature.
- EOS constructors read hydrated `ChemicalProperties`, so Perry/textbook/local
  critical constants can now feed EOS setup indirectly through database
  hydration.

Psat:

In `PropertyResolver.resolve_vapor_pressure(symbol, T, props)`:

1. `vapor_pressure_tables.json`, in range only.
2. Provided portable `Psat` correlation from `property_correlations`, in range
   only.
3. Provided Antoine coefficients from props, normally `chemicals.json`, in
   range only.
4. Perry vapor-pressure correlation, in range only.
5. Other local Antoine candidates, including provided Antoine, Smith textbook,
   and `data/antoine.txt`, in range only.
6. Online Antoine only when the same row covers the normal boiling point and
   reproduces 1 atm at `Tb`.
7. Ambrose-Walton using provided/resolved `Tc`, `Pc`, and `omega`.
8. Lee-Kesler using provided/resolved `Tc`, `Pc`, and `omega`.
9. Any other in-range cached/online Antoine row.
10. Near-range local Antoine extrapolation.
11. Clausius-Clapeyron using `Tb` and `Hvap`.
12. Online `Hvap` plus Clausius-Clapeyron when `Tb` exists.
13. Far local Antoine extrapolation.
14. Error with missing-property detail.

Tc, Pc, Vc, Zc, omega:

In `PropertyResolver.resolve_critical_properties()`:

1. Provided props.
2. Calculate `Zc` from provided/resolved `Tc`, `Pc`, and `Vc` when possible.
3. Perry critical constants.
4. Smith textbook properties.
5. Online PubChem critical lookup when allowed.
6. Online NIST phase-change critical rows for missing `Tc`, `Pc`, or `Vc`.
7. Estimate `omega` from Lee-Kesler when `Tc`, `Pc`, and `Tb` are available and
   estimation is allowed.
8. If real `Tc` and one real member of `Pc`/`Vc` are available, replace a
   missing or estimated counterpart with a Pitzer-Zc back-calculation:
   `Zc = 0.2905 - 0.0787*omega`. When `Pc` is the missing/estimated value,
   omega is solved consistently with Lee-Kesler and the Pitzer Zc relation.
9. Missing result.

Tb:

In `PropertyResolver.resolve_boiling_point()`:

1. Provided props.
2. Perry normal boiling point.
3. Smith textbook `Tb`.
4. Online PubChem phase-change data when allowed.
5. Online NIST phase-change data when allowed.
6. Rough MW estimate when estimation is allowed.
7. Missing result.

Tm:

In `PropertyResolver.resolve_melting_point()`:

1. Provided props.
2. Perry normal melting point.
3. Online PubChem phase-change data when allowed.
4. Online NIST phase-change data when allowed.
5. Missing result.

Hvap:

In `PropertyResolver.resolve_hvap(symbol, props, T=None)`:

1. Provided portable `Hvap` correlation when a target temperature is supplied.
2. Perry heat-of-vaporization correlation.
3. Online PubChem and NIST phase-change lookup when allowed. PubChem scalar
   values have priority over NIST scalar values, but an accepted NIST
   temperature-dependent Watson-style fit beats scalar `Hvap` values.
4. When a target temperature is supplied, Watson scaling is attempted before
   returning any scalar. The normal-boiling reference is chosen from, in order:
   `Hvap(Tb)` from provided/Perry/NIST temperature-dependent fits, provided
   scalar `Hvap`, Smith textbook `Hvap`, then online scalar Hvap when online is
   allowed. Generic Watson scaling applies a `0.92` quality multiplier.
5. Provided scalar `Hvap`.
6. Smith textbook `Hvap`.
7. Legacy PubChem Hvap lookup hook.
8. Nannoolal Part-3 slope with a strict Peng-Robinson `delta Z` through
   `Tr=0.8`, after every source-backed `Hvap(Tb)`/Watson route. Fragmentation,
   PR-root, and temperature-domain failures fall through.
9. Trouton-style estimate from `Tb` when estimation is allowed. When `Tc` is
   usable, Watson-scale the estimate with quality
   `0.72 * min(Tb quality, Tc quality)`; otherwise retain the constant estimate
   with quality `0.55 * Tb quality`.
10. Missing result. Carboxylic acids are refused by both estimation rungs.

Hfus:

In `PropertyResolver.resolve_hfus()`:

1. Provided scalar `Hfus`.
2. Perry heat of fusion.
3. Online PubChem phase-change data when allowed.
4. Online NIST phase-change data when allowed, using a median of the tabulated
   fusion enthalpies when no scalar average is present.
5. Missing result.

Liquid Cp:

In `PropertyResolver.resolve_heat_capacity(..., phase='liquid')`:

1. Provided portable `Cpl` correlation, in range only.
2. Provided constant `Cp_liquid`.
3. Perry liquid-Cp correlation, in range only.
4. If no constant `Cp_liquid` exists, provided `Cpl` or Perry liquid Cp can be
   evaluated up to 20 K outside its tabulated range; farther outside the range
   clamps to that 20 K extrapolated boundary value.
5. Online NIST tabulated liquid Cp, collapsed to an average constant value after
   duplicate-temperature median cleanup.
   - This constant has an analytic enthalpy integral.
6. Provided ideal-gas `Cp_coeffs` as a lower-quality fallback, multiplied by
   1.3 for liquid use.
7. Rough MW estimate.
8. Error.

Ideal-gas / vapor Cp:

In `PropertyResolver.resolve_heat_capacity(..., phase='ideal_gas'/'vapor')`:

1. Provided portable `Cpg` correlation, in range only.
2. Provided ideal-gas `Cp_coeffs`.
3. Perry ideal-gas Cp correlation, in range only.
4. Provided `Cpg` or Perry ideal-gas Cp can be evaluated up to 20 K outside its
   tabulated range; farther outside the range clamps to that 20 K extrapolated
   boundary value.
5. Online NIST tabulated gas Cp after duplicate-temperature median cleanup:
   10+ points use a Shomate-style fit with outlier exclusion; 4-9 points use a
   linear fit with extrapolation warning; 1-3 points use interpolation or clamp
   to the nearest tabulated endpoint.
   - Shomate-style and linear fits have analytic enthalpy integrals; the
     interpolation/clamp fallback remains a pointwise Cp source.
6. Rough MW estimate.
7. Error.

Liquid molar density / volume:

In `PropertyResolver.resolve_liquid_molar_density()` and
`resolve_liquid_molar_volume()`:

1. Provided portable mass-density correlation `rhol`, converted using MW.
2. Perry liquid-density correlation, converted as needed.
3. Up to 20 K extrapolation of the provided/Perry density correlation.
4. Rackett using cached `Z_RA` fitted once from the available density
   correlation.
5. Online PubChem liquid-density text lookup when local/correlated density is
   absent. The parser is deliberately conservative:
   - accepts explicit density units, relative density/specific gravity, density
     ranges, and simple bare density values only from PubChem `Density` rows;
   - rejects vapor/gas/air/saturated-air rows, aqueous/commercial/mixture
     rows, solids/ice/molten rows, bulk/powder density, obvious estimates, and
     invalid temperatures;
   - clusters accepted points after a small 298 K normalization and discards
     inconsistent outliers.
6. If PubChem succeeds and hard `Tc`/`Pc` are available, fit `Z_RA` from the
   PubChem density cluster and evaluate Rackett.
7. If PubChem succeeds but the critical pair is missing or soft, use the
   nearest PubChem density point with a thermal-expansion heuristic:
   `V(T) = V_ref * exp(9e-4 * (T - T_ref))`, with the expansion factor clamped
   to 0.70-1.50.
8. PTV EOS if `Zc` is hard, Rackett/Yamada-Gunn otherwise, then plain PR.
9. Last-resort formula/MW liquid-volume heuristic.

The PubChem density path uses the normal online cache and terminates the chain
when it succeeds, because even sparse experimental density is preferred over
predictive EOS/corresponding-states methods.

EOS liquid-volume fallback pressures are floored at 1 Pa for numerical
stability. `resolve_liquid_molar_volume_nearest()` now shares the same resolver
chain; old endpoint clamping is only retained as a small, prebound
extrapolation optimization in thermodynamics.

Vapor / liquid viscosity:

In `PropertyResolver.resolve_viscosity(symbol, T, phase)`:

1. Provided portable viscosity correlation: `mug` for vapor/gas, `mul` for
   liquid.
2. Perry vapor or liquid viscosity correlation, in range only.
3. Error.

This is still mostly lookup/API support; the simulator does not appear to use
viscosity in unit operations yet.

Formation properties and entropy:

In `PropertyResolver.resolve_formation_properties()`:

1. Provided `Hf`, `Gf`, absolute gas `S`, and `Hcomb`.
2. Perry ideal-gas formation table at 298.15 K.
3. Online NIST WebBook gas/liquid thermochemistry rows when allowed. The
   resolver prefers direct `AVG`/`Review` rows and otherwise takes the median
   of direct rows. Gas `S` and liquid `S_liquid` are absolute standard molar
   entropies, not formation entropies.
4. If gas `Hf` is still missing, derive it from liquid `Hf + Hvap(298.15 K)`.
   Liquid `Hf` comes first from local/provided `Hf_liquid`, then from NIST
   liquid thermochemistry when online is allowed. Solid conversions are not
   attempted yet.
5. Once gas `Hf` is available, complete gas `Gf` or absolute gas `S` from the
   other gas value using elemental standard entropies from
   `data/S_table_elements.txt`.
6. If gas `Gf` is still missing, derive it from `Gf_liquid` plus
   `-RT ln(Psat/1 bar)` at 298.15 K, then complete gas `S` if possible.
7. If only `S_liquid` is available, derive `Gf_liquid` from liquid `Hf` and
   absolute liquid entropy, then derive gas `Gf` and gas `S`.
8. As the final estimation rung, use Domalski-Hearing gas `Hf` and absolute
   gas `S` before leaving either field missing. `Hf` uses quality 0.85 only
   inside the validated structural applicability domain. `S` uses quality
   0.89 after conversion from 1 atm to 1 bar, reduced to 0.75 for S-S bonded
   compounds.
9. If `Hcomb` is missing but `Hcomb_gross` is present, convert gross to net by
   adding the 298.15 K vaporization enthalpy for the produced water.
10. If `Hcomb` is still missing, calculate net combustion from formula
   stoichiometry and the reactant's 298.15 K phase enthalpy. Gas reactants use
   gas `Hf`; liquids prefer `Hf_liquid` and otherwise use gas `Hf -
   Hvap(298.15 K)`; solids use `Hf_solid` only. Supported products are
   `CO2(g)`, `H2O(g)`, `SO2(g)`, `SiO2(s)`, `N2(g)`, and elemental halogen
   standard states. Metal/oxide and phosphorus combustion conventions are not
   inferred yet.
11. Missing result per field.

`ChemicalProperties` now has fields for `S`, `Hcomb`, and phase-specific
formation values, and database hydration fills gas-standard values when
available or derivable.

Simulation wiring summary:

- `thermodynamics.Psat()` uses the resolver.
- `Cp_ideal_gas()` and `Cp_liquid()` use prebound provided/Perry data and fall
  back to the resolver.
- Liquid enthalpy integrates liquid Cp where possible.
- `Hvap_at_T()` delegates temperature-dependent Hvap, including Watson scaling,
  to `PropertyResolver`.
- Activity-model liquid density uses resolver liquid molar volume where
  available.
- Poynting corrections use resolver/prebound liquid molar volume, with endpoint
  clamping when needed.
- EOS setup uses hydrated scalar properties from `ChemicalDatabase`.


-------- Cutoff --------

Composable property methods:

- Vapor fugacity
- Liquid fugacity/activity
- Excess enthalpy
- Molar volume / liquid density
- Vapor pressure
- Poynting correction
- Transport properties, especially viscosity, when future unit ops need them
- Source/range/quality metadata for all of the above


Quality table:

(These are minimums. If you, for example, believe that a property is more trustworthy than its category, you can put it higher.)

1.00 - User-specified authoritative value, exact definitions, exact math  
0.99+ - Direct, validated experimental value under matching conditions 
0.97+ - High-quality curated database / critically evaluated correlation    
0.95+ - Textbooks, trusted local databases, chemicals.json
0.9+ - Lower priority local databases, direct reputable online source, very standard correlation, near-exact estimates
0.85+ - Good fitted correlation, minor extrapolation, strong corresponding-states estimate
0.8+ - Medium-quality estimate/source, ordinary corresponding-states estimate, group contribution method with validated accuracy
0.7+ - Unverified online value, moderate extrapolation, weaker fitted correlation, decent group contribution
0.6+ - Motivated rough estimate, broad group-contribution method
0.4+ - Heuristic, fallback correlation outside its comfort zone
0.2+ - Assumed typical/default value
0.0 - Unknown, placeholder, deliberately unsafe/unresolved



Future work: mixture liquid density and excess volume

Pure-component liquid volume is now in good shape, but mixture liquid density
still mostly lacks excess-volume support. This is lower priority, but it is the
next real limitation for density-sensitive liquid calculations.

Why activity models are not enough:

- Ordinary NRTL/UNIQUAC/Wilson/UNIFAC parameter sets are temperature-dependent,
  not pressure-dependent.
- Excess volume is related to the pressure derivative of excess Gibbs energy:
  `V^E = (dG^E/dP)_T,n`.
- Without pressure-dependent interaction parameters, gamma models cannot give a
  trustworthy `V^E`. Treat them as activity models, not volume models.

Useful Perry sources identified:

- Perry PDF page 143, printed page 2-108: `TABLE 2-58 Ethyl Alcohol`.
  This is a direct aqueous ethanol density table versus mass percent and
  temperature. It is high-quality experimental tabulated solution density.
  Extraction is annoying because repeated leading digits are omitted.
- Perry PDF pages 146-147, printed pages 2-111 to 2-112: `TABLE 2-63
  Densities of Aqueous Solutions of Miscellaneous Organic Compounds`.
  This gives polynomial solution-density correlations from International
  Critical Tables:
  - Section A: `d = dw + A*ps + B*ps^2 + C*ps^3`, where `ps` is wt% solute.
  - Section B: `d = ds + A*pw + B*pw^2 + C*pw^3`, where `pw` is wt% water.
  - Section C: `dt = d0 + A*t + B*t^2` at fixed composition.
  The coefficient notation is compact: e.g. `0.03255` means
  `2.55e-4`, so extraction must account for Perry's scaling convention.
- Perry PDF pages 385-386, printed pages 2-350 to 2-351: Spencer-Danner-Li
  modified Rackett mixture method. Perry states about 7% average uncertainty,
  higher near a component critical temperature. This is predictive, not an
  experimental `V^E` model.

Suggested architecture:

1. Store native Perry binary solution-density data first.
   - Direct tables should stay as reconstructed density grids.
   - Table 2-63 rows should stay as native polynomial density correlations.
   - Use native evaluators directly when the liquid phase is exactly the
     supported binary, such as water + ethanol.

2. Derive binary excess-volume models from native density data/correlations.
   Convert source density to molar volume:
   `V_mix = MW_mix / rho_mix`
   and compute:
   `V^E = V_mix - x1*V1 - x2*V2`.
   Fit Redlich-Kister form:
   `V^E_ij = x_i*x_j*sum_k A_k,ij*(x_i - x_j)^k`.
   Keep the native source as the truth; the Redlich-Kister fit is the
   composable runtime representation.

3. Use binary `V^E` models for multicomponent mixtures.
   Direct density tables do not compose cleanly. For 3+ components use:
   `V_mix = sum_i x_i*V_i + sum_pairs V^E_ij`.
   Pairwise `V^E` is approximate and ignores true ternary terms, but it is
   structured and composable.

4. Use Spencer-Danner-Li modified Rackett as the predictive fallback.
   Perry mixture rules:
   - `phi_i = x_i*Vc_i / sum_j x_j*Vc_j`
   - `Tc_ij = sqrt(Tc_i*Tc_j)`
   - `Tc_mix = sum_i sum_j phi_i*phi_j*Tc_ij`
   - `Tr = T / Tc_mix`
   - `q = 1 + (1 - Tr)^(2/7)`
   - `Z_RA,m = 0.29056 - 0.08775*sum_i x_i*omega_i`
   - `V = R*sum_i(x_i*Tc_i/Pc_i)*Z_RA,m^q`

5. Take advantage of fitted pure-component `Z_RA` values when available.
   Since Yamada-Gunn is equivalent to `Z_RA,i = 0.29056 - 0.08775*omega_i`,
   use:
   `Z_RA,m = sum_i x_i*Z_RA,i`
   where `Z_RA,i` is chosen from:
   - fitted from local/Perry/provided pure density;
   - fitted from PubChem density with hard criticals;
   - Yamada-Gunn from omega.
   This preserves pure-component limits and anchors SDL to known pure liquid
   densities.

6. Hybrid water-organic handling.
   Water tends to break corresponding-states volume predictions, but Perry has
   good real data/correlations for water + many organics. For mixtures with
   water and multiple organics, use:
   `V_mix = sum_i x_i*V_i + V^E_org_predictive + sum_j V^E_water,j`.
   For the organic submixture:
   `V^E_org_predictive = x_org*(V_SDL(y_org) - sum_org y_i*V_i)`,
   where `y_i = x_i/x_org`.
   For each water-organic pair, evaluate the binary model in its own binary
   composition space and scale it back to the full mixture:
   `V^E_water,j = (x_w + x_j)*V^E_binary(y_w, y_j)`,
   where `y_w = x_w/(x_w + x_j)` and `y_j = x_j/(x_w + x_j)`.

Suggested quality ranking:

- Direct Perry solution-density table, in range: about 0.995.
- Perry/ICT polynomial solution-density correlation, in range: about 0.97-0.98.
- Redlich-Kister fit derived from native Perry density data, in range:
  source quality minus about 0.01-0.02 depending on residual.
- Mild Redlich-Kister extrapolation: about 0.90-0.94, with warning.
- Spencer-Danner-Li with hard criticals and fitted `Z_RA` anchors: about
  0.84-0.86.
- Spencer-Danner-Li with hard criticals and Yamada-Gunn `Z_RA`: about 0.80-0.82.
- Spencer-Danner-Li with soft criticals: demote by input quality propagation.
- Ideal pure-volume mixing: use only as fallback; warn for aqueous, polar,
  associating, electrolyte, and salt systems.
