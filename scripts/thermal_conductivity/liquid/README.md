# Liquid thermal-conductivity investigations

This directory contains the reproducible benchmarks, refit probes, external
validation data, and result artifacts used to select pfdsim's liquid
thermal-conductivity estimators.

The scripts answer three separate questions:

1. Which specialized method should be preferred when its chemical class and
   inputs are available?
2. Where do the published methods cease to be reliable?
3. Which local corrections are supported by whole-compound validation or
   external data?

Start with the three decision artifacts:

| Chemical domain | Decision script | Production status |
|---|---|---|
| Selected oxygenated classes and C2+ halocarbons | `benchmark_final_baroncini.py` | Implemented as Baroncini |
| C3+ hydrocarbons | `benchmark_hydrocarbon_model_perry.py` | Implemented as Modified Pachaiyappan |
| Broad structure-based fallback | `benchmark_revised_govender_perry.py` | Implemented with local refits |

Direct supplied correlations, Perry correlations, and Perry saturated-liquid
tables remain preferable to every estimator described here.

## Final recommendation

The intended liquid-conductivity selection order is:

1. Supplied/PFD liquid-conductivity correlation.
2. In-range Perry liquid-conductivity correlation.
3. In-range Perry saturated-liquid tabulation.
4. Modified Pachaiyappan for hydrocarbons with at least three carbon atoms.
5. Baroncini for its selected oxygenated and halocarbon domains.
6. Revised Govender as the broad structure-based fallback.

The current production resolver implements this complete order.

### Modified Pachaiyappan

Production implementation: `../../../modified_pachaiyappan.py`

Resolver integration: `../../../property_resolution/thermal_conductivity.py`

Applicability:

- Hydrocarbons only.
- At least three carbon atoms.
- An acyclic, unbranched carbon skeleton uses the straight-chain parameter
  pair, including linear alkenes and alkynes.
- Branched, cyclic, and aromatic hydrocarbons use the other parameter pair.

The method uses trusted liquid molar volume at 293.15 K when its resolved
quality is at least 0.86. Otherwise it uses the method-local model

\[
V_{20}=28.0821+8.26376n_C+3.80982n_H-10.3776n_{\mathrm{rings}}
\]

in cm3/mol. The model gives 1.72% compound-held-out volume MAPE and 2.70%
conductivity MAPE for C3+ hydrocarbons. Nannoolal critical temperature is also
adequate: the fully structure-based path gives 2.32% conductivity MAPE on the
common C3+ population.

With trusted Perry `Tc` and `V20`, Modified Pachaiyappan gives 2.42% MAPE on
the exact 69-compound common set where Baroncini gives 8.98%. It should remain
the preferred hydrocarbon method.

### Final Baroncini domain

Decision artifact: `results/final_baroncini_benchmark.txt`

Production implementation: `../../../baroncini_method.py`

Resolver integration: `../../../property_resolution/thermal_conductivity.py`

Baroncini is recommended for:

| Category | Selected applicability | Final fitted MAPE |
|---|---|---:|
| Alcohols | One nonphenolic alcohol group | 4.90% |
| Organic acids | Mono- or dicarboxylic; formic acid excluded | 2.57% |
| Ketones | Acyclic aliphatic monoketone local correction; aromatic monoketones retain published parameters | 2.29% |
| Aldehydes | Aliphatic single-aldehyde compounds with local `A,b` refit | 2.21% |
| Esters | Monoesters; unsaturated and aromatic monoesters allowed | 2.69% |
| Ethers | Acyclic ethers | 1.76% |
| Halogenated hydrocarbons | At least two carbon atoms | 5.83% |

The complete selected population contains 140 compounds and gives 3.71% MAPE,
1.80% median absolute error, 13.07% P95 error, and -1.67% bias. These aggregate
metrics contain full-data ketone and aldehyde fits. Their whole-compound
validation estimates are 3.01% and 2.87% MAPE, respectively.

Baroncini exclusions:

- Ordinary hydrocarbons, because Modified Pachaiyappan is substantially more
  accurate.
- Phenols.
- Diols and other polyols.
- Formic acid.
- Diesters.
- Cyclic ethers.
- Cyclic-carbonyl ketones.
- Aromatic aldehydes. Benzaldehyde externally failed at 20.10% MAPE with the
  aliphatic aldehyde refit.
- One-carbon refrigerants. The authors' `A=0.562` correction reduces aggregate
  bias but does not reliably control individual errors.

The selected acyclic aliphatic ketone multiplier is

\[
\exp\left(1.023417-0.243040\ln M+0.0409366s\right),
\]

where `M` is in g/mol and `s` is the number of carbon atoms on the smaller side
of the carbonyl. The aldehyde parameters are `A=0.0110044` and `b=0.728470`.

Input hierarchy for Baroncini:

1. Trusted `Tb` and `Tc`: 3.73% MAPE on the 139-compound sensitivity set.
2. Trusted `Tb` and Nannoolal `Tc` anchored by that `Tb`: 3.93%.
3. Nannoolal `Tb` with trusted `Tc`: 4.32%.
4. Fully structure-based Nannoolal `Tb/Tc`: 5.03%.

Halocarbons have `a=0`, so replacing `Tb` has no numerical effect for that
class.

Resolver base qualities are 0.90 for ethers, aldehydes, and ketones; 0.88 for
acids and monoesters; 0.84 for alcohols; and 0.80 for halogenated
hydrocarbons. Non-halocarbon `Tb` quality from 0.75 through 0.84 incurs -0.03;
lower quality refuses Baroncini. `Tc` quality from 0.80 through 0.89 incurs
-0.01, quality from 0.70 through 0.79 incurs -0.03, and lower quality refuses
Baroncini. Halocarbons are exempt only from the explicit `Tb` requirement and
penalty; the shared critical resolver may still use a resolved real `Tb` to
anchor an estimated `Tc`.

### Revised Govender

Production implementation: `../../../govender_method.py`

Resolver integration: `../../../property_resolution/thermal_conductivity.py`

Govender remains the broad fallback. Local refits are enabled by default;
`local_refits=False` reproduces the published coefficient path. Production
local changes include:

- Aliphatic carboxylic-acid group 53 refit.
- Ordinary-ketone group 58 refit.
- Acyclic-thioether group 100 refit.
- Tertiary-amine group 84 refit.
- One amine/alcohol correction per molecule.
- Nonaromatic three-, four-, and five-membered-ring corrections.

Cyclic thioethers and unsaturated cyclic ketones are unconditional refusal
domains. The production-supported Perry population gives 7.75% MAPE over 259
compounds. A diagnostic C3+ screen gives 6.65% MAPE over 219 compounds.

## Script map

### Final and baseline benchmarks

- `benchmark_final_baroncini.py` is the authoritative selected-domain
  Baroncini benchmark. It combines Perry and CoolProp 7.2 references, applies
  all accepted exclusions and local adjustments, reports each category, and
  compares trusted inputs with Nannoolal `Tb` and both Nannoolal `Tc` modes.
- `benchmark_baroncini_perry.py` is the broad first-pass Baroncini benchmark.
  It deliberately uses permissive structural definitions so failures reveal
  applicability boundaries. Do not treat its 7.16% aggregate MAPE as the
  recommended final domain.
- `benchmark_hydrocarbon_model_perry.py` benchmarks the production Modified
  Pachaiyappan equation with trusted Perry `Tc` and `V20`, and compares it with
  revised Govender on an exact common set.
- `benchmark_govender_perry.py` benchmarks the published Govender coefficient
  path. It supports Perry `Tb` and Nannoolal `Tb` modes and establishes the
  original 13.53% Perry-Tb baseline.
- `benchmark_revised_govender_perry.py` calls the production Govender
  implementation with local refits. Its `all` domain honors production
  refusals; `screened` additionally removes compounds with fewer than three
  carbons.

### Baroncini scope and refit probes

- `analyze_baroncini_class_selection.py` compares Baroncini with Govender by
  oxygenated class, compares Baroncini with Modified Pachaiyappan for
  hydrocarbons, examines size trends, and rejects unsupported alcohol
  corrections through leave-one-compound-out validation.
- `probe_baroncini_ketone_refit.py` selects the acyclic aliphatic ketone
  correction, tests aldehyde applicability, fits the aliphatic aldehyde
  parameters, and externally validates benzaldehyde. It is the source of the
  ketone and aldehyde parameters consumed by the final benchmark.
- `probe_baroncini_refrigerants.py` evaluates the general and author-corrected
  refrigerant parameters against Perry halocarbons, split by carbon count,
  halogen family, halogen count, and topology.
- `probe_baroncini_mixed_halogen_coolprop.py` evaluates mixed-halogen
  refrigerants against CoolProp HEOS saturated-liquid conductivity through
  `Tr=0.95`.
- `probe_baroncini_nonmixed_halogen_coolprop.py` evaluates non-mixed CoolProp
  halocarbons absent from the Perry probe.
- `compare_baroncini_govender_refrigerants.py` compares corrected Baroncini and
  revised Govender on the exact common Perry/CoolProp refrigerant population.
  Baroncini gives 6.85% MAPE versus Govender's 10.83%.

The `coolprop8` text and JSON artifacts were generated with an isolated
CoolProp 8.0 installation. The normal default scripts use the repository
environment's CoolProp version and the non-suffixed artifacts. CoolProp 8 adds
fluid registrations but no additional usable mixed-halogen conductivity
models; it changes the R1234yf reference enough to alter the non-mixed summary
slightly.

### Modified Pachaiyappan probes

- `probe_hydrocarbon_v20_estimation.py` compares simple structure models for
  `V20`. It selects the C/H/ring-count equation used by production when a
  resolved 20 °C volume has quality below 0.86.
- `probe_hydrocarbon_tc_sensitivity.py` compares trusted Perry `Tc`, Nannoolal
  `Tc` anchored by Perry `Tb`, and fully structure-based Nannoolal `Tc` on
  identical hydrocarbon states.

### Govender coefficient and domain probes

- `probe_govender_acid_group.py` identifies the published group-53 acid
  discrepancy and compares a likely transcription correction with a full
  Perry refit.
- `probe_govender_group_transcriptions.py` evaluates one-character coefficient
  changes, unconstrained group refits, and compound-held-out behavior for
  suspect nonacid groups. Its diagnostics also contain the selected ordinary
  ketone and acyclic-thioether subclass refits.
- `probe_govender_ring_corrections.py` fits three- and five-membered-ring
  corrections, interpolates the four-membered value, holds cyclobutane out,
  and validates the five-member term against NMP.
- `probe_govender_amine_refits.py` combines Perry and external literature data
  to fit the tertiary-amine group and compare presence-based versus per-OH
  amine/alcohol interactions.
- `validate_govender_sparse_candidates.py` evaluates Perry-derived candidate
  changes for groups 33, 72, and 93 against external isoprene, DMAc, NMP, and
  azepane data. It is why those published coefficients remain unchanged.

## Data files

- `data/govender_sparse_group_literature.json` contains the externally supplied
  isoprene, DMAc, NMP, and azepane observations used only for validation.
- `data/tertiary_alcohol_amines.json` contains pure tertiary-amine and
  aminoalcohol observations used by the group-84 and amine/alcohol probe.

The original Govender paper is stored at
`../../../data/reference/thermodynamic-models/2020-govender-group-contribution-liquid-thermal-conductivity.pdf`.

## Results directory

Every maintained run writes two files under `results/`:

- `.txt` is the concise human-readable report.
- `.json` contains complete per-compound metrics, exclusions, parameters, and
  reproducibility metadata.

Prefer JSON when comparing scripts or consuming results programmatically. Most
JSON artifacts record SHA-256 hashes for the executing script and important
dependencies. If a script changes, regenerate its artifact and every dependent
artifact rather than editing result files manually.

The most useful result files are:

- `final_baroncini_benchmark.*` — final recommended Baroncini domain.
- `hydrocarbon_model_perry_benchmark.*` — Modified Pachaiyappan benchmark.
- `revised_govender_perry_benchmark.*` — production Govender benchmark.
- `baroncini_ketone_refit_probe.*` — ketone/aldehyde refits and benzaldehyde
  external failure.
- `baroncini_vs_govender_refrigerants.*` — refrigerant method comparison.

## Running the principal benchmarks

Run from the repository root:

```bash
python3 scripts/thermal_conductivity/liquid/benchmark_final_baroncini.py
python3 scripts/thermal_conductivity/liquid/benchmark_hydrocarbon_model_perry.py
python3 scripts/thermal_conductivity/liquid/benchmark_revised_govender_perry.py
```

To reproduce a probe, run its script directly. Each script exposes
`--output-json`; its default points to the corresponding maintained artifact.
The scripts are offline except for whatever locally installed property
libraries provide. CoolProp probes use the installed HEOS fluid registry and
transport models.

## Interpreting reported accuracy

- Temperatures sampled from one correlation are not independent experimental
  observations. Chemical diversity is counted by compounds.
- Fitted full-data metrics are not external validation. Ketone and aldehyde
  artifacts therefore report whole-compound leave-one-out results separately.
- Perry liquid-conductivity and density correlations are DIPPR-derived and are
  treated as respected independent property references, but they are not raw
  experimental datasets.
- CoolProp comparisons use saturated-liquid conductivity at `Q=0` and stop at
  `Tr=0.95` to avoid conflating Baroncini's zero-at-`Tc` form with CoolProp's
  critical enhancement.
- Coverage must be considered with error. A method with a lower MAPE is not
  preferable if its inputs or structural domain are unsupported for the
  requested compound.
