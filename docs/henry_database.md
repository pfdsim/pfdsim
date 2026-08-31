# Henry 5.0 database build

PFDSim's runtime Henry database is built from R. Sander's official
**Henry 5.0.0** PostgreSQL archive:

- Source: <https://www.henrys-law.org/henry_data/henry_5.0.0_sql.zip>
- Publication: R. Sander, *Compilation of Henry's law constants (version
  5.0.0) for water as solvent*, Atmos. Chem. Phys. 23, 10901-12440 (2023)
- DOI: <https://doi.org/10.5194/acp-23-10901-2023>
- Expected archive SHA-256:
  `d90b7264a55ebb2c42fcca4b3483d0cc5361caa177224685cc66f3f7d7fb9d1b`

The official 5.0 format is not the flattened CSV used by the old 4.0 build.
It contains normalized PostgreSQL tables for species, values, literature,
notes, and note links, and its numeric fields use HTML scientific notation.
`scripts/build_henry_constants_db.py` stages that archive in SQLite, resolves
the relationships, and writes the smaller runtime database atomically.

## Suitability filtering

The thermodynamic model needs an infinite-dilution Henry solubility constant
for pure light water. The builder therefore excludes source rows explicitly
marked as:

- seawater (98 rows in the pinned 5.0.0 archive);
- concentrated brine, heavy water, or 60% aqueous ethanol;
- pH-specific or reactive effective constants;
- non-dilute total solubilities or non-Henry equilibrium definitions; or
- concentrated-solution extrapolations that are only order-of-magnitude
  estimates.

The source archive is hash-pinned, and the exceptional note labels are listed
in the builder. This avoids changing suitability behavior silently when an
upstream note changes. Rows marked `W` (wrong) and species without CAS numbers
are also unavailable to the CAS-keyed runtime database.

Values reported away from 298.15 K are normalized to 298.15 K when the same
row contains a suitable temperature coefficient and is within 40 K of the
reference. An uncorrected value within 7 K may be retained with a provenance
penalty. More distant uncorrected values and values with unknown temperature
are excluded. A temperature coefficient measured in a non-water medium or
defined for a pH-specific effective constant is excluded independently even
when that row's 298.15 K intrinsic constant remains usable.

## Aggregation and outliers

Sander's source types are ordered `L`, `M`, `V`, `R`, `T`, `X`, `C`, `Q`,
`E`, and `?`. PFDSim first selects the best available reliability band
(`L/M`, `V/R`, `T`, `X/C`, `Q/E`, or `?`) so numerous estimates cannot
outvote measured data. Individual type and provenance weights are applied
inside that band. Multiple rows from one literature reference share one
reference's total weight.

`Hcp` is combined as a weighted geometric mean because disagreement is
multiplicative. `B` is combined as a weighted arithmetic mean. Before
combining, the builder applies one-pass robust outlier filtering:

- `Hcp`: exclude values whose log-distance from the weighted median exceeds
  `max(ln(3), 3.5 * 1.4826 * MAD)`.
- `B`: exclude values whose distance from the weighted median exceeds
  `max(1000 K, 35% of the median magnitude, 3.5 * 1.4826 * MAD)`.

No outlier is removed when fewer than three candidates exist. Conflicting
two-source data remain visible and receive a poor agreement score rather than
having one value chosen arbitrarily.

## Confidence grading

Confidence is calculated separately from the weights used to resolve the
value. This prevents additional agreeing measurements from lowering a score
merely because a measured (`M`) row has a lower nominal type rank than a
review (`L`) row.

Evidence contributions are `L=1`, `M=1`, `V=0.75`, `R=0.65`, `T=0.45`,
`X=0.25`, `C=0.20`, `Q=0.15`, `E=0.10`, and `?=0.05`, multiplied by the
row's provenance multiplier. Duplicate rows from one literature reference
share that reference's contribution. `L` rows are clustered into review
families using known report series and normalized lead-author lineages.
Successive editions within a family share one contribution. Distinct review
families contribute with diminishing weights `1`, `1/2`, `1/4`, and so on,
recognizing additional expert evaluations without treating them like new
experiments. `M`, `V`, and `R` contribute to the separately reported
direct-evidence total.

Raw agreement uses the weighted RMS dispersion around the resolved value.
Because perfect agreement between two rounded review values is weak evidence,
agreement is shrunk toward 0.5 according to total effective evidence `E`:

`information = 1 - exp(-E/1.5)`

`adjusted agreement = 0.5 + (raw agreement - 0.5) * information`

The score before grade gates is:

`0.60 * strongest source quality + 0.25 * adjusted agreement + 0.15 * information`

An evidence-weighted outlier fraction applies up to a 30% conflict penalty.
Additional agreeing evidence therefore increases support and reduces
small-sample shrinkage without lowering the strongest-source term; highly
conflicted evidence can still lower the result.

A review-only record supported by one family is capped at `B+`; agreement
across multiple review families may reach `A-`, but never `A` or `A+` without
primary evidence. Fewer than 1.5 direct-evidence units cannot exceed `A-`. An
`A+` requires at least four direct-evidence units, or review evidence plus at
least 2.5 direct-evidence units, adjusted agreement of at least 0.90, and no
statistical outlier. These gates make high grades statements about
corroborated primary evidence rather than repeated database entries.

## Auditability

The runtime SQLite file contains:

- `henry_constants`: resolved values, numeric quality/agreement scores,
  selected source IDs, outlier IDs, independent-source counts, and the paired
  `B` fraction;
- `henry_source_values`: every eligible source value with its selection or
  exclusion status and effective weight; and
- `henry_temperature_source_correlations`: the 238 eligible pure-water
  three-parameter fits parsed from Sander's provenance notes;
- `henry_temperature_correlations`: one resolved active richer-temperature
  model for each covered CAS; and
- `henry_temperature_correlation_rejections`: richer curves rejected because
  an A-range reference anchor would require excessive value or slope changes;
  and
- `henry_metadata`: source version, DOI, hash, row counts, exclusion counts,
  and the exact aggregation policies.

## Runtime temperature and pressure quality

The stored grade describes evidence at the 298.15 K reference state. Runtime
reporting applies independent temperature and pressure penalties without
changing that stored grade.

With only `Hcp`, the full-quality point is 298.15 K and effective quality falls
by `0.01` per kelvin away from it. With both `Hcp` and `B`, the default
full-quality range is 278.15 to 323.15 K (5 to 50 C), with a loss of `0.005`
per kelvin beyond the nearest boundary. A resolved literature correlation's
Tmin/Tmax replaces that default; user-supplied `henry_Tmin` or `henry_Tmax`
has highest precedence for the corresponding boundary. Correlation quality
begins at the lower of the numeric `Hcp` and `B` qualities because both affect
the temperature-adjusted value. Fit RMS or uncertainty remains separately
reported rather than silently remapping the evidence grade.

Three richer temperature models are available:

- `iapws_g7_04`: official wide-range correlations for 14 common gases, using
  the IAPWS equation, gas-specific fitted ranges and RMS deviations. The raw
  guideline curve is normalized to the evidence-resolved `Hcp_298` and local
  `B`, preserving PFDSim's reference value and derivative while retaining the
  IAPWS curvature and near-critical behavior. Its private evaluator uses the
  exact Wagner-Pruss saturation-pressure coefficients specified by G7-04 so
  published check values remain reproducible; it does not replace PFDSim's
  general water `Psat` resolver.
- `brockbank_dippr101`: 313 multi-temperature experimental or
  experimental-plus-predicted regressions considered from Brockbank (2013)
  Appendix Table C.2, including per-compound ranges and uncertainty bands.
  Prediction-only and single-temperature rows are not considered
  automatically; 297 curves remain active after compatibility checks.
- `chapoy_dippr101`: the measured propane-water correlation from 277.62 to
  368.16 K, with reported 1.5% average absolute deviation.
- `exp_a_b_over_t_c_log_t`: 50 remaining resolved curves based on Sander's
  embedded `Hcp=exp(A+B/T+C ln(T))` fits. Curvature is robustly combined.

Reference normalization is allowed only when both resolved `Hcp_298` and `B`
have numeric quality at least 0.84 (`A-`). When either anchor is below A-range,
the original richer curve is retained without normalization and its own fit
quality is used for runtime reporting. When normalization is eligible, the raw
curve must first agree within 25% in `Hcp_298` and within
`max(500 K, 25%*abs(B))` in local slope. Otherwise it is rejected and PFDSim
falls through to another compatible richer curve or the van 't Hoff model.
The current build contains 362 active curves and 18 audited rejections.

Pressure has no quality penalty through 10 bar. Above 10 bar, quality falls
per additional bar by:

- `0.003` when no aqueous pressure correction is available;
- `0.001` when `Vinf` is estimated from critical volume; or
- `0.0002` when `Vinf` is measured, fitted, or explicitly provided.

The Krichevsky-Kasarnovsky correction for solubility-form `Hcp` is:

`Hcp(T,P) = Hcp(T,Pref) * exp(-Vinf * (P-Pref) / (R*T))`

where `Pref` is 1 bar. Measured `Vinf` values for 13 gases at 298.15 K are
stored separately in `henry_partial_molar_volumes`, from Zhou and Battino,
*J. Chem. Eng. Data* 46, 331-332 (2001),
<https://doi.org/10.1021/je000215o>. When measured or provided data are absent,
the documented fallback is `Vinf[cm3/mol] = 10.74 + 0.2683*Vc[cm3/mol]`. It is
used only when the resolved `Vc` source quality is at least 0.8 and is labeled
with its approximately 8-10% mean absolute error. Otherwise no correction is
applied.

Unit performance records include the solved T/P extrema, effective quality,
temperature and pressure penalties, `Vinf` source/method, and the minimum and
maximum applied `Hcp` pressure-correction factors.

For 246 resolved `B` correlations, at least one selected upstream row carries
Sander's `moreTdep` note. The upstream SQL contains 247 explicit
three-parameter notes in total; after excluding seawater and other unsuitable
rows, 238 are retained in the audit table. A `moreTdep` flag without usable
coefficients remains only an availability report—PFDSim does not invent them.

Regenerate the database with:

```bash
python scripts/build_henry_constants_db.py
```

The checked Brockbank Appendix C extract can be regenerated from the locally
held thesis PDF with:

```bash
python scripts/extract_brockbank_henry_table_c2.py
```

The extractor validates an exact 409-row table before replacing its CSV.

The builder validates the pinned archive and generated SQLite integrity before
atomically replacing `data/henry_constants.sqlite`.
