# Experimental interaction fitting

Open **Parameter fitting** in the web laboratory, or run `pfdsim fit --help`.
The Python API is `pfdsim.thermodynamics_models.interaction_fitting.fit_interactions`.
All three interfaces use the same regression and PFD export engine.

## Workflow

1. Choose an independent binary mixture, a saved laboratory, or import a PFD.
   Existing PFDs supply component definitions and the chosen scope's inherited
   vapor corrections. Fitting never solves that PFD's equipment.
2. Select NRTL/UNIQUAC, vapor treatment, temperature law and objective weights.
   NRTL alpha can be fixed or fitted. UNIQUAC R/Q can be resolved automatically,
   explicitly prefilled, or entered manually; all four manual values are needed.
3. Paste a webpage/book/paper table, JSON, Markdown, CSV or spreadsheet cells,
   or use the manual-entry textboxes. **Parse and add** appends observations.
   A popup resolves uncertain columns, units and common conditions without
   requiring a reformatted table. **Clear observations** clears the data explicitly.
   New generated IDs are short sequential numbers. Meaningful named IDs are
   retained where unique; prior UUID display IDs are shortened on draft restore.
   Set σ for incoming data or apply it to all observations/an imported set.
4. Fit and inspect physical errors and actual predicted/data curves. Scaled
   residuals and optimizer scores are under **Diagnostic data**. Designate
   individual points **validation-only** when they must not influence fitting.
   Optional cross-validation refits training folds separately.
5. Copy individual interaction entries, download a complete fitted-mixture PFD,
   or map the two components into an existing PFD and select its destination
   scope. Saved laboratories can be updated directly; imported files can be
   downloaded as updated PFDs. An open Workspace tab must reload the changed
   laboratory before editing it further. Unapplied source drafts must be resolved
   in Workspace before application.
6. To propose general inclusion, supply a source citation and submit the fit.
   Optional URL, DOI and processing notes help reviewers reproduce the result.
   Publication citation text is not sent as a model setting or inserted into
   fitted interaction parameters. It remains in session/submission provenance;
   CLI request source metadata is also retained separately from numerical parameters.

## Observation input

The data-type selector, input example, clear, parse/add and uncertainty controls
share one toolbar. **Set uncertainty** opens a popup; the **? data-help icon** is
at the top right of the observation panel. **Reset settings**, beside **Fit
parameters**, restores models, weights, scales, coefficients,
property policy, custom Psat/property controls, optimizer and CV settings. It keeps
the component identities, selected PFD, observations (including per-point flags),
source provenance and completed results so reset cannot reinterpret data as a
different chemical pair or destroy work.
Supporting-property guidance is behind its own **?** popup. Estimate and HOC η
policies appear only when applicable to the active data and correction model.
IDEAL hides correction constants; needed fields follow the model requirements.
Unused fields remain under **Additional property overrides** and keep their
entered values, including Tc/Pc anchors if a custom canonical Psat needs them.

### Webpage, paper and unlabeled pastes

The importer extracts rectangular numeric blocks surrounded by headings and
captions. It accepts tabs, commas, semicolons, whitespace, multiline group
headings, HTML tables with rowspan/colspan, and flattened PDF numbers read
across rows or down columns. Multiple detected tables can be selected in the
popup. It understands decimal commas (including whitespace-separated numbers), negative
Unicode signs, scientific notation, and common footnote/uncertainty annotations.
The original text, raw cells, selected interpretation, excluded rows and assigned
observation IDs are retained with the fit's provenance.

An unlabeled Txy table can be suggested from a temperature-like column and two
co-varying compositions. Temperature units and mole-percent/fraction basis are
suggestions when unstated; liquid/vapor order uses endpoint boiling trends when
available and is provisional otherwise. Confirm or change them in the popup.
Mass versus mole basis cannot be proven from numeric ranges. Mass compositions
can be converted when both molecular weights are supplied.

Paper notation includes compact headers such as `P [kPa] T [K] x₁ y₁`,
`Hₘᴱ / J mol⁻¹`, and primed LLE endpoints. Captions outside an HTML table
can supply common conditions. Uncertainty and calculated columns are ignored
instead of being treated as measured values. Confidence percentages do not
set the composition scale. Energy units include J/mol, kJ/mol, cal/mol and
kcal/mol, with thermochemical calories converted using 4.184 J/cal.

For OCR-damaged VLE headings, recognizable columns constrain the numerical
inference. Relative magnitudes, plausible temperature ranges and composition
trends help rank the remaining columns. The largest column is supporting
evidence for temperature, not a rule: pressure may be larger. A remaining
positive column can be proposed as pressure, but missing pressure units require
confirmation. Inferred roles remain visible in the interpretation popup.

Labeled columns retain their own composition scales and component indices,
including tables that mix `x₁` with `y₂`. Explicit import controls override those
labels. Repeated series can use different energy units without a common override.
`ln γ∞` and `log10 γ∞` are converted to dimensionless activity coefficients;
an unspecified `log` base blocks import until the heading is clarified.

Shared HE headings followed by several temperature conditions are recognized
even when OCR has changed `x1` to `xl`/`xI` or split the quantity/unit notation.
The preview retains the complete table and prefills those temperatures when
repeated-series mode is selected. Short rows without empty column placeholders
keep their values as unassigned measurements; restore their column positions or
exclude them explicitly before importing. Explicit empty cells retain their
known positions. Separated minus signs and clearly split decimal points are
normalized, while numeric signs already present in the paste are preserved.

For the acetone/water Wikipedia-style paste with `Temp.`, `°C`, `% by mole
acetone`, and separate `liquid`/`vapor` headings, 18 source rows are recognized.
The two pure endpoints are retained in the preview and excluded from binary
regression, leaving 16 observations. Pressure is absent from that paste and is
requested explicitly. At 1 atm, the row `87.8  1.0  33.5` becomes
`T_K=360.95`, `P_bar=1.01325`, `x1=0.01`, `y1=0.335`, when acetone is component 1.
If acetone is component 2, both fractions are complemented. A named source
component that does not match the mixture requires an explicit choice.

Cancelling import preserves the existing observations and pending paste.
Appending never replaces earlier rows, and adding another manual point does
not reparse a previously imported table. The paste box is cleared only after
a successful append. Fit results already calculated remain snapshots of their
original observations; changing the editor does not change a saved result.

The CLI uses the same interpreter:

```sh
pfdsim fit pasted-table.txt --components acetone water --preview-import
pfdsim fit pasted-table.txt --components acetone water --pressure 1 \
  --pressure-unit atm --temperature-unit C --composition-basis mole_percent \
  --composition-component 1 -o report.json
```

For ambiguous columns, use `--import-options` or `import_options` in a JSON
request, for example `{"mapping":["temperature","x1","y1"],"pressure":1,
"pressure_unit":"atm","temperature_unit":"C","composition_basis":"mole_percent"}`.
Additional controls include `table`, `kind`, `temperature`, `enthalpy_unit`,
`composition_component`, `molecular_weights`, `exclude_rows` (zero-based),
`column_count` and `layout` (`rows`/`columns`). `--preview-import` performs no fit
and exits with code 2 when information is missing. Clear numeric-table suggestions
are used in noninteractive calls when sufficient conditions are supplied; the
import report records their inferred status.

### Explicit repeated-series tables

The interpretation popup has a **Repeated series / multicolumn** button. Repeated
roles do not automatically expand data. After choosing this mode, select a layout:
shared x plus HE columns, shared x plus repeated T/y or P/y pairs, shared x plus
temperature/pressure columns, repeated T/x/y triples, or custom column groups.
Every source column has a meaning and a scope: shared across series, belonging to
one series, or ignored. Each series has its own temperature/pressure and units,
name and validation-only flag. A missing condition blocks addition of the whole
selected import rather than silently importing just the already complete series.

The 19-row acetone/1-propanol HE paste contains five measured HE columns but no
temperatures. Assign column 1 as shared x and the other five as HE, then enter
the five actual temperatures and HE units. This produces **95 observations**.
No temperatures are invented from the curvature or magnitudes of the HE values.
Conditions printed in headings are reviewable hints, used only after selecting
the repeated layout. Source strings, raw rows, per-series exclusions, projected
column indices and source-row lineage remain in the import report.

Each expanded series becomes its own observation set and validation group unless
the source supplied a group. Its uncertainty can be set through the existing set
control. Missing HE values omit only that series' observation at the source row;
other series remain available. Missing optional y does not discard a VLE point.
Pure endpoints are handled per series by the same single-series converter.

The CLI/API can request the same explicit projection with `import_options`:

```json
{
  "kind": "HE",
  "mapping": ["x1", "enthalpy", "enthalpy"],
  "shared_columns": [0],
  "temperature_unit": "K",
  "enthalpy_unit": "J/mol",
  "composition_basis": "mole_fraction",
  "series": [
    {"name": "First condition", "columns": [1], "temperature": 298.15},
    {"name": "Second condition", "columns": [2], "temperature": 308.15}
  ]
}
```

Those temperatures are illustrative explicit inputs, not the unstated
temperatures of the supplied table. Column indices are zero-based. Conditions
and units can differ per series; API series also accept `weight`, `sigma`, and
`validation_only`. All series use the authoritative observation converter.

### Manual entry and uncertainty sets

Choose **Enter an observation**, choose its kind, and fill its textboxes. Units
and composition basis have explicit controls. Use the same **Parse and add**
button to append it. There is no separate autogenerated sample-row action.

Under **Uncertainty / σ for an observation set**, fill the relevant residual
scales. They apply to new observations, including every row of a newly imported
table. **Apply σ to this set** updates either all current observations or a
selected imported/manual set. Blank scales preserve existing row settings.
Overriding one scale on a row that had a scalar σ preserves that scalar's values
for its other objectives. Individual row σ remains editable for exceptions.

The component order is fixed for the whole fit. Every composition field refers
to the mole fraction of **component 1**, including both LLE endpoints.

| Kind | Required measurements in addition to temperature | Optional measurements |
| --- | --- | --- |
| `VLE` | `x1`, pressure | `y1` |
| `LLE` | `x1_alpha`, `x1_beta` | pressure (not used by the activity-coefficient LLE model) |
| `HE` | `x1`, excess enthalpy | |
| `GAMMA_INF` | `gamma1_inf` and/or `gamma2_inf` | |
| `AZEOTROPE` | `x1`, pressure | `y1`, if equal to `x1` |
| `VLLE` / `HETEROAZEOTROPE` | temperature and pressure | `y1`; both `x1_alpha` and `x1_beta`, or neither |
| `UCST`, `LCST` | temperature | `x1`; otherwise critical composition is fitted |

Specify temperature as `T_K` or `T_C`, pressure as `P_bar`, `P_kPa` or `P_atm`,
and excess enthalpy as `HE_J_mol` or `HE_kJ_mol`. Do not supply competing units
for the same measurement. Unknown fields and duplicate observation IDs are
errors. VLE compositions and LLE endpoints must be strictly inside `(0,1)`;
pure-component points contain no binary interaction information.

This illustrative table is an input example, not experimental reference data:

```markdown
| kind | T_K | P_bar | x1 | y1 | HE_J_mol | gamma1_inf | weight | pin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VLE | 350 | 1 | 0.3 | 0.6 | | | 1 | false |
| HE | 300 | | 0.5 | | 500 | | 2 | false |
| GAMMA_INF | 300 | | | | | 3 | 1 | false |
```

JSON may be an array of observations, `{ "rows": [...] }`, or
`{ "datasets": [{ "kind": "VLE", "source": "...", "weight": 2,
"rows": [...] }] }`. Dataset metadata supplies defaults; individual rows can
override them. JSON/Markdown code fences can be pasted directly.

Rows also accept `id`, `source`, `group`, `weight`, `pin`, `pin_tolerance`, `validation_only` and
`sigma`. A source is provenance; a group keeps related observations together
when using leave-group-out validation. Without a group, source is used, then ID.
`validation_only: true` excludes a point from optimization, pins, training folds
and calibration temperature bounds. Such points retain their predictions, plots,
physical MAE/RMSE and diagnostics in a separate validation report. They cannot be
hard-pinned, and at least one training point is required. A validation failure or
outlier does not change fitted coefficients. Training and validation markers are
distinguished in the plots; validation-only markers are hollow.

## Objectives, weights and pins

For unpinned observations, the objective is

```text
sum_rows objective_weight[kind] * row_weight * sum_j (residual_j / sigma_j)^2
```

These are sums, not averages over each objective. A larger dataset therefore
has more influence at equal weights. A zero weight excludes unpinned rows;
hard pins remain active independently of weights.

The residual definitions use runtime activity coefficients, excess enthalpy,
pure-liquid references and vapor fugacity coefficients:

- **VLE with vapor compositions:** component log-fugacity differences at the
  supplied T/P/x/y. Without y, the vapor composition is solved at T/P/x so both
  component differences agree; the common difference is the bubble closure.
- **HE:** calculated minus measured J/mol. Runtime kJ/kmol is numerically the
  same unit, so no factor of 1000 is applied to runtime values.
- **Gamma infinity:** log(calculated/measured), evaluated at exactly zero mole
  fraction of the dilute component.
- **LLE:** both component log-activity differences at the observed endpoints,
  plus supporting-tangent penalties that discourage unstable stationary pairs.
  The final report independently runs the runtime equilibrium solver and reports
  its stable binodal predictions and phase fractions.
- **Azeotropes:** the VLE fugacity equalities with y=x, with their own objective
  weight and optional per-point weighting/pinning.
- **VLLE/heteroazeotropes:** two liquid fugacity states are compared to one shared
  vapor state, with supporting-tangent stability checks and a separate objective
  weight. With no liquid endpoint measurements, bounded latent phase coordinates
  are fitted alongside interaction parameters; their interval is constrained to
  stay nonzero. Final predictions use the runtime binary three-phase invariant.
  Validation-only points with missing endpoints infer phases from the fixed model,
  never through new fitted nuisance variables.
- **UCST/LCST:** zero second and third composition derivatives of dimensionless
  total liquid g/RT, positive fourth derivative, and the appropriate sign of the
  curvature's temperature derivative. Derivatives use centered finite differences
  of the runtime chemical potentials. Final supporting-tangent checks reject
  critical stationary points on globally unstable branches.

The default scales are:

```json
{
  "log_fugacity": 0.01,
  "log_gamma": 0.01,
  "HE_J_mol": 100,
  "curvature": 0.1,
  "third_derivative": 0.1
}
```

They are normalization choices, not assumed experimental uncertainties. Set
them to appropriate values for the data. `sigma` on a row may be a number or an
object overriding selected keys. The LLE tangent grid contributes a
grid-normalized penalty. Critical fourth-derivative and temperature-direction
penalties have their own dimensionless feasibility scaling.

`pin: true` removes that row from the soft objective and imposes two-sided
constraints on its scaled residuals. The default `pin_tolerance` is `1e-5`;
this is a numerical tolerance, not an infinite weight. With the default HE
scale, for example, that means an absolute enthalpy tolerance of 0.001 J/mol.
Infeasible pins fail explicitly. Numerical feasibility allows `1e-7` additional
scaled residual tolerance. Pins include the row's phase-stability penalties.

## Model controls

### Physical assessment and curves

Main assessment uses unweighted physical errors: MAE, RMSE, bias, maximum error,
number evaluated and unavailable predictions. VLE/azeotrope/VLLE report temperature
differences in K (numerically the same as °C differences), pressure in bar and
vapor mole fractions. Isobaric datasets compare vapor compositions at predicted
bubble temperature; isothermal datasets use predicted bubble pressure. Both T/P
predictions are available when their qualified property ranges permit them.
LLE reports both liquid endpoints as component-1 and complementary component-2
mole fractions. HE errors are J/mol. Gamma-infinity errors include dimensionless
and relative-percent scores. Critical points report model critical temperature
and composition when an independent fixed-model root can be localized.

Plots sample 21 model states per condition, rather than connecting only the
calibration points. They include Txy/Pxy curves for repeated pressure/temperature
conditions, LLE liquid branches, VLLE liquid/vapor/pressure behavior, HE versus
composition, gamma-infinity versus temperature and parity comparisons. No valid
state is invented across a failed phase/property solve: plots show gaps and the
underlying reason. Hover/focus/click markers for numerical values. Training,
validation-only and cross-validation physical scores remain separate from
scaled residuals used by the optimization objective.

### Supporting-property and Psat policy

`estimate_properties` defaults to true: criticals, acentric factor, dipole and
other supporting corrections may be estimated with recorded source/quality and
warnings. Dipole quality below 0.9 does **not** prevent fitting. Users can enter
values instead. `allow_hoc_eta_default` defaults to true; missing HOC pure eta
uses its group correlation or a warned zero default. Enter `hoc_eta` to override
it, or disable the zero-default policy. When the optional quantum geometry backend
is unavailable, HOC's modified radius can use a deterministic ETKDG/MMFF/UFF
conformer estimate at lower stated quality, with a warning. It uses the same
principal-moment formula and shared starting-geometry implementation.

| Vapor treatment | Supporting inputs |
| --- | --- |
| Ideal | No vapor correction constants; supplied Psat is evaluated directly, otherwise normal canonical Psat uses Tc/Pc anchors |
| RK | Tc, Pc |
| PR | Tc, Pc, omega |
| VDM | Library or component association definitions; generic associator estimates are warned |
| Tsonopoulos | Tc/Pc/Vc/omega; dipole for the relevant polar classes |
| Pitzer–Curl, Abbott | Tc/Pc/Vc/omega; no dipole or association eta |
| HOC | Tc/Pc, dipole, modified radius, pure eta; cross eta is a separate vapor parameter |

Psat is the exception: vapor-equilibrium objectives (VLE, homogeneous azeotrope
and VLLE) require non-estimated Psat quality **at least 0.9** over their target
temperature span. The audit checks endpoints plus all source-region boundaries
and interval midpoints, so weak interior segments cannot pass because measured
nodes happen to land in strong regions. Failure identifies the component,
temperature, quality and source and asks for a supplied Psat correlation. Model
temperature predictions outside a qualified region remain explicitly unavailable.
Liquid-only LLE/HE/gamma-infinity/UCST/LCST fits never require Psat or initialize
unused vapor corrections. If vapor data are validation-only, their predictions
still need a qualified vapor/property basis; they do not train its parameters.

### Custom Psat and obscure components

**Custom saturation pressure / obscure components** supplies a Psat definition
for either component, with explicit coefficient fields and validity bounds:

- **Antoine:** `log10(P/unit)=A−B/(T/unit+C)`, Celsius or Kelvin denominator.
  Absolute pressure units are converted into PFDSim's bar/Celsius PFD fields.
- **DIPPR 101:** `ln(P/unit)=A+B/T+C ln(T)+D T^E`, T in kelvin. The pressure
  conversion adjusts A. Its analytic value/derivative are shared with other
  DIPPR 101 property users and its PFD equation is `dippr_eq101`.
- **PFDSim canonical A–F/A–G/A–H:** existing canonical Kelvin, ln(P/bar)
  coefficients; A–H additionally requires inverse power −3, −5 or −7.

JSON requests accept `psat: [definition_or_null, definition_or_null]`, with
`form`, `coefficients`, `Tmin_K`, `Tmax_K`, `pressure_unit`, `temperature_unit`,
optional `inverse_power`, and `source`. Forms are `antoine`, `dippr101`,
`canonical_psat_af`, `canonical_psat_ag`, and `canonical_psat_ah`.
`component_properties` accepts two objects using `MW`, `Tc_K`, `Pc_bar`,
`Tb_K`, `omega`, `Vc_cm3_mol`, `Zc`, `dipole_D` (Debye), `hoc_eta`, `Rprime_A`
(angstrom) and `smiles`. Unknown components need MW. Supplied Antoine, DIPPR,
canonical A–F and A–G are directly evaluated without critical anchors; A–H needs
its actual Tc for the inverse-power term. Normal unsupplied Psat and vapor
corrections can still need criticals, resolved/estimated with warnings when
needed. Unresolvable supporting constants require an identifiable structure or
entered values; they are never fabricated silently. Resolved supporting values
are preserved in fit provenance and, where needed, in PFD component definitions.
Property-only initialization retains explicit overrides for the intended vapor
model even before that provider is constructed. This prevents supplied HOC η or
VDM component definitions being filtered as unused by the initial liquid package.
HOC classification resolves the molecular structure through the chemical identity
before assigning its group. A simple organic acid uses the recorded **η=4.5**
default; a supplied pure η overrides it. Unsupported multifunctional acids need
an explicit η and cannot fall through to the missing-value zero default. The
4.5 table default is distinct from a low-quality property estimate.

## Saving fitting sessions

**Saved fits**, beside the fit/reset controls, saves a named session with all
control values, data points, row weights/pins/validation flags, set-level σ,
custom Psat and pure-fluid properties, vapor parameters, CV/optimizer settings,
component/PFD scope definitions, import provenance and completed results. Pending
pastes and partially completed repeated-series interpretations are saved too.
Opening a session restores its state without rerunning a calculation and keeps
the prior draft as **Previous unsaved draft** for recovery.

Guests can save multiple named sessions in their browser. Signed-in accounts save
private versioned copies in the existing web store's `fit_sessions` table. Account
copies can be opened on another device. **Save new copy** avoids overwriting an
existing session; stale saves are rejected instead of replacing newer work.
Temporary account-network failures retain a browser copy marked pending upload.
**Download session JSON** and **Import session JSON** provide portable complete
snapshots with `type: pfdsim_fit_session` and `schema_version: 1`. Fitting-session
storage is separate from sourced submissions and the administrator publication
database; saving a session does not submit or publish it.

Within the fitter, supplied Antoine, DIPPR and canonical correlations are evaluated
directly with their original coefficients and declared validity range. They are
not canonically refitted, anchored to a critical/boiling point, clamped or
extrapolated. An out-of-range evaluation raises an error. This is confined to the
fitting context; shared Psat processing and PFD export retain their existing
behavior. Components without a supplied correlation still use PFDSim's normal
qualified Psat. Custom coefficients are fixed during interaction fitting; they
are not simultaneously fitted. Direct evaluation is reapplied if fitting a vapor
parameter rebuilds the fitter's thermodynamic package.

The full PFD export preserves these definitions. Applying a fit copies explicitly
entered Psat and pure-fluid properties into the destination component globally;
other scopes using that component therefore see those values as well.
For imported PFD-only property definitions, incompatible destinations must still
be reconciled before VLE parameter application. R/Q prefill can run while custom
Psat coefficient fields are still being filled.

Each direction has independent coefficients. The selected form describes
NRTL **tau** or UNIQUAC **log(tau)**:

| Form | Law |
| --- | --- |
| `constant` | A |
| `inverse` | B/T |
| `constant_inverse` | A+B/T |
| `constant_inverse_anchored` | A+B/T+C h(T) |
| `constant_inverse_linear` | A+B/T+D T |
| `full` | A+B/T+C h(T)+D T+E T² |

Here `h(T)=(Tref-T)/T+ln(T/Tref)`, with `T_ref_K=298.15` by default. A constant
law predicts zero excess enthalpy and cannot fit nonzero HE. Multiple terms
need enough independent temperature/caloric information; rank diagnostics
identify locally underdetermined fits.

Parameter names for `initial` and `bounds` are `12.constant`, `12.inverse`,
`12.anchored`, `12.linear`, `12.quadratic` and the corresponding `21.*` names,
restricted to the selected form. Their values are physical coefficients, not
internally scaled optimization coordinates. Fitted alpha uses `alpha12` and is
bounded within `[0.01,1]`. Unknown critical compositions use
`critical_x1.<observation-id>`. By default, coefficient bounds correspond to
scaled coordinates `[-30,30]`; inverse, linear and quadratic coefficients are
scaled with Tref to keep optimization well conditioned. Default starts are
deterministically seeded (`seed=1729`, `starts=3`), with `max_nfev=500` per start
and constrained refinement where pins are present.

Vapor treatments are `IDEAL`, `RK`, `PR`, `VDM`, `TSONOPOULOS`, `PITZER-CURL`,
`ABBOTT` and `HOC`. They retain their runtime physical-property requirements.
Optional joint vapor-parameter fitting supports PR kij, Tsonopoulos kij, HOC
eta and VDM cross-association residual H/S. Example:

```json
{
  "vapor": "PR",
  "vapor_parameters": [
    {"model": "PR", "field": "kij", "value": 0,
     "fit": true, "lower": -0.5, "upper": 0.5, "scale": 1}
  ]
}
```

For VDM use `delta_H_residual_J_per_mol` and
`delta_S_residual_J_per_mol_K`; HOC uses `eta`. A `fit: false` parameter is a
fixed override. Component-level vapor/association definitions can be supplied
through an imported PFD. RK uses its runtime zero-kij treatment; it has no
adjustable cross parameter here. A failure of the selected vapor provider is
reported rather than silently fitted as an ideal vapor.
Regression also rejects the runtime's extreme activity/exponent clipping
limits: clipping can break Gibbs–Duhem or enthalpy derivative consistency, so
matching data in that clipped regime would not establish a physical fit.

## Cross-validation

`cv.method` supports `none`, seeded `kfold`, `leave_temperature_out`, and
`leave_group_out`. K-fold has `cv.folds` (default 5, at most 20). Grouped methods
require 2–20 distinct groups. Each fold starts independently and uses only its
training observations. Pins are retained in every training fold and are never
held out. Designated validation-only points are not members of any fold's training
or held-out sets. Missing critical compositions in held-out rows are located from the
training-fitted thermodynamic surface, without fitting to the held-out target.

Reports include the training/held-out IDs, rank and parameter count, validation
residuals and failures. A fold with insufficient information can be rank
deficient even if optimization converges; check both. Objective diagnostics
describe goodness of fit; an optimizer success flag does not prove that every
measurement was reproduced at its experimental precision.

## CLI and Python examples

```sh
pfdsim fit request.json -o report.json --entry fitted-entry.pfd --pfd-output mixture.pfd
pfdsim fit measurements.md --config settings.json -o report.json
pfdsim fit request.json --apply-to process.pfd --pfd-output process-fitted.pfd \
  --component-map '{"Fit_1":"EtOH","Fit_2":"Water"}'
pfdsim fit request.json -o report.json --source 'Authors, journal/year, Table 2' \
  --submission-bundle submission.json
pfdsim fit request.json -o report.json --source 'Authors, journal/year, Table 2' \
  --submit-url http://localhost:5000
pfdsim fit submit report.json --source 'Authors, journal/year, Table 2' \
  --server http://localhost:5000
```

The CLI also accepts observations on stdin (`pfdsim fit -`), and flags for
components, model, temperature form, vapor treatment, alpha, weights, R/Q and
validation. Input and output paths must differ. Existing outputs require
`--force`, which preserves a numbered `.bak` copy. A fit needing review exits
with code 1 while still providing its diagnostics.

A request JSON contains the settings and observations together:

```python
from pfdsim.thermodynamics_models.interaction_fitting import fit_interactions, export_fit

result = fit_interactions({
    "components": ["ethanol", "water"],
    "model": "NRTL",
    "vapor": "IDEAL",
    "form": "constant_inverse",
    "fit_alpha": True,
    "weights": {"VLE": 1, "HE": 2, "GAMMA_INF": 1},
    "observations": observations,
    "cv": {"method": "kfold", "folds": 5},
})
entry = result["entry"]
merged = export_fit(result, pfd_text=existing_text, scope="global",
                    component_map={"Fit_1": "EtOH", "Fit_2": "Water"})
```

Results include the normalized request, component definitions, physical
coefficients, fitted vapor corrections, R/Q, per-point predictions and
residuals, optimizer diagnostics, warnings and validation folds.
Saved reports remain reusable for export and submission after the original
web calculation job has expired. `pfdsim fit submit` never reruns regression.
Individual entries need the selected method and any manual component R/Q/VDM definitions
in the receiving PFD; complete exports carry those definitions automatically.
Export preserves unrelated equipment, streams, reactions, scopes and pairs,
replacing only matching model/pair entries in the selected scope. Component
definitions are global in PFD, so changed R/Q or VDM parameters also affect
other scopes using those components. Global BV correlations are preserved;
named scopes support default Tsonopoulos and the existing HOC alias, while other BV
correlations require global scope.

For VLE/azeotrope fits, export checks that the destination's explicit pure-fluid
definitions and minimum-Psat policy agree with the fitting basis. If they differ,
fit using the destination PFD or reconcile its definitions before application;
silently transferring coefficients between different property bases can change
the predicted equilibrium. The complete fitted-mixture PFD carries its source
definitions.

## General-inclusion submissions

Web submissions use an owned completed job. CLI submissions upload a complete
report through the server's session/CSRF-protected endpoint. External reports
are explicitly marked as awaiting review; their numerical claims are not
trusted as independently recomputed results.
Multiline citations are preserved exactly in the submission metadata. PFD metadata
values are serialized as one physical line with line breaks flattened to spaces
and quotes/backslashes escaped, so a multiline comment cannot create a spurious
top-level directive. Citation edits never change fitting coefficients or residuals.

Submissions persist in `data/source/activity_fitting/user_activity_fits.sqlite`
(override with `PFDSIM_ACTIVITY_FITS_PATH`) in the `fits` and `events` tables,
with ID, owner, timestamps, source, complete result, CAS identities, review
version, reviewer/notes, publication status and an append-only event history. They
survive calculation-job expiration. Repeat submission of the same job by the
same owner returns its existing ID. `/api/fitting/submissions` lists only the
current owner's submissions. Maintainers can retrieve review artifacts with
`pfdsim fit submissions --help`.

Signing in transfers guest submissions to the account. If the account already
submitted the same external report, both submission IDs, sources and histories
are retained; the transferred job ID is disambiguated and its original identity
is recorded in an ownership event.

Submitting alone never writes runtime JSONs. The authenticated administrator can
review, approve, reject, publish, replace or withdraw fits. Root can publish its
own completed sourced fit directly. The shared builder consumes published fits
from the SQLite file, replaces this model/CAS pair's former entries and includes
the submission ID and source/reviewer metadata in the generated record.
Only one user fit is active per model/pair; superseded coefficients and provenance
remain recoverable in the store.

### Root account setup

The name `root` is reserved for the first administrator and requires a setup
token. The server creates `root-setup-token` with mode 0600 in its web-data
directory (by default `~/.local/share/pfdsim/web/`), or can use
`PFDSIM_ROOT_SETUP_TOKEN`. Enter it in the root registration field that appears
when registering that username. Configured tokens must have at least 32 characters;
the generated token meets this requirement. The root role is committed atomically with the
account. The token cannot create another root account after the name is claimed.
It is not returned by any HTTP endpoint. A supplied `is_admin` registration field
has no effect. Login/session permissions come from the server's account-role table.

### Administrator publication

The root account sees an **Administrator fit review** panel. It can download the
full provenance bundle, record review notes, approve/reject using a versioned
decision, and publish an approved fit. **Publish this fit to runtime** submits and
approves root's own current fit, then runs the same publication path.
Review notes are optional; a blank note records the administrator's decision in
the existing history. The dialog remains open after approval, displaying the
saved status and **Publish to runtime**. Approval alone does not publish or
remove the data. The queue includes compound names, model and objective types
for pending, approved, published and rejected fits. **View fit assessment**,
**Download fit result**, **Download fitted PFD** and **Download full provenance**
make the stored data directly accessible. Failed review requests are shown next
to the dialog actions; shared notifications also appear above modal backdrops.
Publication is a queued, cancellable job using existing compute accounting.

`scripts/build_cas_interaction_parameters.py` owns both the normal SQLite overlay
and publication. It rebuilds the affected activity table from source, keeps a
snapshot under `runtime_snapshots/` beside the store, and atomically replaces the
table. History records the new artifact's SHA-256 and relative backup path.
Full builds and administrator publications share a lock in the runtime table
directory, including when different provenance stores are selected, and use atomic table
replacement. Normal builds can select another provenance source explicitly with
`--user-fits-path`. Rebuilds reject a provenance store missing fit IDs already in
the runtime table, preventing one deployment's store from silently removing
another's published fits. Copies or relocations retaining the review history
remain valid. Separate stores should use separate runtime directories.
An interrupted republication retains the previously published
fit until recovery or a successful replacement.
New/reinitialized simulation packages detect changed table generations; already
constructed standalone thermodynamic objects remain snapshots. Withdraw rebuilds
the table from maintained sources and remaining published fits. Interrupted
publication can be explicitly recovered by an admin once its prior job has ended.
`PFDSIM_INTERACTION_DATA_DIR` selects an alternate runtime table directory for
an isolated deployment; the publisher and runtime lookup honor it together.

Publication verifies CAS identities, coefficient/export agreement, phase/pin
checks and compatibility with the shared property basis. A fit that requires
different global UNIQUAC R/Q, Psat or vapor corrections remains stored and can
be used through its complete PFD. Its pure-fluid/structural/vapor basis must be
curated before liquid coefficients can safely enter the common runtime tables;
publishing does not silently discard those dependencies. General publication
requires the repository's maintained builder and source collections.

## Validation

Numerical/HTTP/import/publication tests include `tests/test_interaction_fitting.py`,
`tests/test_web_fitting.py`, `tests/test_fitting_import.py`, `tests/test_fitting_psat.py`
and `tests/test_activity_fit_admin.py`. The opt-in real-browser workflow is
`python tests/browser_interaction_fitting.py`; its screenshots and downloaded
PFDs are written to a temporary report directory.
Wide-table projection and HOC/saved-state regressions are covered by
`tests/test_repeated_series_import.py` and `tests/test_fit_sessions_and_hoc_overrides.py`.
`python tests/browser_repeated_fit_sessions.py` checks explicit wide-table import,
95-point expansion, local/account save and reopening, JSON portability, and
resumption of a partly interpreted table without rerunning a fit.
