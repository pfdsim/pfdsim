# Gas thermal-conductivity fallback investigation

Date: 2026-09-13

## Executive summary

pfdsim resolves pure liquid and dilute-vapor thermal conductivity from explicit
`kl`/`kg` correlations and Perry data, followed by the predictive dilute-vapor
fallback described here when those sources are unavailable. This investigation
compared two viscosity/heat-capacity relations
against Perry 9th-edition vapor-conductivity curves, measured their systematic
chemical-class residuals, and tested regularized functional-group corrections.

The recommended experimental fallback is the geometry-dependent relation plus
a sparse, multiplicative functional-group correction affine in inverse absolute
temperature:

\[
k = k_0 \exp\!\left[\sum_g I_g\left(A_g + \frac{B_g}{T}\right)\right].
\]

Here `T` is in K and `I_g` is one when group `g` is present and zero otherwise.
The correction uses 14 active binary features. Seven investigated features are
fixed exactly to zero because they offered less than 15% relative held-out
class improvement. All fitted corrections are gated to pfdsim's strict
molecular-organic domain; inorganics use the base relation unchanged.

With oxalic acid removed before fitting and scoring, compound-held-out
cross-validation gave:

| Metric | Uncorrected | Recommended correction |
|---|---:|---:|
| Point-weighted MAPE | 12.46% | **7.46%** |
| Compound-equal mean curve MAPE | 11.79% | **7.23%** |
| Median absolute error | 9.39% | **4.47%** |
| P90 absolute error | 26.33% | **17.07%** |
| P95 absolute error | 34.05% | **25.13%** |
| Mean signed error | -9.66% | **-0.72%** |
| Maximum absolute error | 108.06% | 108.06% |

The correction removes 5.00 MAPE points, or 40.1% of the baseline error. These
are internal Perry-held-out results, not independent external validation.

## Scripts

- `benchmark_gas_viscosity_cv_relation.py` benchmarks the two uncorrected gas
  relations against Perry.
- `experiment_functional_group_corrections.py` performs compound-held-out
  functional-group correction experiments. Its default exclusion list contains
  oxalic acid, CAS 144-62-7.

Both scripts are offline. Machine-readable output can be requested with
`--output-json`.

## Reproducing the investigation

From the repository root, run:

```bash
bash scripts/thermal_conductivity/reproduce.sh
```

The reproduction runner uses Python 3.14.4 and `uv run --frozen`, so the
environment comes from the committed `uv.lock` without updating dependency
resolution. It regenerates:

- `results/geometry_dependent_benchmark.json` and `.txt`
- `results/acentric_factor_benchmark.json` and `.txt`
- `results/acentric_correction_comparison.json` and `.txt`
- `results/geometry_correction_comparison.json` and `.txt`

The geometry correction artifact contains all temperature-basis, acid-detail,
pruning, and hydrocarbon-split candidates used to choose the final model. Each
JSON artifact records its exact CLI arguments, Python and package versions,
SHA-256 hashes of both Perry inputs, the canonical ideal-gas heat-capacity
database, `uv.lock`, and the executing scripts. Platform and machine metadata
are recorded as well. The correction artifacts also
contain the selected model's final full-data ridge coefficient fit and label
its resubstitution metrics as non-validation results.

The JSON files are formatted deterministically with sorted keys. Re-running the
runner in an unchanged checkout and locked environment should reproduce the
same reported metrics. Floating-point least-squares results can differ in their
last insignificant digits across operating systems or CPU/BLAS implementations;
the recorded platform metadata makes that boundary explicit. The recorded CLI
output path in `argv` and the script hash metadata are expected provenance
fields; version control will expose any intentional input, script, or result
change.

## Resolver behavior

The production resolver in `property_resolution/thermal_conductivity.py` uses
this order:

1. An explicit supplied/PFD `kg` or `kl` correlation.
2. An in-range Perry correlation.
3. For liquids, an in-range Perry saturated-liquid tabulation.
4. For dilute vapor only, the PFDSim-modified Stiel-Thodos fallback when all
   required inputs resolve.
5. Failure with `PropertyResolutionError`.

The fallback resolves ideal-gas Cp and uses `Cv = Cp - R`, dilute-vapor
viscosity, molecular weight, critical temperature, formula, and SMILES-derived
geometry. Monatomic and diatomic geometry needs only a formula. The result's
notes preserve input provenance and active correction groups.

## Reference data and benchmark construction

The gas benchmark used:

- Perry Table 2-145 vapor thermal conductivity as the reference target.
- Perry Table 2-138 dilute-vapor viscosity as an estimator input.
- Perry Table 2-106 molecular weight, critical temperature, and acentric factor.
- pfdsim's canonical local ideal-gas heat-capacity kernels. Every selected
  kernel in this population was sourced from Perry.
- \(C_v=C_p-R\), because pfdsim has no separate ideal-gas `Cv` resolver.

Each Perry conductivity curve was sampled uniformly over the intersection of
the declared conductivity, viscosity, and heat-capacity ranges. The primary
correction experiment used 51 points per curve.

Using empirical Perry viscosity intentionally isolates the conductivity
relation. The results do not include error from pfdsim's predictive viscosity
fallbacks. End-to-end accuracy will therefore be worse whenever viscosity is
not supplied by a high-quality provider.

### Molecular geometry

Monatomic and diatomic species were identified from molecular formula atom
counts. Larger molecules were classified from local SMILES and hydrogen-complete
RDKit topology. An unbranched chain with sp-hybridized internal atoms was
classified as linear; all other molecules were classified as nonlinear. No
quantum geometry calculation was required.

Air cannot be assigned a molecular geometry and was excluded from the
geometry-dependent benchmark. Oxalic acid was explicitly excluded from the
final correction experiments because it is a pathological outlier for the
group-contribution investigation.

## Uncorrected relations

### Geometry-dependent relation

The first relation was

\[
k_0 = \frac{\mu C_v}{M}F,
\]

where `M` is in kg/mol, `mu` is in Pa s, and

\[
F =
\begin{cases}
2.5, & \text{monatomic},\\
1.3 + \dfrac{R}{C_v}\left(1.7614-\dfrac{0.3523}{T_r}\right),
    & \text{linear},\\
1.15 + 2.033\dfrac{R}{C_v}, & \text{otherwise}.
\end{cases}
\]

With oxalic acid still included, it achieved 12.68% MAPE, -9.36% bias, and
34.68% P95 absolute error over 344 compounds and 354 Perry curves. By geometry:

| Geometry | Compounds | MAPE |
|---|---:|---:|
| Monatomic | 3 | 0.66% |
| Linear | 18 | 8.61% |
| Nonlinear | 323 | 13.01% |

### Acentric-factor relation

The second relation was

\[
k_0 = \frac{\mu C_v}{M}\,3.75S\frac{R}{C_v},
\]

with

\[
\begin{aligned}
S &= 1 + A\frac{0.215+0.28288A-1.061B+0.26665C}
                 {0.6366+BC+1.061AB},\\
A &= \frac{C_v}{R}-1.5,\\
B &= 0.7862-0.7109\omega+1.3168\omega^2,\\
C &= 2+10.5T_r^2.
\end{aligned}
\]

It achieved 12.46% MAPE, -3.32% bias, and 38.17% P95 error over all 345
compounds. It reduced central error and global bias slightly, but worsened the
upper error tail.

The acentric relation was especially effective for uncorrected hydrocarbons
(7.09% versus 13.97% MAPE), but was worse for most heteroatom classes. Once
functional-group corrections were applied, the geometry-dependent relation was
clearly superior.

## Observed chemical-class biases

The uncorrected relations exhibited strong, directional class effects. Selected
point-weighted results before oxalic exclusion were:

| Class | Geometry relation | Acentric relation | Dominant behavior |
|---|---:|---:|---|
| Hydrocarbon | 13.97% | **7.09%** | Geometry relation predicts low |
| Polar organic | **15.07%** | 17.39% | Heterogeneous |
| Sparse heteroatom | **9.47%** | 12.02% | Mostly low |
| Inorganic | **8.87%** | 10.31% | Relatively balanced |
| Carboxylic acid | **25.26%** | 34.48% | Strongly low |
| Fluorinated | **21.91%** | 27.50% | Usually high, but inconsistent |
| Ketone | 20.73% | **13.87%** | Strongly low |
| Alcohol | **10.62%** | 16.00% | Low |
| Thiol | **3.66%** | 8.98% | Small geometry-relation error |

Carboxylic acids likely violate the monomer ideal-gas assumptions through
vapor association. Fluorinated compounds showed topology-specific errors that
could not be represented by fluorine count or broad organic/inorganic splits.
For example, 1,1- and 1,2-difluoroethane had errors of opposite sign.

### Inorganic treatment

The uncorrected geometry-dependent relation achieved 8.88% MAPE, -1.10% bias,
and 23.56% P95 error over 30 inorganic compounds. Generic chlorine and amine
SMARTS initially leaked corrections learned from organic compounds onto HCl,
Cl2, NF3, and hydrazine, worsening inorganic MAPE to 9.33% and P95 to 28.08%.

The final model therefore applies no fitted group correction to any compound
outside pfdsim's shared strict molecular-organic domain. This gate improved the
total held-out MAPE and P95 slightly while preserving the base inorganic
performance. Carbon-containing conventional inorganics such as HCN, CO, CO2,
CS2, and cyanogen are also rejected by the shared classifier.

## Functional-group correction experiments

### Validation protocol

All temperatures and all Perry rows belonging to one CAS number stayed in the
same fold. The main experiment used five outer folds. Ridge strength was chosen
using four inner compound-held-out folds from

\[
\alpha \in \{0,0.1,1,10,100,1000,10000\}.
\]

Each training compound received equal total weight, regardless of its number of
curves or sample points. Feature support was selected from molecular structure,
not target values. Ordinary groups required at least eight compounds; targeted
acid and fluorine subclasses required at least three and must therefore be
treated as exploratory.

### Correction placement

The tested correction families included:

1. One multiplicative constant per group.
2. Multiplicative group corrections linear or quadratic in \(T_r-1\).
3. Multiplicative group corrections affine in \(1/T_r\).
4. Multiplicative group corrections affine in \(1/T\).
5. Direct additions proportional to \(R/C_v\) in the dimensionless factor.
6. Presence versus occurrence-count group encodings.

Multiplicative presence corrections generalized best. Direct \(R/C_v\)
additions were consistently weaker. Group counts were inferior to binary group
presence.

In the canonical oxalic-free correction comparison, the geometry baseline
improved from 12.46% to 9.03% with constants alone and to 8.56% with linear
\(T_r\) terms. Quadratic terms previously produced no meaningful mean
improvement and worsened extreme errors. Over the same oxalic-free correction
population, the acentric baseline improved only from 12.19% to 10.38%; the raw
all-compound acentric benchmark remains 12.46%.

### Temperature basis, with oxalic acid excluded

| Group structure | Correction basis | Held-out MAPE |
|---|---|---:|
| Broad groups | Constant | 9.03% |
| Broad groups | \(A+B(T_r-1)\) | 8.56% |
| Broad groups | \(A+B/T_r\) | 8.54% |
| Broad groups | \(A+B/T\) | **8.38%** |
| Detailed acid groups | Constant | 8.55% |
| Detailed acid groups | \(A+B(T_r-1)\) | 7.82% |
| Detailed acid groups | \(A+B/T_r\) | 7.65% |
| Detailed acid groups | \(A+B/T\) | **7.45%** |

For numerical conditioning, the implementation uses

\[
a_g+b_g\left(\frac{500\ \mathrm K}{T}-1\right),
\]

which is exactly equivalent to \(A_g+B_g/T\), with

\[
A_g=a_g-b_g,\qquad B_g=500b_g\ \mathrm K.
\]

Inverse absolute temperature outperformed inverse reduced temperature. This is
consistent with the residual depending more directly on absolute thermal
energy and, for associating species, equilibrium-like inverse-temperature
behavior than on critical scaling.

### Acid subclasses

Straight saturated monocarboxylic acids showed a strong chain-length effect and
were split into C1-C3, C4-C6, and C7+ classes. The final held-out results were:

| Acid class | Compounds | Baseline MAPE | Corrected MAPE |
|---|---:|---:|---:|
| Straight saturated C1-C3 | 3 | 46.43% | 23.98% |
| Straight saturated C4-C6 | 3 | 24.53% | 7.61% |
| Straight saturated C7+ | 4 | 24.48% | 5.50% |
| Branched saturated monocarboxylic | 5 | 2.32% | 2.04% |
| Unsaturated/aromatic monocarboxylic | 3 | 4.60% | 0.76% |
| Dicarboxylic, excluding oxalic | 3 | 7.62% | 12.70% |

The branched class required little correction. Dicarboxylic acids remained
internally inconsistent after oxalic removal, so both classes were pruned from
the final parameter set. The three-member subclass results are too small for a
standalone production claim, even when their cross-validation results are good.

### Fluorine subclasses

Replacing the broad fluorine term with mono-fluoro organic, poly-fluoro organic,
and inorganic fluoride terms worsened overall MAPE from 8.76% to 8.89% in the
paired experiment. Combining those splits with the acid split was also worse
than the acid split alone. Fluorine therefore receives no term in the final
model.

### Parameter pruning

Features with less than 15% relative held-out class-MAPE improvement were fixed
exactly to zero. This removed:

- Branched saturated monocarboxylic acid
- Dicarboxylic acid
- Amine
- Bromine
- Ester
- Ether
- Fluorine

After enforcing the organic-domain gate, the intentionally pruned model has
7.458% MAPE versus 7.449% for the larger unpruned candidate. Its P95 is 25.13%
versus 23.92%. The selected model accepts this small accuracy tradeoff to obey
the fixed 15% class-improvement rule and remove seven weak or harmful terms.
The canonical result artifact records both the numerical minimum and the
deliberately selected production candidate.

### Hydrocarbon split

Replacing the common hydrocarbon term with acyclic alkane, saturated cyclic,
alkene, alkyne, and aromatic hydrocarbon terms did not help:

| Metric | Shared hydrocarbon term | Split hydrocarbon terms |
|---|---:|---:|
| MAPE | **7.458%** | 7.516% |
| Compound-equal MAPE | **7.226%** | 7.301% |
| P95 | 25.13% | **24.70%** |
| Bias | **-0.72%** | -1.57% |

The model already contains alkene, alkyne, aromatic-ring, and aliphatic-ring
terms. The shared hydrocarbon feature supplies the common offset, while those
features supply useful structural differences. Splitting the common term adds
mostly redundant parameters.

## Final recommended experimental model

### Baseline

Calculate the geometry-dependent \(k_0\) from viscosity, ideal-gas \(C_v\),
molecular weight, and critical temperature as defined above. Use
\(C_v=C_p-R\) until pfdsim has a more authoritative ideal-gas `Cv` path.

### Correction

Apply

\[
k = k_0\exp\!\left[\sum_g I_g\left(A_g+\frac{B_g}{T}\right)\right],
\]

with `T` in K and the following production coefficients, rounded to five
significant figures. The canonical JSON artifact retains the full-precision
final ridge fit after model selection, using 343 compounds, 18,003 states,
equal compound weights, oxalic exclusion, and \(\alpha=10\). Neither coefficient
table is the source of the held-out performance estimates above.

| Active binary feature \(g\) | \(A_g\) | \(B_g\) [K] |
|---|---:|---:|
| Alcohol | 0.21989 | -76.607 |
| Aldehyde | 0.20855 | -78.969 |
| Aliphatic ring | 0.029391 | -17.068 |
| Alkene | 0.071467 | -21.455 |
| Alkyne | 0.044307 | -21.510 |
| Aromatic ring | 0.0013274 | 16.477 |
| Chlorine | 0.32856 | -114.42 |
| Hydrocarbon only | 0.25035 | -70.837 |
| Ketone | 0.57891 | -222.27 |
| Thiol | -0.00062287 | 21.590 |
| Straight saturated monocarboxylic acid, C1-C3 | -0.69671 | 747.60 |
| Straight saturated monocarboxylic acid, C4-C6 | 0.26548 | 9.3632 |
| Straight saturated monocarboxylic acid, C7+ | 0.81214 | -359.36 |
| Unsaturated/aromatic monocarboxylic acid | -0.031682 | 30.260 |

All indicators are binary, even when a molecule contains multiple occurrences
of a group. Matching active features contribute simultaneously. The acid
subclasses are mutually exclusive. `Hydrocarbon only` may coexist with alkene,
alkyne, or ring features; this is intentional because it represents a shared
hydrocarbon offset plus structural deviations.

The following feature coefficients are exactly zero:

\[
A_g=B_g=0
\]

for branched saturated monocarboxylic acids, dicarboxylic acids, amines,
bromine, esters, ethers, and fluorine. Inorganic compounds bypass the entire
correction independently of these zeroed organic features.

## Final held-out class performance

The table below reports the selected pruned model. Classes overlap, so their
improvements are not additive.

| Class | n | Baseline MAPE | Corrected MAPE | Relative reduction |
|---|---:|---:|---:|---:|
| Straight acid C1-C3 | 3 | 46.43% | 23.98% | 48.3% |
| Straight acid C4-C6 | 3 | 24.53% | 7.61% | 69.0% |
| Straight acid C7+ | 4 | 24.48% | 5.50% | 77.5% |
| Unsaturated/aromatic acid | 3 | 4.60% | 0.76% | 83.5% |
| Hydrocarbon only | 96 | 13.97% | 4.83% | 65.4% |
| Ketone | 17 | 20.73% | 10.39% | 49.9% |
| Aldehyde | 17 | 15.31% | 9.39% | 38.6% |
| Alkene | 35 | 13.07% | 6.33% | 51.6% |
| Alkyne | 16 | 13.17% | 4.64% | 64.8% |
| Alcohol | 30 | 10.62% | 8.07% | 24.0% |
| Thiol | 16 | 3.66% | 1.72% | 53.1% |
| Chlorine-containing organic | 21 | 13.95% | 8.38% | 39.9% |
| Aromatic ring | 49 | 9.44% | 4.87% | 48.5% |
| Aliphatic ring | 30 | 14.03% | 9.46% | 32.6% |

Zeroed features are intentionally absent from this active-feature table.

## Resolver quality policy

The dilute-vapor fallback runs only when all required inputs and a supported
molecular classification are available, after supplied and Perry data.

For each active correction group, the resolver uses its corrected held-out
class MAPE above as a relative mean absolute error. When groups overlap, it
uses the largest group error. Organics with no active group use the overall
7.458% corrected MAPE. Inorganics use their separate 8.88% uncorrected
class MAPE. The model score is `1.00 - 3 * relative_MAE`, bounded below by zero,
rounded to two decimals, and capped at 0.88. Final quality cannot exceed the
weakest resolved input quality; the input limit is conservatively rounded down
to two decimals. No liquid or solid conductivity behavior is inferred.

## Limitations and next validation

- The target and all primary inputs come from Perry. Holding out compounds
  prevents curve leakage but does not provide independent-source validation.
- The benchmark uses empirical Perry viscosity. An end-to-end benchmark using
  pfdsim's actual viscosity resolution hierarchy is still required.
- Several acid subclasses contain only three or four compounds.
- The remaining 108% maximum error shows that the estimator is not uniformly
  reliable, particularly for chemically unusual compounds.
- Functional-group selection and the 15% pruning cutoff were informed by this
  dataset. The resolver quality score is a heuristic based on internal
  validation, not an externally calibrated probability; a fresh external
  dataset is needed to validate it.
- Dense-gas pressure corrections were not studied. The model is a dilute-vapor
  relation.
- Oxalic acid is excluded from fitting and validation, not declared physically
  solved. When Perry data are available, the resolver should continue to prefer
  them.

The most valuable next experiment is an external, compound-held-out validation
against a non-Perry gas-conductivity source, followed by an end-to-end run using
pfdsim-resolved rather than Perry-fixed viscosity.
