# Ordinary-liquid heat-capacity results

This directory contains the offline ordinary-liquid heat-capacity compiler and
the retained conclusions from the liquid-\(C_p\) estimator investigation.
These notes distinguish ordinary isobaric liquid heat capacity from the
derivative of saturated-liquid enthalpy along the saturation curve.

## Thermodynamic meaning

The runtime property is ordinary constant-pressure liquid heat capacity,

\[
C_{p,l}=\left(\frac{\partial H_l}{\partial T}\right)_P.
\]

Zabransky records marked `sat` approximate the saturation-path derivative

\[
C_\sigma=\frac{dH_{\mathrm{sat},l}}{dT},
\]

and are therefore excluded. Comparisons against CoolProp showed that `sat`
tracks \(C_\sigma\), particularly above \(0.85T_c\), while VDI ordinary-liquid
data track isobaric `Cpmolar`.

## Canonical database

The compiler is `build_liquid_heat_capacity_db.py`; its runtime artifact is
`data/liquid_heat_capacity.sqlite`.

Final database statistics:

```text
Canonical records:                    1,130
Candidate records:                    1,458
Unique candidate CAS:                 1,133
Quarantined entries:                      8
Unavailable identities:                  3
Stored primitive maximum error:  1.13e-7
Runtime load failures:              0/1,130
```

Selected sources:

```text
Zabransky p spline:      468
Perry:                   314
NIST WebBook Shomate:    218
CoolProp:                 71
Zabransky C spline:       31
Zabransky p quasi:        23
Zabransky C quasi:         5
```

Stored models:

```text
Linear Chebyshev:                  1,102
Native Zabransky quasipolynomial:     28
```

The linear Chebyshev degree distribution is:

```text
Degree 8:     1,018
Degree 12:       67
Degree 16:       17
```

The compiler never raises the source minimum temperature. It attempts degrees
8, 12, and 16 and limits ordinary-liquid curves to \(0.95T_c\) where a critical
temperature is known. Native Zabransky quasipolynomials retain their original
six coefficients and analytic enthalpy and entropy primitives.

## Runtime source order

Ordinary-liquid resolution uses one retained executable kernel for scalar
\(C_p\), enthalpy, entropy, provenance, and range-quality reporting:

```text
1. Provided .pfd Cpl correlation
2. Explicit .pfd constant Cp_liquid
3. Bundled liquid_heat_capacity.sqlite
4. Non-.pfd Cp_liquid point at 298.15 K
5. Persisted online NIST kernel
6. Online NIST kernel
7. Predictive liquid-Cp estimator
8. 1.3 x resolved ideal-gas Cp kernel
```

An explicit range on a provided `Cpl` correlation is authoritative. When an
endpoint is omitted, the lower endpoint defaults to resolved `Tm` and the upper
endpoint defaults to resolved `Tb`. If the corresponding phase point cannot be
resolved or would make the completed interval invalid, the endpoint falls back
to 273.15 K or 1500 K, respectively. The selected values and phase-point
provenance are stored with the kernel.

The predictive rung was originally reserved for Rowlinson--Poling. The
investigation below supports a hybrid Rowlinson--Bondi and HBD ratio-GC rung
instead.

Liquid range conditioning uses:

```text
Extrapolation width:                10 K
Initial quality penalty:           0.02
Additional penalty per 5 K:        0.02
Maximum accumulated penalty:       0.40
```

Cubic EOS models retain \(C_{p,ig}+dH^{dep}/dT\), and STEAM retains IF97, so
their reported heat capacities remain derivatives of their own enthalpy
models.

## Rowlinson forms

Define \(T_r=T/T_c\). The later Rowlinson--Poling form is

\[
\frac{C_{p,l}-C_{p,ig}}R =
1.586+\frac{0.49}{1-T_r}
+\omega\left[
4.2775+\frac{6.3(1-T_r)^{1/3}}{T_r}
+\frac{0.4355}{1-T_r}
\right].
\]

The older Rowlinson--Bondi form is

\[
\frac{C_{p,l}-C_{p,ig}}R =
1.45+\frac{0.45}{1-T_r}
+\omega\left[
4.2775+\frac{6.3(1-T_r)^{1/3}}{T_r}
+\frac{0.4355}{1-T_r}
\right].
\]

The apparent differences sometimes printed inside the acentric bracket are
only the expansion of an outside factor of \(0.25\). The two published forms
differ materially only in the first two constants.

### Broad canonical benchmark

The initial matched benchmark contained 398 compounds and 9,950 temperature
points over \(0.30\leq T_r\leq0.95\). It required canonical liquid and gas
curves plus effective \(T_c\) and \(\omega\).

```text
                              Poling       Bondi       1.3 x Cpig
Point median error             2.824%       2.787%      15.358%
Point mean error               6.622%       6.522%      17.867%
Case-median MARD               2.981%       2.908%      15.277%
```

The large mean relative to the median came from associating compounds and
inorganic or elemental liquids.

### Strict chemical partition

The strict organic definition treats conventional inorganic carbon compounds
such as HCN as inorganic and ordinary molecular carbon compounds as organic.
RDKit supplies Lipinski HBD counts.

```text
Population                 n    Form       Mean MARD   Case median   Case P95
Organic, no HBD          280    Poling        3.778%       2.225%      12.292%
                               Bondi         3.570%       1.972%      10.879%
Organic, HBD              86    Poling       11.021%       8.236%      33.606%
                               Bondi        11.151%       8.607%      32.728%
All inorganics            31    Poling       20.241%       4.926%      41.498%
                               Bondi        20.435%       6.101%      38.703%
```

For non-HBD organics, Bondi placed 79.3% of compounds below 5% MARD and 92.5%
below 10%. Refitting its two differing constants did not produce a meaningful
out-of-sample improvement. The published \(1.45\) and \(0.45\) values remain
on the empirical Pareto frontier and should be retained.

Inorganics have a deceptively acceptable median but an unacceptable tail.
They must not enter the Rowlinson rung.

## HBD residual structure

The direction and magnitude of Rowlinson error depend strongly on donor class.
Before glycerol was added, Bondi produced:

```text
Donor class          n    Mean bias    Mean MARD
Alcohol OH          34      -3.80%       16.15%
Amide-like NH        3     +13.61%       13.84%
Amine NH            13      -3.47%        5.68%
Carboxylic acid     19      +5.40%       10.70%
Phenol               6      -4.33%        9.40%
Thiol SH            10      -2.39%        2.79%
```

The gas/liquid ratio showed a strong molecular-size trend:

```text
Cpig/Cpl versus heavy atoms:       Spearman rho = +0.660
Cpig/Cpl versus heavy atoms/HBD:   Spearman rho = +0.609
Cpig/Cpl versus MW/HBD:            Spearman rho = +0.640
Cpig/Cpl versus HBD count alone:   Spearman rho = -0.042
```

The number of donors alone is not predictive. The relevant quantity is donor
density or molecular size per donor. The ratio is more useful for predicting
error magnitude than Rowlinson error direction.

Thiol-only compounds are already described well by Bondi and should not be
treated as members of the problematic O--H/N--H population.

## Direct HBD ratio model

For O--H/N--H organics, directly predicting

\[
r(T)=\frac{C_{p,ig}(T)}{C_{p,l}(T)}
\]

substantially outperformed Rowlinson. CAS-grouped validation keeps every
temperature for a held-out compound outside training.

The original 37-column screen achieved about 5.4% mean case MARD. Removing
zero and algebraically redundant columns, \(\omega\), heteroatom density,
quadratic temperature terms, and unsupported class splits showed that a
15-feature polyol model retains the useful accuracy. Cutting to 11 features
by removing either class-specific size or temperature slopes caused a
repeatable loss.

### Glycerol and the polyol split

Glycerol has a resolver-supplied statistical-mechanical gas curve:

```text
Source:   Burcat/ReSpecTh NASA-9 curve, Shomate refit
Range:    250-1000 K
Quality:  0.90
Cpig at 298.15 K: 131.718 J/(mol K)
```

Its canonical liquid curve spans 293.1-382.7 K. Adding glycerol produced the
first three-OH polyol and increased the polyol set to four compounds:

```text
Ethylene glycol
1,2-Propylene glycol
1,3-Butanediol
Glycerol
```

The held-out comparison was:

```text
Model                               Glycerol MARD   Four-polyol MARD
No separate polyol class                6.762%           5.560%
Separate fraction-coded polyol          2.953%           3.242%
Explicit count plus polyol             24.585%           9.713%
```

Glycerol exposed the explicit donor-count result as an extrapolation failure.
The final model therefore uses donor fractions, a separate polyol class, and
\(\ln N_{ONH}\), not a contribution added once per OH bond.

## Shomate compression of JSON Zabransky curves

Four former `chemicals.json` liquid correlations were polynomial refits of
constant-pressure Zabransky splines. They were removed after the canonical
database became authoritative; historical coefficients remain only in the
benchmark fixture. A five-coefficient Shomate refit was tested with the
compiler's quality objective. Over the former JSON ranges:

```text
Compound       Shomate MAPE   Maximum error   Resulting quality
Glycerol          ~0%             ~0%              0.900000
Aniline          0.3002%          1.3507%           0.862589
Pyridine         0.01056%         0.02261%          0.957345
Acetophenone      ~0%             ~0%              0.960000
```

Glycerol is linear in temperature and acetophenone is cubic over its stored
range, so both lie exactly inside the Shomate basis with \(E=0\). Pyridine is
better represented by Shomate than by the existing `poly_x` curve. Aniline
crosses multiple piecewise-spline segments; no single smooth five-term curve
compresses it well.

Extending the fits to the full compiler-admitted ranges is inappropriate for
the segmented high-temperature curves:

```text
Compound       Full-range Shomate quality
Glycerol                    0.900000
Aniline                     0.694265
Pyridine                    0.957324
Acetophenone                0.798206
```

The benchmark and full coefficients are retained in
`benchmark_zabransky_json_shomate.py`. The canonical native/quasi or
Chebyshev records remain preferable to duplicate JSON approximations.

## Final recommended formula

### Applicability

Use this GC only for strict molecular organics with at least one O--H or N--H
donor. Define \(N_{ONH}\) as the number of O--H and N--H donors; exclude S--H.

Recommended routing:

```text
Strict non-HBD organic:             Rowlinson--Bondi
Thiol-only organic:                 Rowlinson--Bondi
Single-class O-H/N-H organic:       ratio GC below
Pure polyol:                        ratio GC below
Mixed O/N donor classes:            insufficient validation; prefer Bondi
Inorganic or unresolved structure:  bypass both to terminal fallback
```

Implemented quality multipliers are applied to the weakest required input:

```text
Non-HBD or thiol-only Bondi:  0.89 x min(QCpig, QTc, Qomega, Qidentity)
HBD ratio GC:                 0.82 x min(QCpig, QTc, Qidentity)
Mixed-donor provisional Bondi:0.70 x min(QCpig, QTc, Qomega, Qidentity)
1.3 x Cpig terminal fallback: 0.60 x QCpig
```

Predictive admission also requires every critical input used by the method to
have quality at least `0.70`. Bondi therefore requires both \(Q_{T_c}\geq0.70\)
and \(Q_\omega\geq0.70\); the HBD ratio GC requires
\(Q_{T_c}\geq0.70\). A failed gate continues to the `1.3 x Cpig` rung.

When SMILES is unavailable, a strict organic formula with no donor-capable
heteroatom can use ordinary Bondi. A formula containing O, N, S, P, or Se but
without enough structure to determine donor type uses provisional Bondi at
the `0.70` multiplier. Formula-only and structure-aware admission share the
central classifier in `property_resolution/organic_classification.py`.

The model was validated over \(0.30\leq T_r\leq0.95\). Outside that interval,
use ordinary liquid-kernel range conditioning rather than evaluating it
unbounded.

### Definitions

\[
s=\ln\left(\frac{C_{p,ig}}{R N_{ONH}}\right),
\qquad T_r=\frac{T}{T_c},
\qquad y=\ln\left(\frac{C_{p,ig}}{C_{p,l}}\right).
\]

Donor fractions are donor counts divided by \(N_{ONH}\):

```text
f_polyol
f_nitrogen
f_acid
f_phenol
```

Ordinary alcohol OH is the reference class. A compound is a polyol only when
it has at least two ordinary alcohol-OH donors and all of its donors are
ordinary alcohol OH. Amines, amides, and aromatic N--H share the nitrogen
class. Carboxylic-acid OH and phenolic OH remain distinct.

For mixed classes, the fractions sum to one together with the implicit
ordinary-alcohol fraction. This interpolation is mathematically defined but
is not yet empirically supported by enough mixed-donor compounds.

### Fitted equation

\[
\begin{aligned}
y={}&-0.6965662470
+0.1329644055s
-0.4221203264T_r
+0.2275997437\ln N_{ONH}
\\
&+f_{polyol}\left(
-0.3181350137+0.04165256631s+0.2635252263T_r
\right)
\\
&+f_{nitrogen}\left(
-0.7094419314+0.1725632336s+0.6135363104T_r
\right)
\\
&+f_{acid}\left(
-0.1811372784+0.05081944249s+0.2090743470T_r
\right)
\\
&+f_{phenol}\left(
-0.9425459878+0.2605313918s+0.4554492123T_r
\right).
\end{aligned}
\]

Apply a defensive ratio bound and recover liquid heat capacity:

\[
\hat r=\min\left(1.25,\max\left(0.15,e^y\right)\right),
\qquad
\boxed{C_{p,l}=\frac{C_{p,ig}}{\hat r}}.
\]

The model has 15 predictor coefficients plus an intercept. It was fitted by
ridge regression on standardized predictors with \(\alpha=1\), selected by
repeated grouped validation. The equation above has already been converted to
raw feature space and requires no runtime scaler.

### Held-out error

The final O/N data contain 77 compounds and 1,925 temperature points. The
table reports five-repeat, ten-fold, CAS-grouped held-out errors. Thiols use
Bondi rather than the GC.

```text
Class                    n    Mean MARD   Median MARD   P95 MARD    Bias
Monohydric alcohol      31       6.327%        5.838%    11.459%   +0.669%
Polyol                   4       3.242%        2.948%     5.896%   +1.667%
Nitrogen donor          16       4.605%        4.010%     9.288%   +0.174%
Carboxylic acid         19       6.198%        4.421%    14.426%   +0.330%
Phenol                   6       7.706%        6.645%    13.908%   -0.971%
Mixed O/N donor          1      17.595%       17.595%    17.595%  -17.595%
Thiol-only, Bondi       10       2.793%        2.969%     5.432%   -2.393%
```

Aggregates:

```text
Population          n    Mean MARD   Median MARD   P95 MARD    Bias    <10%
O/N ratio GC       77       6.031%        4.727%    14.531%   +0.169%   85.7%
Thiol Bondi        10       2.793%        2.969%     5.432%   -2.393%  100.0%
Complete hybrid    87       5.659%        4.478%    14.531%   -0.125%   87.4%
Bondi for all      87      11.175%        8.987%    32.728%   -0.697%   55.2%
```

The single mixed O/N compound was diethanolamine. Bondi gave about 6.78% MARD
for it versus 17.60% for fractional GC interpolation, so mixed donor classes
must remain outside the validated GC admission gate until more data exist.

## Runtime implementation

The predictive rung is implemented in `property_resolution/heat_capacity.py`.
Each admitted estimator is compiled over \(0.30\leq T_r\leq0.95\), intersected
with the resolved ideal-gas kernel range, into a linear-Chebyshev liquid kernel.
That portable kernel supplies mutually consistent \(C_p\), \(\Delta H\), and
\(\Delta S\) and uses ordinary liquid range conditioning outside its admitted
range.

The centralized strict classifier rejects conventional inorganic carbon
families, net-charged/disconnected salts, ionic carbonates/bicarbonates, and
metal carbides while retaining organic nitriles, formic/oxalic acids, urea,
carbon tetrachloride, neutral molecular organometallics, and ordinary
organosilicon compounds.
