# GRAPPA normal boiling temperatures versus Nannoolal

Investigation completed 2026-10-13. Both methods predict from molecular
structure alone. No experimental boiling temperature is supplied to either
model. GRAPPA predicts Antoine coefficients and evaluates
`Tb/K = B / (A - ln(101.325)) - C`, with pressure in kPa.

## Original Nannoolal benchmark population

The Tb-only pass of
[`final_nannoolal_benchmark.py`](../nannoolal/final_nannoolal_benchmark.py)
uses Perry Table 2-10, excluding chloral hydrate, rows without Tb, and rows
with listed Tm >= reference Tb - 2 K. Its PFDSim-validated overrides take
priority over Perry temperatures. All 14 override identities resolved; six
occur in the selected rows. The current selection contains **909 references**
from **921 table compounds**. Identity resolution uses the same offline
`ChemicalDatabase.resolve_smiles_info` call as the original benchmark.

Nannoolal was recomputed with current defaults and local calibrations.
The older stored report gives 875 estimates and 10.00 K MAE, but its README
explicitly warns that it predates a subsequent chain-rule correction. The
current run gives **869 Nannoolal estimates**, **854 GRAPPA estimates**, and
**828 compounds shared by both**. No extra chemistry or temperature filtering
was imposed on the original population's matched headline.

| Metric, 828 identical compounds | GRAPPA | Current Nannoolal |
|---|---:|---:|
| Mean absolute error / K | **5.4821** | 9.6588 |
| Median absolute error / K | **2.2292** | 6.2417 |
| RMS error / K | **12.9898** | 16.0640 |
| Signed bias / K | +1.8473 | +0.6404 |
| 90th-percentile absolute error / K | **11.7183** | 19.5234 |
| 95th-percentile absolute error / K | **20.6973** | 28.2340 |
| Mean absolute percentage error | **1.1147%** | 2.2102% |
| Fraction within 5 K | **73.43%** | 42.87% |
| Fraction within 10 K | **88.16%** | 68.24% |

GRAPPA reduces MAE by **43.24%** and RMS by **19.14%**, and wins on
**596/828 compounds**. Twenty thousand paired bootstrap replicates, seed
20261012, give a 95% percentile interval for GRAPPA-minus-Nannoolal MAE of
**[-5.0440, -3.3297] K**. Resampling treats compounds as independent despite
shared chemical families; this is an exploratory interval conditional on the
reference collection, not proof of generalization to unseen chemical space.

### Temperature-domain sensitivity

| Reference-temperature subset | Compounds | GRAPPA MAE / K | Nannoolal MAE / K | GRAPPA RMS / K | Nannoolal RMS / K |
|---|---:|---:|---:|---:|---:|
| 250–600 K | 772 | **4.7842** | 9.3565 | **11.7401** | 15.7454 |
| Outside 250–600 K | 56 | 15.1037 | **13.8272** | 24.3882 | **19.9436** |

The second row is extrapolation beyond GRAPPA's published temperature range.
Even in the first row, ten GRAPPA predictions fall outside that range; they
are retained because filtering by the predicted answer would bias evaluation.
Excluding the table's three flagged CAS collisions leaves 825 matched cases
and MAEs **5.4610 K** versus **9.6542 K**, preserving the conclusion.

On each model's own coverage, rather than a fair matched comparison, GRAPPA's
MAE is **5.6299 K over 854 compounds**, and Nannoolal's is **9.7870 K over 869**.
GRAPPA's missing coverage mainly reflects charged structures and unsupported
elements such as Si, Se, Sn, As, Pb, and B. Six selected references have no
resolved offline structure. Exact rejection messages are saved.

### Outliers retained

| Compound | Reference Tb / K | GRAPPA / K | Nannoolal / K |
|---|---:|---:|---:|
| tri-p-Cresyl phosphate | 586.15 | 776.98 | 715.83 |
| Triphenylmethane | 532.35 | 647.64 | 626.09 |
| Quinizarin | 723.15 | 824.68 | 726.20 |
| Ethanedinitrile | 252.15 | 249.12 | 387.40 |

These are discrepancies against the original stored references, not certified
experimental errors. For example, the chemicals CRC entry gives
triphenylmethane **632.15 K**, rather than Perry's stored 532.35 K. NIST's
[triphenylmethane page](https://webbook.nist.gov/cgi/cbook.cgi?ID=C519733&Mask=27FF&Units=SI)
also contains two entries near 632 K, alongside an inconsistent 449 K entry.
No reference correction or selective outlier removal was made.

## Expansion using chemicals 1.5.2

[`benchmark_grappa_boiling_chemicals.py`](benchmark_grappa_boiling_chemicals.py)
uses explicit source tables rather than `Tb(CAS)`'s automatic fallback.
Reference priority is HEOS, CRC_ORG, CRC_INORG, COMMON_CHEMISTRY, WEBBOOK.
Only one reference per CAS is retained. This priority is determined before
predictions, not by which target favors a method. Where available, a
same-source Tm >= Tb - 2 K rejects that source row. Chloral hydrate remains
excluded. Source conflicts are saved rather than silently averaged.

The [library documentation](https://chemicals.readthedocs.io/chemicals.phase_change.html)
identifies Joback as an estimator and Yaws as a compilation of experimental
and predicted values. Neither is used as a reference target here. Wikidata
is also omitted in favor of the more directly attributed compilations.

| Source | Nonmissing Tb entries before selection | Preferred references retained | Matched compounds |
|---|---:|---:|---:|
| HEOS | 140 | 140 | 110 |
| CRC organic | 4,986 | 4,875 | 4,162 |
| CRC inorganic | 556 | 509 | 0 |
| CAS Common Chemistry | 10,419 | 5,954 | 3,137 |
| NIST WebBook | 5,847 | 1,676 | 1,090 |
| Unique selected union | — | **13,154** | **8,499** |

There are **12,306 new reference CAS numbers** beyond the original 909-row
Perry selection. GRAPPA can predict **8,896**, Nannoolal **9,183**, and both
**8,499** chemicals entries. The matched HEOS/CRC subset contains **4,272**
compounds, including **3,564 absent from the original Perry selection**.
Zero matched inorganic-CRC compounds reflects the supported molecular domains
and source priority; it does not mean that inorganic CRC has no Tb records.

### Matched source-specific accuracy

| Preferred source | Compounds | GRAPPA MAE / K | Nannoolal MAE / K | GRAPPA median / K | Nannoolal median / K | GRAPPA RMS / K | Nannoolal RMS / K |
|---|---:|---:|---:|---:|---:|---:|---:|
| HEOS | 110 | **2.2310** | 8.6951 | **1.0774** | 6.4570 | **4.5818** | 11.7138 |
| CRC organic | 4,162 | **8.7781** | 12.0506 | **3.8059** | 6.6451 | **22.0187** | 25.2136 |
| Common Chemistry | 3,137 | 55.2796 | **51.1429** | **9.1578** | 13.4777 | 115.4604 | **86.2070** |
| WebBook | 1,090 | **12.6844** | 15.0998 | **4.4435** | 6.8455 | **34.3500** | 35.6871 |

**HEOS/CRC is the stronger expanded reference comparison:** GRAPPA has
MAE **8.6095 K**, median **3.7076 K**, and RMS **21.7458 K**, versus
Nannoolal's **11.9642 K**, **6.6395 K**, and **24.9577 K**. GRAPPA wins on
**2,731/4,272 compounds**. Its MAE advantage is **3.3547 K**, with a paired
95% bootstrap interval of **[2.7201, 3.9757] K** in GRAPPA's favor.

The full 8,499-case flattened union gives MAE **26.3581 K** versus
**26.8273 K**, median **4.9661 K** versus **8.2505 K**, and RMS
**72.8667 K** versus **56.7403 K**. GRAPPA wins **5,119/8,499** individual
cases but its MAE difference is unresolved: GRAPPA-minus-Nannoolal 95%
interval **[-1.2322, +0.4048] K**. Its larger RMS reflects extreme tails.

Combining the original matched Perry references with only additional CAS
numbers from the flattened union gives **8,548 unique matched compounds**,
MAEs **26.2417 K** and **26.7230 K**. This avoids double weighting compounds
present in both banks; it inherits the flattened union's source limitations.

### Why the flattened union is supplementary

Common Chemistry and WebBook's local Tb columns lack detailed pressure,
decomposition, substance-specific and uncertainty metadata. They are not a
uniformly verified set of normal boiling measurements. For example:

- Hexacontane has Common Chemistry Tb **523.15 K**, versus Yaws **893.15 K**.
  This is a source-condition warning, not proof of which value is correct.
- Polyoxyethylene (20) cetyl ether has Common Chemistry Tb **310.65 K**,
  while the representative discrete graph gives GRAPPA **3022.48 K** and
  Nannoolal **1018.05 K**. A polymeric/commercial substance and a single
  molecular graph need not identify the same experimental material.
- The local WebBook triphenylmethane entry is **450.93765 K**, while the
  organic CRC entry is **632.15 K** and the live NIST page lists conflicting
  boiling temperatures. Deterministic CRC priority avoids using the WebBook
  value for this compound, without consulting model errors.

The original CAS API could not be independently inspected: browser access
failed and the outside-sandbox request returned **HTTP 401**. Reduced pressure
is a plausible explanation for some low entries, but was not confirmed from
that API. No entries were removed simply for disagreeing with the models.
Even reference temperatures inside 250–600 K do not resolve missing pressure
conditions or material identity, so temperature clipping alone cannot clean
this collection. All temperatures and conflicts remain available for audit.

## Numerical verification and reproduction

The original comparison completed in **25.56 s excluding imports**, with
**2.06 s** spent loading GRAPPA and predicting parameters. The larger pass,
using five workers for offline identities/Nannoolal, completed in **47.44 s**,
with **13.64 s** for new GRAPPA parameter predictions. Timings are single-run
measurements, not comprehensive performance estimates. No numerical roots
are needed for GRAPPA boiling points.

The published `GRAPPAnormalbp` wrapper subtracts a full pressure list within
each internal batch of 32, making multi-batch input broadcasting invalid.
The maintained benchmark instead inverts the correctly batched Antoine
output row by row. This is confined to investigation scripts; upstream and
production implementations were not modified. Eighteen boiling temperatures
were checked against that official wrapper with a single batch, and their
direct GRAPPA pressures reproduce **101.325 kPa** within relative **2e-5**.
Another 24 direct-pressure checks passed for the expanded collection.

Independent checks reproduced the exact 909-reference original selection,
overrides, all matched cohort MAE/median/RMS/bias statistics, and all accepted
Antoine inversions. Expanded checks covered **13,154 distinct references**,
all five saved matched populations, **8,896 GRAPPA inversions**, and exact
reuse of overlapping cached predictions. Source/script hashes match; five
installed reference-table hashes and package versions are saved. Ruff and
compilation pass. No production code, existing benchmark output or reference
table was changed.

Artifacts are under `/tmp/grappa-tb-benchmark-8b1q5env/`: original `results/`,
expanded `chemicals-results/`, execution/verification logs, and source
provenance. Each results directory contains compound-level predictions,
exclusions, and a summary; the expanded directory also saves source conflicts.

Run using the existing temporary environment, with new output directories:

```bash
set -o pipefail
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /tmp/grappa-rlpws2CN/venv/bin/python -u \
  scripts/psat_experiments/benchmark_grappa_boiling.py \
  --grappa-root /tmp/grappa-rlpws2CN/GRAPPA \
  --output-dir /tmp/grappa-tb-repeat \
  2>&1 | tee /tmp/grappa-tb-repeat.log

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /tmp/grappa-rlpws2CN/venv/bin/python -u \
  scripts/psat_experiments/benchmark_grappa_boiling_chemicals.py \
  --perry-results /tmp/grappa-tb-repeat \
  --grappa-root /tmp/grappa-rlpws2CN/GRAPPA \
  --output-dir /tmp/grappa-tb-chemicals-repeat --workers 5 \
  2>&1 | tee /tmp/grappa-tb-chemicals-repeat.log
```

GRAPPA training overlap is unknown and its released checkpoint was trained on
all curated DDB data. Nannoolal includes local calibration. These comparisons
measure agreement with the reference banks; they are not independent tests
of unseen chemistry. Coverage and source quality must accompany error scores.
