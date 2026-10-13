# Fusion estimation benchmark

This is an offline investigation of **fusion enthalpy with measured melting
temperature**, not a melting-point predictor or a production resolver change.
Code, extracted benchmark inputs, and retained results live in this directory.
Papers and supplementary sources are collected under
`data/reference/property_estimation/` with descriptive filenames.

## Run

From the repository root:

```bash
python scripts/fusion/benchmark_fusion.py > /tmp/pfdsim-fusion-benchmark.log 2>&1
```

Outputs:

- `results/benchmark.txt`: aggregate results and extraction/identity audit.
- `results/benchmark.json`: every prediction, structural group assignment,
  source descriptor, rejection, paired comparison, chemical-class breakdown,
  implementation replay diagnostic, input hash, and package version.

The normal benchmark requires the existing `chemicals`, RDKit, and NumPy
environment. It performs no network lookup, quantum calculation, fitting,
random sampling, or production-cache hydration. Source measurements are not
used to choose descriptors or adjust any parameter. Existing generated files
are backed up by content hash under `/tmp` before replacement.

To rank all general methods on the same population rather than compare their
different coverage populations:

```bash
python scripts/fusion/compare_methods.py > /tmp/pfdsim-fusion-comparison.log 2>&1
```

This reads the saved predictions and writes `results/comparison.json` and
`results/comparison.txt`. It additionally distinguishes handbook compounds
with a compatible single-transition record from compounds whose transition
history is unknown. Paired bootstrap intervals use 10,000 compound resamples
with seed 1729. They quantify uncertainty within these populations; they
cannot remove shared source lineage or make the chemical sample representative.

To regenerate the extracted inputs from the retained PDFs and spreadsheet:

```bash
uv run --with xlrd python scripts/fusion/prepare_data.py > /tmp/pfdsim-fusion-prepare.log 2>&1
```

This additionally requires `pdftotext` and the temporary preparation dependency
`xlrd`; there is no change to the project's dependency declarations.

Validation:

```bash
python -m pytest scripts/fusion/test_fusion_methods.py scripts/fusion/test_compare_methods.py -q
python3 -m ruff check scripts/fusion
```

Focused checks cover all three
Perry examples, updated group-arithmetic examples, Jain flexibility examples,
acid-supplement predictions, and recovery of either missing numeric column in
the scanned Dannenfelser supplement.

## Methods and their sources

| Report name | Implementation/source |
|---|---|
| `chickos_perry_1991` | Perry 9th, pp. 2-335–2-336, equations 2-42–2-44 and Tables 2-162/163; the cited 1991 Chickos parameters |
| `chickos_structure_1999` | Structure-derived group assignments using the updated Chickos Tables 1/2 and equations 2–7 |
| `chickos_published_1999` | Separately labeled replay of the paper's published group-additivity predictions; no automatic fragment assignment |
| `dy1996_author_descriptors` | Original DY equation 10 with its own supplementary `ln(sigma)`/`ln(phi)` assignments |
| `jain_author_descriptors_2004` | Jain equation VI, `Phi=2.435**tau`, with the authors' supplementary symmetry/flexibility assignments |
| `compernolle_entropy_2011` | Acid-specific equation 21 followed by `H=Tm*S` |
| `compernolle_enthalpy_2011` | Direct acid-specific equation 20 |
| `walden_54_4` | Simple `H=54.4*Tm` control from Chickos 1999 equation 1 |

Bibliography:

1. Chickos, Braton, Hesse, Liebman (1991),
   [10.1021/jo00003a007](https://doi.org/10.1021/jo00003a007).
   The benchmark uses Perry's explicitly reproduced parameter tables for this
   version, not an inferred reconstruction of the paywalled original paper.
2. Chickos, Acree, Liebman (1999),
   [10.1063/1.556045](https://doi.org/10.1063/1.556045).
   The complete paper is retained as
   `data/reference/property_estimation/1999-chickos-acree-liebman-solid-liquid-phase-change-enthalpies-entropies.pdf`.
3. Dannenfelser, Yalkowsky (1996),
   [10.1021/ie940581z](https://doi.org/10.1021/ie940581z).
   Main paper: `data/reference/property_estimation/1996-dannenfelser-yalkowsky-total-entropy-melting.pdf`;
   original supplement: `data/reference/property_estimation/1996-dannenfelser-yalkowsky-total-entropy-melting-supporting-information.pdf`,
   obtained through ACS Figshare.
4. Jain, Yang, Yalkowsky (2004),
   [10.1021/ie0497745](https://doi.org/10.1021/ie0497745).
   Main paper: `data/reference/property_estimation/2004-jain-yang-yalkowsky-total-entropy-melting.pdf`;
   original supplementary XLS: `data/reference/property_estimation/2004-jain-yang-yalkowsky-total-entropy-melting-supporting-information.xls`,
   ACS Figshare file 5170753.
5. Compernolle, Ceulemans, Müller (2011),
   [10.5194/acp-11-8385-2011](https://doi.org/10.5194/acp-11-8385-2011).
   Paper and supplement are retained under `data/reference/property_estimation/`
   with the prefix `2011-compernolle-ceulemans-muller-fusion-properties-functionalised-acids`.
6. Perry Table 2-68: existing `data/perry_heat_of_fusion.json` with original
   row/page provenance and the project's existing identity/printed-erratum audit.
7. `chemicals` 1.5.2: CRC fusion-enthalpy table and CRC organic physical
   constants. The normal run records the actual installed version and hashes
   of both underlying data files.

`data/source_manifest.json` gives source hashes, relative locations, DOI/URL,
and the equations/tables used. No source PDF or spreadsheet is needed during
an ordinary benchmark run; the extracted inputs are retained as readable JSON.

The saved benchmark also contains the full identity-resolved training-CAS
audit. Postprocessing uses that frozen audit, including numerically rejected
training observations, so comparing a saved run never mixes it with a newer
local extraction.

## What the datasets measure

These targets are intentionally separate:

1. **Final fusion enthalpy:** the final melting peak from Chickos Tables 5/7;
   separately, Perry's paired `Tm/Hfus` rows and a same-CAS cross-table CRC
   join. The CRC join does not guarantee that the tables describe the same
   polymorph or experimental specimen.
2. **Total transition entropy:** all solid–solid transition entropies plus
   fusion entropy, from the original DY and Jain supplementary datasets.
3. **Acid total-enthalpy diagnostic:** the Compernolle supplement explicitly
   sums enthalpies across reported transitions. It must not silently be used
   as a final-fusion reference.

The identity-resolved Chickos enthalpy panels distinguish single-transition
observations, known multiple transitions, and paper-excluded observations.
The handbook panels retain an `unknown` transition status when no compatible
Chickos observation is available. Thus `no_reported_multiple_transition` means
that known multiple transitions have been removed; it does **not** certify
that every remaining compound lacks them.

The frozen extraction contains:

| Input | Count |
|---|---:|
| Chickos paired observations, including repeated measurements and qualified rows | 2,349 |
| Chickos formula/name blocks | 2,305 |
| Jain total-entropy observations | 1,799 |
| DY original total-entropy observations recovered | 1,309 |
| DY rows rejected by the OCR/layout parser | 1 |
| Acid supplementary paired observations | 49 |
| Perry paired observations admitted after identity/value checks | 240 |
| CRC observations admitted after identity/value checks | 505 |

Counts are not counts of independent measurements or fitting compounds.
For example, Chickos's paper-level 1,858 training compounds and 260 test
compounds are filtered fitting populations, whereas the tables also include
alternate measurements, starred rows, and extra observations.

The PDF extractor locates columns from each page's headers, preserves the
individual transitions, and checks their sums against printed totals. It
does not guess a missing total or collapse alternate polymorph measurements.
The stored audit includes 39 enthalpy-sum flags, 38 entropy-sum flags, 10
transition-order/value flags, and 3 numeric-column collisions. Flags can
overlap. These rows are excluded from the identity-resolved enthalpy panels.
Unresolved names and formula conflicts remain available in the extraction
audit; they are not assigned a convenient neighboring isomer.

The scanned DY supplement has missing text-layer cells in several columns.
Their positions identify which of observed/predicted/difference is missing;
the identity `observed - predicted = difference` recovers that cell. Each
recovery is labeled. Printed log descriptors are rounded to 0.1; their replay
differs from printed predictions by at most 0.438 J/(mol K). Source `*` and
`+` qualified rows and polymers are omitted from the main small-molecule DY
replay, but retained in its data/audit and diagnostic panels.

One invalid SMILES in the acid supplement is preserved as a source rejection,
rather than reusing the preceding molecule's identity. Stereo labels and
unencoded stereochemistry remain in the raw rows; acid compound aggregation
groups by the supplied molecular connectivity because the source often does
not encode stereochemistry in SMILES.

## Fair comparisons and validation limits

Each method receives the same observed melting temperature for a given row.
Aggregate statistics include absolute MAE/RMSE, MAPE, median/P90/P95 errors,
signed bias, and percentages within 10% and 20%. Compound-equal summaries
first summarize each CAS, then average across compounds, preventing repeated
measurements from dominating. Where CAS is unavailable, aggregation uses the
source structure/name. These fallback keys are not claimed to establish
independence between unresolved identities.

Pairwise tables compare methods on **identical observations**. Broader coverage
must not be mistaken for worse accuracy when looking at unpaired summaries.
No numeric outliers are removed based on estimator error.

Chickos Table 5 is development data. Table 7 is paper-designated test data,
but five resolved CAS identities also occur in its training table. The extra
`chickos_test_single_transition_disjoint_resolved_CAS` panel removes these
overlaps. Unresolved training identities still prevent an absolute guarantee
of chemical independence. Perry and CRC can share underlying measurements
with the source papers. The Compernolle supplementary observations are its
development set; its good in-set results are not external validation.

The Jain supplement obtains its experimental data and Chickos predictions
from the Chickos lineage. It is useful for a controlled same-population
entropy comparison, but is not fresh validation of Chickos. The diagnostic
`dy1996_equation_on_jain_descriptors` column changes only the flexibility base
from 2.435 to 2.85. It is **not** a replay of the original DY assignment rules;
the separate original DY panel and `dy1996_author_descriptors` method supply
that comparison.

DY/Jain symmetry is a specialized melting descriptor, including empirical
effective symmetries and assumptions about rotating terminal groups. It is
not the gas-entropy symmetry implemented in production Domalski–Hearing.
The benchmark therefore uses published assignments and admits a CAS join
only when all matching source assignments agree. It does not choose an
assignment by its agreement with the experimental target. This limits coverage
and does not provide an automatic predictor for arbitrary new molecules.

The structure-derived Chickos implementations cover the tabulated ordinary
CHNOS/halogen groups and explicit composite/ring environments in
`fusion_methods.py`. Rare-element groups and undefined composite environments
are rejected, not approximated with ordinary atoms. They are separate from
the complete published-value replay. The updated implementation reproduces
969 of 1,096 checked source predictions within 0.5 J/(mol K), with median
absolute difference 0.1 J/(mol K). Remaining differences and printed group
expressions are stored in `implementation_audit`; they include source
arithmetic/group-assignment inconsistencies and unresolved assignment
conventions. For example, methyl nonadecanoate's printed expression yields
about 201.0 J/(mol K) while its predicted column prints 102.3. The benchmark
does not tune the structural model to such columns.

Formate esters use their specific 1999 Table 2(a) group; the implemented
Perry table has no formate assignment and rejects them. The long-chain
methylene test counts each multi-atom functional group once while retaining
the physical ring atoms of cyclic groups. Regression checks cover formates,
nitro/sulfone groups, and a lactone whose ring prevents the chain correction.

Automatic flexibility counts are also audited separately: 90 of 1,129
admitted structure-derived counts differ from Jain's published assignments.
These inferred counts do **not** replace the author descriptors in the Jain
prediction columns. The acid implementation uses the source's descriptor
formula within its narrower CHO domain, and is checked against printed acid
examples. This is why these scripts should not be promoted wholesale into a
production property resolver.

Jain's paper uses `R=8.314`; its supplementary predictions mostly reflect a
rounded `19.1*log10(...)` equivalent. The equation replay retains the paper's
constant and records the resulting maximum difference of 0.304 J/(mol K).

## Retained findings

### All general methods on identical, single-transition populations

All methods in each column below use exactly the same measured fusion
enthalpies and melting temperatures. The populations have compatible
single-transition records in the Chickos extraction; they are still handbook
comparisons, not guaranteed independent validation.

| Method | CRC, 123 compounds: MAPE / MAE (kJ/mol) | Perry, 68 compounds: MAPE / MAE (kJ/mol) |
|---|---:|---:|
| Perry Chickos 1991 | 10.71% / 1.331 | 13.13% / 1.694 |
| Updated Chickos 1999 | 11.36% / 1.361 | 12.56% / 1.676 |
| Dannenfelser 1996 | 15.35% / 1.811 | 16.98% / 2.122 |
| Jain 2004 | 17.31% / 2.099 | 17.66% / 2.330 |
| Walden baseline | 23.63% / 3.200 | 18.95% / 2.733 |

The Chickos family is strongest on these populations. Its versions exchange
first place. The paired 95% bootstrap intervals for updated-minus-Perry MAPE
include zero: [-0.985, +2.265] percentage points for CRC, and
[-2.711, +1.531] points for Perry. They do not establish a clear version winner.

In the 44-compound disjoint test intersection below, Perry has smaller
relative error for 27 compounds and the updated method for 17. The updated
mean MAPE is lower, but Perry's absolute MAE and median relative error are
lower. Thus the updated method is not uniformly better, and chemical-class
splits in `comparison.json` are exploratory rather than validated switching
rules. The all-general-method test intersection contains only one compound
because original DY descriptor coverage is limited.

### Broader pairwise coverage intersections

Compound-equal mean absolute percentage error, measured `Tm`, identical
populations for each row:

| Comparison population | Compounds | Perry/1991 | Updated/1999 |
|---|---:|---:|---:|
| Paper-designated single-transition test intersection | 48 | 26.26% | 22.95% |
| Test intersection after resolved training-CAS overlaps are removed | 44 | 27.75% | 24.00% |
| CRC intersection after known multiple-transition cases are removed | 357 | 30.88% | 29.83% |
| Perry intersection after known multiple-transition cases are removed | 193 | 34.34% | 33.36% |

The updated model is modestly better on this relative-error measure, but it
does not dominate every measure: on the 44-compound disjoint test
intersection, MAE is 2.63 kJ/mol for Perry versus 2.99 kJ/mol for updated
Chickos, and median relative errors are 9.17% versus 10.73%. The mean relative
improvement is affected by the tail, not a universal per-compound win.

On the identical 45-compound disjoint test intersection with source-assigned
Jain descriptors, updated structural Chickos has 23.63% mean relative error
versus 31.01% for Jain; absolute MAEs are 2.81 versus 3.39 kJ/mol.

The 1,799-row entropy replay reproduces the published scale of error:
10.38 J/(mol K) MAE for Chickos's tabulated predictions and 12.28 J/(mol K)
for Jain. The vetted original DY small-molecule replay gives 12.49 J/(mol K).
These are distinct datasets/validation conditions and must not be pooled into
a universal ranking.

Treating total transition entropy as final fusion entropy causes much larger
errors on known multiple-transition cases. A future production estimator
needs a transition-aware admission policy before the comparatively modest
choice between these correlations can be useful.

Finally, the 1999 paper's abstract appears to repeat its training standard
deviations as test statistics. Section 7.2 reports test standard deviations
of 18.4 J/(mol K) and 7.2 kJ/mol, and test MAEs of 13.9 J/(mol K) and
5.28 kJ/mol. Those detailed results should take precedence over the abstract's
13.0 J/(mol K)/4.88 kJ/mol figures when describing its test performance.
