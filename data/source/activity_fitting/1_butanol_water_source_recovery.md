# Recovery of the 1-butanol/water fitting observations

The measurements cited by `1_butanol_water_interactions.json` are recoverable
from Aoki and Moriyoshi's 1978 study through the IUPAC compilation. The exact
historical fitting input is not preserved: the file already contains only
parameters and diagnostics in the initial Git import (`3b8eb43`), and its later
history changes parameter-field names, adds temperature limits, and relocates
the file without adding observations or a fitting script. The second checkout
also contains the same parameter file.

## Saved evidence

- [Machine-readable observations](1_butanol_water_lle_source_data.json): all
  15 one-atmosphere tie lines from the cited study, both mass percentages and
  compiler-calculated mole fractions, the critical point, and a separately
  identified ambient reconstruction candidate.
- [Scanned source pages](../../reference/liquid-liquid-equilibria/1984-barton-butanol-water-solubility-source-excerpt.pdf):
  printed page 49 and pages 82–85 of A. F. M. Barton (ed.), *Alcohols with Water*,
  IUPAC Solubility Data Series, volume 15 (Pergamon Press, 1984).
  These are PDF pages 68 and 101–104 of the
  [complete compilation](https://iupac.github.io/SolubilityDataSeries/volumes/SDS-15.pdf).
  The JSON records SHA-256 hashes of the downloaded volume and saved excerpt.

## Recovered tie lines

All compositions below are **mole fractions of 1-butanol**, including those
in the water-rich phase. The 15 Aoki–Moriyoshi observations are at 1 atm.
The ambient row is a separately sourced candidate, not a confirmed historical
input. Its source sheet does not specify pressure; using it at 1 atm is an
assumption.

| Temperature / K | Water-rich phase | Butanol-rich phase | Source |
| ---: | ---: | ---: | --- |
| 298.15 | 0.0188 | 0.488 | Butler et al. (1933), reconstruction candidate |
| 302.95 | 0.0180 | 0.482 | Aoki and Moriyoshi (1978) |
| 322.75 | 0.0177 | 0.459 | Aoki and Moriyoshi (1978) |
| 332.65 | 0.0180 | 0.436 | Aoki and Moriyoshi (1978) |
| 342.65 | 0.0171 | 0.422 | Aoki and Moriyoshi (1978) |
| 362.65 | 0.0193 | 0.359 | Aoki and Moriyoshi (1978) |
| 372.45 | 0.0249 | 0.328 | Aoki and Moriyoshi (1978) |
| 380.05 | 0.0280 | 0.290 | Aoki and Moriyoshi (1978) |
| 382.95 | 0.0292 | 0.281 | Aoki and Moriyoshi (1978) |
| 391.25 | 0.0396 | 0.233 | Aoki and Moriyoshi (1978) |
| 392.95 | 0.0481 | 0.220 | Aoki and Moriyoshi (1978) |
| 395.25 | 0.0513 | 0.182 | Aoki and Moriyoshi (1978) |
| 395.85 | 0.0536 | 0.183 | Aoki and Moriyoshi (1978) |
| 397.05 | 0.0635 | 0.165 | Aoki and Moriyoshi (1978) |
| 397.45 | 0.0739 | 0.148 | Aoki and Moriyoshi (1978) |
| 397.75 | 0.0859 | 0.134 | Aoki and Moriyoshi (1978) |

The additional critical constraint is **397.85 K, 1 atm, x(1-butanol) = 0.110**.
It is not a finite-width tie line. These temperature limits and critical
properties match the existing fit metadata exactly and are independently
confirmed by the
[publisher's abstract](https://www.sciencedirect.com/science/article/abs/pii/0021961478900344).

## The 298.15 K observation

Butler, Thomson, and Maclennan, *J. Chem. Soc.* (1933), 674–686,
[DOI 10.1039/JR9330000674](https://pubs.rsc.org/en/content/articlelanding/1933/jr/jr9330000674),
report a 25.00 °C row reproduced on printed page 49 of the compilation:
7.31 g butanol/100 g solution in the water-rich phase and 79.64 g
butanol/100 g solution in the alcohol-rich phase. The compiler gives
butanol mole fractions 0.0188 and 0.488, respectively.

This provides a directly tabulated literature observation at the requested
temperature. It does not prove that the original fit used this particular
observation or these rounded mole fractions. The existing parameter file calls
its ambient input a "supplied" point without recording its composition or
provenance. The recovery JSON therefore keeps this candidate separate from
the 15 observations of the explicitly cited study.

## Precision and data quality

The saved mole fractions are the compilation's printed values, not newly
calculated values. Mass percentages and mole fractions have independent finite
precision and small discrepancies. With molecular weights 74.1216 g/mol for
butanol and 18.01528 g/mol for water, conversion uses

`x = (w / 74.1216) / ((w / 74.1216) + ((1 - w) / 18.01528))`,

where `w` is the butanol mass fraction. The maximum absolute difference from
the 30 compiled mole fractions of the 1978 study is about 0.000619. The ambient
mass percentages convert to approximately 0.0188077 and 0.4873671, compared
with the printed 0.0188 and 0.488. Preserve the printed columns as distinct
source values rather than silently replacing either column to enforce exact
conversion agreement.

The critical evaluation on printed page 33 of the full compilation explicitly
rejects this study's water-rich points around 323 K and 333 K because they
disagree with other studies. The recovery retains the original 322.75 K and
332.65 K rows; that quality assessment matters if these data are used for a
new fit. The study's compiled mean solubility error is ±0.28 g butanol/100 g
solution, and its thermostat control is ±0.02 K. These are not automatically
per-point statistical standard deviations.

The source excerpt also includes high-pressure measurements. They are not
included in the recovered fitting dataset because the parameter file explicitly
describes a 1-atm fit. Its printed variables heading incorrectly equates
2450 atm with 25 MPa; 2450 atm is approximately 248.25 MPa. Also, the
384.45 K / 2000 atm row on printed page 84 pairs an alcohol-rich mass percentage
of 41.0 with x = 0.227, although mass conversion gives approximately 0.1445.
That apparent source-table error does not affect any recovered 1-atm row.

## Limits of reproduction

The historical diagnostics report a 40% UCST objective weight for both models,
tie-line RMS values around 0.0105 (NRTL) and 0.0120 (UNIQUAC), and predicted
critical points. They do not preserve the objective's residual definition or
normalization, optimizer settings, UNIQUAC structural basis, exact ambient
input, or the numerical precision used when fitting. The recovered literature
observations therefore establish the stated experimental basis, but do not
guarantee bit-for-bit reproduction of the fitted coefficients.

The existing parameter JSON and generated runtime interaction tables were
left unchanged. This recovery adds source evidence only.
