# Nannoolal benchmark & calibration scripts

Companion scripts for `nannoolal_method.py` (Parts 1-3 + thesis eq. 8-6).
Policy conclusions live in `compaction_notes.md`; reports in `outputs/`.
All scripts are standalone (`python <script>.py`), use absolute repo paths,
and pull SMILES by CAS/name via `chemicals`.

## Validation & benchmarks

- `psat_validate.py` - the 8 numeric worked examples of Part 3 (Tables
  15a-15h) + dHvap/mono-alcohol/thioester spot checks.  Mirrors the unit
  tests; useful for quick manual reruns while editing the module.
- `psat_benchmark.py` - 4-source Psat benchmark (Perry 2-8 correlations,
  chemicals Wagner fits, Perry 2-10 point tables, data/antoine.txt), binned
  ELP/LP/MP/HP.  Writes `outputs/nannoolal_psat_benchmark.txt`.
- `final_nannoolal_benchmark.py` - the Tb/Tc/Pc/Vc benchmark vs Perry 2-106
  + corrected ATN Part-12 review (pre-dates the 2026-07-10 35/36 chain-rule
  fix; alcohol/diol/glycol-ether rows of its stored output are stale).
- `omega_aw_benchmark.py` - acentric factor in 3 information tiers +
  first NN-vs-Ambrose-Walton Psat comparison.  Writes
  `outputs/nannoolal_omega_aw_benchmark.txt`.
- `omega_lk_classifier_tiers.py` - NN/LK/mean/polarity-classifier omega
  across tiers A/B/C; the script behind the omega-rung policy (LK-only
  values, classifier as quality judge).
- `aw_nn_matrix.py` - the 12-arm NN x AW anchor/carrier matrix (cases 1-6d).
  Writes `outputs/nannoolal_aw_matrix.txt`.  Contains the note on why
  (Tc, Pc) alone cannot determine (Tb, omega).
- `hvap_benchmark.py` - Hvap slope x dZ decomposition (ref/NN slope x
  {1, PTV, virial+Rackett}) vs Perry 2-150 correlations + CRC experimental
  points.  Writes `outputs/nannoolal_hvap_benchmark.txt`.
- `compare_hvap_formula.py` - focused comparison of raw, virial/Rackett-, and
  PTV-corrected Nannoolal Hvap against the corresponding-states `Hvap/(R Tc)`
  formula and Watson-scaled Perry/CRC `Hvap(Tb)` anchors through `Tr=0.99`,
  with Trouton baselines and real/Nannoolal-chain property tiers. Writes
  `outputs/nannoolal_vs_hvap_formula.txt`.
- `compare_estimated_deltaz.py` - compares virial/Rackett, Peng-Robinson, and
  PTV delta-Z corrections for Nannoolal Hvap through `Tr=0.8`, with a real Tb
  but Nannoolal-estimated criticals and Lee-Kesler omega. Writes
  `outputs/nannoolal_estimated_deltaz_comparison.txt`.
- `compare_acid_hvap_dimer_correction.py` - tests the production generic
  vapor-dimerization correction against the carboxylic-acid Hvap cases excluded
  from the ordinary Nannoolal, corresponding-states, and Trouton comparisons.
  Writes `outputs/nannoolal_acid_hvap_dimer_correction.txt`.
- `visc_benchmark.py` - Part-4 liquid viscosity (dBv/Tv) vs the Hsu native
  engine: NN with real Tb / fully predictive / anchored at one ~298 K point,
  per-class on Perry 2-313 + a VDI-PPDS-only block.  Writes
  `outputs/nannoolal_visc_benchmark.txt`.  2026-07-12 headline (240-compound
  common subset, post domain-gates): NN-anchored 11.9 %/7.1 % median beats
  everything nearly everywhere (exceptions: phenols, C7+ acids); Hsu
  14.0 %/10.8 % is the best predictive arm (wins alkanes, unsaturated HC,
  alcohols, phenols, acids); NN predictive 21-23 % but covers ~40 more
  compounds.  Tr profile: everyone's sweet spot is Tr 0.5-0.6 (NN-anc
  6.8/3.1 %); both degrade below Tr 0.4 (24-40 %) and above 0.7 (NN-anc
  keeps the better median, Hsu the better mean).  VDI-only block after the
  2026-07-12 domain gates (small dimerizing acids, ortho-chelates, heavy
  perhalomethanes, hydrazines, substituted CCl3): Hsu mean 66 % -> 20.3 %,
  NN-anc 14.8 %/7.7 % -- the block now behaves like slightly-harder Perry.

## Calibration provenance (fits baked into nannoolal_method.py)

- `refit_group69_aromatic_nitro.py` - read-only inverse diagnostic for the
  missing aromatic-nitro group-69 Tc/Pc/Vc contributions.  Preserves Perry,
  q>0.90 EOS-effective, and PFDSim-compiled experimental records separately;
  reports per-compound implied values, experimental uncertainty intervals,
  alternate-Tb sensitivity, and a clean single-Q subset.  It does not install
  a refit.

- `subtype_split.py` - the 3-regime OH-OH Tc split (36+36 -> published,
  34-involved -> -252, 35-involved -> 0); source of `_AA_TC_34`/`_AA_TC_35`.
- `refit_aa_nonvicinal.py` - the exploratory per-diol exact-fit scan that
  revealed the subtype clustering (superseded by subtype_split.py).
- `flip_impact.py` - before/after impact of the 35/36 chain-rule fix on the
  benchmark databases (21 flipped compounds).
- `pc_and_ternary_probe.py` - diol AA-Pc exact-fit clusters (no refit
  justified) and the ternary A-A-D Tc probe (rejected: 4-oxa-1,7-heptanediol
  contradicts it; dipropylene glycol data would settle it).
