# Hsu viscosity method: investigation & benchmark scripts

Companion scripts for `hsu_method.py` (Hsu-Sheu-Tu GC liquid viscosity,
CEJ 88 (2002) 27-35 / Perry 9th Table 2-174).  All standalone.

- `hsu_table6_replay.py` - replays the paper's own Table 6 comparison
  (35 compounds) with the printed Tables 2-4 coefficients.  Established:
  Q7 sign misprint (1-butene +1380% as printed), printed d% signs are
  typeset-mangled but magnitudes reproduce, and Q56/Q88 misprints are in
  the ORIGINAL paper (Perry inherited them faithfully).
- `hsu_group_investigation.py` - the excluded/caution group probe vs Perry
  2-313 + VDI PPDS + Viswanath-Natarajan references.  Source of every
  enable/reject verdict, the shape-vs-bias taxonomy, the implied-anchor-Pc
  (gauge fixing) analysis, and the Sum(d) Pc-sensitivity rule.
- `hsu_refit.py` - leave-one-out-validated linear refits.  Baked into
  hsu_method.py: terminal alkyne unit (LOO 12.0%) and internal alkyne unit
  (LOO 10.4%), both d=0 (a free d term worsened LOO - tested per PFDSim's
  shape/compactness suggestion).  Rejected honestly: polyhydric OH refit
  (LOO 38% even with a vicinal penalty), secondary-amine re-refit (LOO 24%
  vs 9-19% for PFDSim's a,b,d/10 correction).
- `perry_benchmark.py` - acceptance benchmark: every Perry 2-313 compound
  through the native engine with real Pc.  Writes
  `outputs/hsu_perry_benchmark.txt`.  2026-07-12: 252/339 covered
  (previous ugropy route ~230), mean 15.1% / median 11.5% / p90 33.2%,
  quality factors rank the observed error.
