# Handoff notes: Psat / omega / Hvap estimation rungs (2026-07-10)

> **Vapor-pressure policy superseded on 2026-07-16.** The Psat carrier and
> source hierarchy below are retained as historical benchmark notes, not the
> current implementation plan. See
> `docs/vapor_pressure_canonicalization.md` for the canonical PFD/CoolProp/
> source-segment architecture and current completion policy. The viscosity and
> underlying Nannoolal benchmark sections remain applicable.

Status: benchmarked and decided, NOT yet wired into `property_resolution/`.
All machinery exists in `nannoolal_method.py` (NN = Nannoolal Part-3 slope
model) and `chemicals` (AW = Ambrose-Walton, LK = Lee-Kesler omega).
Benchmarks: `outputs/nannoolal_psat_benchmark.txt`,
`outputs/nannoolal_omega_aw_benchmark.txt`, `outputs/nannoolal_aw_matrix.txt`.

## Recommended policy (the short version)

### Vapor pressure (structure-estimation rung, below any fitted correlation)

Two independent decisions: pick the ANCHOR, then pick the CARRIER.

Anchor (best available, in order):
1. real Tb                                    -> floor ~2 % MARD at MP
2. real (Tc, Pc, omega), no Tb: pseudo-point (0.7 Tc, Pc*10^-(1+omega)),
   fed to NN via `estimate_psat(smiles, psat_point=...)`  -> ~2 % at MP
3. real Tc only: Tb = Tc / g(structure) where g = estimate(s, tb=100).tc_K/100
   (NN Tc/Tb ratio inversion, ~4-5 K)          -> floor ~12 % at MP
4. nothing: internal NN Tb estimate (~10 K)    -> floor ~22 % at MP

Carrier:
- AW (with LK omega from the anchor Tb + the real Tc/Pc) for P >= ~10 kPa,
  IF AND ONLY IF Tc and Pc are REAL.  Estimated criticals: never AW.
- NN otherwise: all cases without real criticals, all P < ~10 kPa, and
  mono-functional alcohols at P < 10 kPa regardless (NN carries the
  thesis eq. 8-6 bowing correction; AW has nothing comparable).

Quality (PFDSim's scale; 1 - 3*MAPE in the calibrated regime, 0.5 = "bad but
best reasonable estimate", 0.3-0.4 = Joback-bad):
- anchor 1 or 2:  MP/HP 0.94 (NN) / 0.95 (AW real crit); LP 0.6; ELP reject
  (or ~0.2 if something must be returned)
- anchor 3:       MP ~0.6; LP ~0.3; ELP reject
- anchor 4:       flat 0.5 (deliberately: litters QUALITY_REPORT, as intended)

HARD RULE: never mix information tiers inside a derived constant.  omega
computed from est-Tb-anchored Psat evaluated at REAL Tc/Pc was the single
worst arm in the whole matrix (94 % MARD, worse than using nothing real).
Compute omega at one consistent tier, always.

### Acentric factor

Value: Lee-Kesler ONLY (`chemicals.acentric.LK_omega`), at a consistent input
tier, for ALL compounds.  (NN omega wins only for polar/associating compounds
with real criticals; taking LK everywhere keeps homologous series smooth and
needs no SMILES.  Decision: simplicity over the last 0.0005 MAD.)

Quality via the polarity classifier (RDKit Lipinski counts):
    score = HBA + 2*HBD;  polar if score > 3 and score / n_heavy > 0.3
- real Tb, Tc, Pc:   q = 0.97 normal | 0.92 polar | 0.70 polar-associating
                      (the classifier's "polar" bucket = score>3&ratio>0.3)
- any estimated input: q = 0.73 normal | ~0.3 (floor) if classifier fires
  (measured: associators with estimated criticals hit MAD ~0.2 in omega)
- omega inputs must be one tier (see HARD RULE above); note that estimated-Tb
  errors CANCEL in omega when Tc/Pc come from the same NN chain (measured:
  tier "nothing real" == tier "real Tb only", both MAD 0.047)

Optional upgrade if SMILES is in hand anyway: route classifier-positive
compounds to `nannoolal_method.acentric_factor` (tier-A MAD 0.0043 vs LK
0.0048, and it fixes LK's worst class: glycols +0.06, nitroaromatics +0.09).
Record |NN - LK| as the uncertainty note either way - it flags CSP-breaking
chemistry at every tier (shared input errors cancel in the difference).

### Hvap (benchmarked 2026-07-10, dz_vap="auto" not yet implemented)

Hvap = dZ * R * T^2 * dlnPs/dT.  Same anchor/carrier decomposition as Psat:
the SLOPE source and the dZ source compose independently.  Benchmark
(`scripts/nannoolal/hvap_benchmark.py` -> `outputs/nannoolal_hvap_benchmark.txt`,
278 Perry compounds vs 2-150 correlations + 182 CRC experimental points;
carboxylic acids excluded - slope methods give apparent Hvap under
dimerization):

- dZ = 1 is the dominant error, all bias: +2 % (Tr 0.6), +7 % (0.7),
  +20 % (0.8); ~+5 % at Tb.  Never ship dZ=1 above Tr ~0.55.
- dZ = PTV (Patel-Teja-Valderrama; needs REAL Tc, Pc, omega AND Vc->Zc):
  with the reference slope, 0.6-1.0 % MAPE to Tr 0.8, bias ~0; 1.3-1.8 %
  vs CRC experiment.  PFDSim's "PTV beats any non-fitted method for pure Z"
  is confirmed emphatically.
- dZ = Tsonopoulos-virial vapor + Rackett liquid: with the NN slope
  (real Tb), 2.2-3.5 % MAPE at Tr 0.5-0.8, ~2.5 % vs CRC at Tb, bias ~0.
- dZ INPUT TIER IS NEARLY IRRELEVANT (5-tier ladder tested: none real /
  +TcPc / +TcPc+omega / +TcPcVc / all real): virial arms agree to <0.3 %
  regardless of estimated-vs-real Tc/Pc, and LK omega == real omega
  exactly.  dZ is a small multiplicative correction (1-dZ ~ 5-20 %), so
  input errors enter at second order - unlike the Psat carrier, where an
  estimated Tc corrupts a first-order anchor.  Only the FORM matters, and
  only near critical: PTV (needs real Vc) beats virial at Tr 0.8
  (2.0 vs 3.3 %); identical at Tr <= 0.7.
- NN slope contributes a flat ~2-2.5 % of its own (Z5-vs-R2 gap): that is
  the floor; better dZ cannot buy it back.
- Hvap is nearly ANCHOR-INSENSITIVE (dlnH/dlnTb ~ 1.29, vs exponential for
  the Psat level): with a fully ESTIMATED Tb (MAD 7.6 K) + est-tier virial
  dZ, 4.5-5.5 % MAPE at Tr 0.5-0.7 (7.4 % at 0.8), bias ~0 -> q ~0.85 even
  at the nothing-real tier.  Same curve, wildly different tier sensitivity
  per property: Psat level exponential / omega ~zero (cancellation) /
  Hvap ~Tb^1.29.

Policy: dhvap rung = NN slope (best anchor per the Psat tree) with
  dZ = virial+Rackett at whatever tier exists  -> q ~0.90 (Tr<=0.7)
  dZ = PTV if real Vc available and Tr > ~0.7  -> q ~0.92 (Tr<=0.8)
  dZ = 1 only below Tr ~0.55 (else forbid)     -> q ~0.90
  (no tier-based quality haircut on dZ inputs - measured irrelevant)
Implementation TODO: `dz_vap="auto"` in `dhvap_J_mol` (virial default,
PTV when real Vc supplied) - all ingredients already importable in-module.

Near-critical extension (curiosity run, same script, Tr to 0.99): with the
REFERENCE slope + PTV(all real), Hvap holds to 1.7 % (Tr 0.85), 3.1 % (0.9),
6.4 % (0.95), then -15 % bias at 0.99 - the classical cubic exponent
(dZ ~ (1-Tr)^0.5 vs real ~(1-Tr)^0.34) finally shows.  Virial form dies as
expected (+87 % at 0.99).  dZ INPUT tier stays irrelevant to the very end:
estimated-criticals arms track real-criticals arms wherever both exist and
merely lose coverage (est Tr >= 1 guard).  NN slope beyond its Tr ~0.8
validity roughly doubles the R2 error (6 % at 0.9, 13 % at 0.95).

### Liquid viscosity (decided 2026-07-13; benchmarks 2026-07-12)

Engines: `nannoolal_method.estimate_viscosity` (Part 4 dBv/Tv; anchored
mode = closed-form Tv from one (T, eta) point) and `hsu_method`
(standalone native engine; misprint fixes, refit alkyne units, domain
gates, Sum(d) Pc guard baked in).  Benchmarks:
`outputs/nannoolal_visc_benchmark.txt`, `outputs/hsu_perry_benchmark.txt`.

Ladder (below any fitted correlation / CoolProp):
1. NN-ANCHORED whenever >= 1 experimental liquid viscosity point exists
   (12/7 % mean/median; wins nearly every class).  ONE exception, and it
   is decisive: phenolic OH goes to Hsu instead (anchoring amplifies the
   Vogel-form shape miss: 52 % measured vs Hsu 20 %).
   Quality: 0.72 in Tr 0.40-0.70; taper x0.85 for Tr 0.70-tr_max.
2. HSU predictive otherwise (14.0/10.8 %; best predictive arm - wins
   alkanes, unsaturated HC, alcohols, phenols, acids).  Quality: as
   computed by hsu_method (0.45-0.75 method factors calibrated to rank
   measured error, x Sum(d)*(1-pc_quality) guard, tr_min/tr_max windows).
3. NN-PREDICTIVE when Hsu rejects the molecule (multifunctionals, diols,
   ring ketones/ethers, chloroethanes, ~40-50 extra compounds): q 0.55
   with real Tb, flat 0.5 with estimated Tb ("bad but best reasonable").
Both GC rungs: additional taper below Tr 0.40 (everyone is 24-40 % there
- near-triple-point viscosity is nobody's territory); recommend x0.7.
Gas side (Yoon-Thodos + Jossi-Stiel) and Lucas pressure correction
unchanged.  Domain refusals live in the ENGINES, not the ladder
(2026-07-12 gates): ortho-chelating aromatics and dimerizing small /
halogenated acids everywhere; heavy perhalomethanes (NN); hydrazines,
substituted CCl3, adjacent polyhalogenated carbons (Hsu - NN covers the
chloroethanes at 2-17 %).

## Justification (details)

- Full 12-arm NN x AW matrix in `outputs/nannoolal_aw_matrix.txt` (Perry 2-8
  + Wagner fits references, ~297+272 compounds, common subsets, bins
  ELP<0.01 kPa / LP<10 / MP<500 / HP>500, Tr<=0.85).
- Anchor tier sets the error floor; carrier only reshapes around it
  (6c vs 6d: identical MP ~12 %, HP halves under AW, LP/ELP better under NN).
- AW's high-P advantage exists only with REAL criticals: with estimated
  criticals AW loses every bin including HP (arm 2b vs 2a).  "An estimated
  Tc isn't an anchor, it's a rumor."
- AW + LK-omega is self-anchoring: LK omega inverts the Pitzer expansion at
  Tb, so AW(LK omega) passes ~through (Tb, 1 atm) whatever the criticals.
  This is why 6d works and why AW+LK degrades gracefully across tiers.
- (Tc, Pc) alone CANNOT determine (Tb, omega): LK and AW are the same Pitzer
  expansion solved in opposite directions; iterating them is the identity map
  (tested; drifts along the null direction).  A third relation is required -
  structure (NN ratio) supplies it in anchor 3.
- NN Psat itself: all 8 Part-3 worked examples reproduce; overall benchmark
  matches the paper's own 6.2 % claim (antoine.txt source); mono-alcohol
  low-P correction recovered from the 2006 thesis (eq. 8-6) after the journal
  misprint was decoded - alcohols LP 43-64 % -> 23-30 %.
- omega tier data: real-input LK MAD 0.005, NN 0.007, mean 0.0048, classifier
  0.0042; estimated tiers all ~0.045 (LK==NN==mean within noise); associators
  with estimated criticals ~0.2 (hence the 0.3 quality floor).
- Classifier thresholds (score>3, ratio>0.3) tuned on 425 compounds; the
  ratio cut separates nitroaromatics (0.375-0.40, belong with NN) from
  diesters (<=0.24, belong with LK).  Evidence base for 0.3-vs-0.5 is four
  compounds - documented as such.
- Psat quality is pressure-dependent because a Tb-anchored slope model has a
  trumpet-shaped error budget: %P error ~ (dHvap/RT^2)*dT_anchor grows toward
  low T.  In T-space the same errors are benign (MP |dT| ~ 0.8 K at anchor
  tier 1), which is why saturation-temperature queries deserve their own
  (higher) quality than pressure queries at LP.
- Viscosity ladder receipts: 240-compound Perry common subset, 12+ classes
  (scripts/nannoolal/visc_benchmark.py).  NN-anchored 11.9/7.1 %; Hsu
  14.0/10.8 %; NN-Tb 21.1/13.9 %; NN-est 23.0/17.3 % (real Tb buys only
  ~2-3 points: dTv/dTb ~ 0.58, the anchor-insensitivity pattern again).
  Rejected carve-out: C7+ acids (Hsu 12.3 vs NN-anc 15.1 - not decisive).
  Tr profile: sweet spot 0.5-0.6 (NN-anc 6.8/3.1 %), symmetric degradation
  at both edges for BOTH methods, so their validity envelopes differ less
  than the papers imply; above Tr 0.7 NN-anc keeps the better median
  (9.5 vs 12.0), Hsu the better mean (18.5 vs 22.2) - no reason to cap
  one earlier than the other.
  Perry's own editorial picks (Hsu for viscosity, NN for Tb/criticals,
  AW+LK for Psat) match the pure-predictive diagonal of this matrix; the
  resolver's edge is conditioning on which inputs are real (anchored mode,
  estimated-criticals tiers) - a handbook cannot recommend a workflow.
  VDI-only block = adverse selection (everything Perry left out); after
  the domain gates it behaves like slightly-harder Perry (Hsu 66->20 %).
