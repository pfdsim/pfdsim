"""Standalone Nannoolal group-contribution estimator for Tb, Tc, Pc and Vc from SMILES.

Implements, from the original papers (both in ``data/source/``):

  [1] Y. Nannoolal, J. Rarey, D. Ramjugernath, W. Cordes,
      "Estimation of pure component properties. Part 1: Estimation of the
      normal boiling point of non-electrolyte organic compounds via group
      contributions and group interactions",
      Fluid Phase Equilibria 226 (2004) 45-63.
  [2] Y. Nannoolal, J. Rarey, D. Ramjugernath,
      "Estimation of pure component properties. Part 2: Estimation of
      critical property data by group contribution",
      Fluid Phase Equilibria 252 (2007) 1-27.
  [3] Y. Nannoolal, J. Rarey, D. Ramjugernath,
      "Estimation of pure component properties. Part 3: Estimation of the
      vapor pressure of non-electrolyte organic compounds via group
      contributions and group interactions",
      Fluid Phase Equilibria 269 (2008) 117-133.

Model equations
---------------
With n = number of non-hydrogen atoms, M = molar mass (g/mol), and
S = sum(N_i * C_i) + GI over first-order groups, second-order corrections
and group interactions (GI, eq. (3)/(4) of the papers):

    Tb [K]        = S_tb / (n**0.6583 + 1.6868) + 84.3395
    Tc [K]        = Tb * (0.6990 + 1 / (0.9889 + S_tc**0.8607))
    Pc [kPa]      = M**(-0.14041) / (0.00939 + S_pc)**2
    Vc [cm3/mol]  = S_vc * n**0.2266 + 86.1539

    GI = (1/n) * sum_i sum_j C(i-j) / (m - 1)      (double sum over ordered
    pairs of distinct interaction-group instances; m = number of such
    instances in the molecule; missing pair parameters contribute zero)

Vapor pressure (eq. (6)/(7) of [3]; anchored at the normal boiling point):

    log10(Ps [atm]) = (4.1012 + dB) * (Trb - 1) / (Trb - 1/8),  Trb = T/Tb
    dB              = sum(N_i * C_i) + GI - 0.176055

dB is a pure structure property (the slope of the vapor pressure curve);
the anchor Tb can be an experimental value, a value back-calculated from
any single (T, Ps) point (Appendix A of [3]), or the internal Tb estimate.
Stated applicability: from about the triple point up to Tr ~ 0.75-0.8.
The same closed form yields the heat of vaporization (Appendix A of [3]):

    dHvap = 56 ln(10) R (4.1012 + dB) Tb dZvap / (Tb/T - 8)^2

Fragmentation follows the standardized group definitions of Table 2 of [2]
(which supersedes Table 1 of [1]): each heavy atom is claimed by exactly one
structural group, with groups tried in the published priority (PR) order.

Implementation notes / documented assumptions
---------------------------------------------
The papers' verbal group definitions are ambiguous in a few corners.  The
choices below were made so that every numeric worked example in both papers
(Tables 17a-d of [1], Tables 42a-d of [2]) is reproduced exactly:

* Halogen sub-groups (19-29, 102): "substituted with ... atoms" is read as
  counting heavy (non-hydrogen) substituents.  F/Cl on aromatic carbons and
  on non-aromatic sp2 carbons are routed to the aromatic/vinyl groups first.
  With this reading the three example compounds of group 23 (PhCF3, CF3CH2OH,
  CF3COOH) fragment to group 21, as required by worked example 42d of [2]
  (CF3 groups next to a carbonyl -> group 21); group 23 then covers carbons
  bearing exactly two other halogens and a hydrogen (e.g. CHF3, R22).
* Alcohol chain-length rule (35 vs 36): "at least five C or Si containing
  chain" counts the C/Si atoms on the longest simple heavy-atom path
  starting at the hydroxyl oxygen; the path may traverse heteroatoms but
  only C/Si count toward the five.  This is fixed by the numeric worked
  examples 15e/15g of [3] (glycol monoacetate and diethanolamine, 4-carbon
  chains -> group 36) together with 42b of [2] (DEGME, 5 carbons across two
  ether oxygens -> group 35), and matches the groups' own short names
  "-OH (<C5)" / "-OH (>C4)".  An earlier revision counted heavy atoms
  (including the hydroxyl O), which contradicted 15e/15g and silently made
  every plain C4 primary alkanol (1-butanol!) group 35.  The single
  counter-signal is ethylene cyanohydrin (3 carbons) sitting in the
  ILLUSTRATIVE examples column of group 35 in all three papers -- read as
  one more inherited misprint; no numeric example ever pins it.
* Group 109 (">N(C=O)", example "methyl thioacetate" in both papers) is
  implemented as the thioester group -C(=O)S-; plain amides are covered by
  groups 48-50.
* Group interaction 171, printed "L AF" in [1], is read as thiol-ester (L-F).
* Group 96 (cyclic sp2 anhydride, "(-C=O-O-C=O-)r") claims its whole ring,
  i.e. also the two sp2 ring carbons; with core-only claims all three of
  maleic/citraconic/phthalic anhydride come out ~+80 K high, while the
  whole-ring reading reproduces the published 7.7 K accuracy of the group.
* Saturated cyclic anhydrides (succinic-type) fit neither published anhydride
  group (76: ~-120 K, 96: ~-60 K); they get the local extension group 219,
  whose Tb contribution was fitted in this work (see its table entry).  A
  warning is attached to results that use it.
* The COOH-COOH group interaction has a published Tb value but no published
  Tc/Pc/Vc values; Tc and Pc were fitted in this work to pulse-heating
  measurements for linear dicarboxylic acids (see the table entry).  A
  warning is attached to results that use them.
* Aromatic nitro group 69 has a published Tb value but no published Tc/Pc/Vc
  values.  Local first-order critical extensions were fitted in this work to
  Perry physical criticals, high-quality EOS-effective records, and PFDSim's
  uncertainty-qualified experimental compilation (see LOCAL_GROUP_EXTENSIONS
  and scripts/nannoolal/refit_group69_aromatic_nitro.py).  Nitro compounds
  bearing a directly aromatic primary amine (group 41, nitroanilines) or
  phenolic hydroxyl (group 37, nitrophenols) are refused for critical
  estimation because those conjugated interactions are too large to absorb
  into a universal first-order value.  Remote/aliphatic NH2 and OH groups are
  not refused without evidence; their missing interactions remain additive
  under the published convention.  Published Tb/Psat/viscosity paths remain
  available in either case.
* Interaction class A (alcohols) is {34, 35, 36} per Table 4 of [2]
  (tertiary alcohols, group 33, are additive); class Q is aromatic nitro
  (group 69) per both papers' tables.
* Groups renumbered/added by [2] are used throughout: 134 is the C=C-C=O
  conjugation correction (Tb value taken from old ID 118 of [1]); 118 is the
  1,2-diketone "do not estimate" flag; 214 (biphenyl bridge carbon) uses
  group 18's Tb value, and group 98 (aromatic secondary amine, new in [2])
  uses group 42's Tb value - both exactly reproduce the Part-1 fragmentation
  for the normal boiling point.
* Silicon: [2] re-engineered the Si groups by the number of C/H neighbors
  (216/215/93/71/70 for 0/1/2/3/4+) plus halogen correction 217; [1] used
  neighbor type (93 if F/Cl, 71 if O, else 70).  Both mappings are applied,
  each for its own property set.
* Second-order corrections with no published value for a property (only 129,
  para pairs, for Tc/Pc/Vc) contribute zero; first-order groups with no
  published value make that property unavailable (None), as stated in [2].

Vapor pressure layer ([3]) judgment calls:

* Fragmentation is shared: Table 1 of [3] reprints the group scheme of
  [1]/[2] with identical IDs and priorities (spot-checked), and all eight
  worked examples of [3] (Tables 15a-15h) reproduce with the existing
  fragmenter unchanged.
* Eq. (9) of [3] (low-pressure "bowing" correction for mono-functional
  alcohols) is hopelessly misprinted in the journal: +0.75407/
  (1+7 exp(201/T-91))-1 can neither vanish nor satisfy Ps(Tb) = 1 atm.  The
  correct form is Eq. (8-6) of Nannoolal's PhD thesis (Univ. KwaZulu-Natal,
  2006, data/reference/thermodynamic-models/2006-nannoolal-group-contribution-property-estimation-thesis.pdf, p. 193):
      f(T) = a [ 2 / (1 + exp((201/(T-91))^7)) - 1 ],   a = 0.37704
  added to log10(Ps/atm) for mono-functional alcohols only.  The journal's
  0.75407 is 2a, its "7 exp" is the 7th POWER on the argument, and
  "201/T-91" lost the parentheses of 201/(T-91).  The correction is a
  function of ABSOLUTE temperature (H-bonding competes with RT, not Tr):
  ~0 above ~450 K, saturating at -0.377 (factor 2.38 in pressure) below
  ~250 K.  It is applied here exactly as in the thesis, including the small
  nonzero value it takes at Tb for light alcohols (methanol -10 %; the
  published alcohol dB values were regressed jointly with it).  psat/
  temperature/Tb-from-point stay mutually consistent: inversions switch to
  bisection when the correction is active.
* Worked example 15g (diethanolamine) is internally inconsistent: its
  printed dB (1.7493490) is obtained by ADDING the group-interaction term
  although the printed interaction sum is negative (-0.9531006).  Applying
  eq. (8) with its signs -- exactly as examples 15b/15e/15f do -- gives
  dB = 1.6131919, which also reproduces the experimental pressure better
  (0.404 vs. 0.354 kPa calculated, 0.410 kPa experimental).  This module
  follows the equation, not the example.
* Group 91 and the epoxide-epoxide interaction (187) are flagged
  "questionable" by [3] itself; using them appends a warning.
* Correction 129 (para pairs) has no dB value in Table 5 of [3] and
  contributes zero; correction 217 (Si-halogen) DOES have a dB value and is
  applied (unlike for Tb, where it does not exist).
* dB for silicon uses the Part-2 style C/H-neighbor-count group mapping
  (216/215/93/71/70), matching the Si group set priced in Table 4 of [3].
* The local extension group 219 (saturated cyclic anhydride) has no dB of
  its own; it borrows the dB of group 96 (cyclic sp2 anhydride, the nearest
  published relative), with a warning.
* Group 110 (phosphite, absent from Table 1 of [3] but priced in its
  Table 4 with 4 components -- the same ID-elimination situation as in [2])
  keeps the phosphite assignment of this module.

Only RDKit is required.  Public API:

    >>> from nannoolal_method import estimate, estimate_psat
    >>> r = estimate("CCO")                  # Tb estimated internally
    >>> r = estimate("CC(=O)C(F)(F)F", tb=245.9)   # experimental Tb for Tc
    >>> r.tb_K, r.tc_K, r.pc_kPa, r.vc_cm3_mol
    >>> p = estimate_psat("OCCO", tb=470.5)  # anchor at experimental Tb
    >>> p.psat_kPa(410.65), p.temperature_K(13.05), p.dhvap_J_mol(410.65)
    >>> p = estimate_psat("OCCO", psat_point=(410.65, 13.05))  # anchor at
    ...                                      # any single (T [K], Ps [kPa])

or from the command line:

    python nannoolal_method.py "CC(C)(C)c1ccccc1" [--tb 442.3] [-v]
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import Descriptors


class NannoolalError(ValueError):
    """Base error for the Nannoolal method."""


class FragmentationError(NannoolalError):
    """The molecule contains atoms not covered by the published group scheme."""


# ---------------------------------------------------------------------------
# Model constants (eq. (1) of [1]; Table 16 of [2])
# ---------------------------------------------------------------------------

TB_A, TB_B, TB_C = 0.6583, 1.6868, 84.3395            # Tb = S/(n^a + b) + c
TC_A, TC_B, TC_C = 0.9889, 0.6990, 0.8607             # Tc = Tb (b + 1/(a + S^c))
PC_A, PC_B = 0.00939, -0.14041                        # Pc = M^b / (a + S)^2
VC_A, VC_B = -0.2266, 86.1539                         # Vc = S / n^a + b
AROMATIC_NITRO_GROUP = 69

_TC_SCALE = 1.0e-3   # Table 7-9 of [2] list contributions * 10^3
_PC_SCALE = 1.0e-4   # Tables 10-12 of [2] list contributions * 10^4

# ---------------------------------------------------------------------------
# First-order group contributions.
# id: (tb [K], tc [*1e-3, dimensionless], pc [*1e-4], vc [cm3/mol])
# Sources: Table 3 of [1] (Tb); Tables 7, 10, 13 of [2].  None = no value
# published for that property.
# ---------------------------------------------------------------------------

GROUP_CONTRIBUTIONS: dict[int, tuple[float | None, ...]] = {
    #     Tb          Tc(e-3)    Pc(e-4)    Vc
    1:   (177.3066,   41.8682,   8.1620,    28.7855),
    2:   (251.8338,   33.1371,   5.5262,    28.8811),
    3:   (157.9527,   -1.0710,   4.1660,    26.7237),
    4:   (239.4531,   40.0977,   5.2623,    32.0493),
    5:   (240.6785,   30.2069,   2.3009,    32.1108),
    6:   (249.5809,   -3.8778,   -2.9925,   28.0534),
    7:   (266.8769,   52.8003,   3.4310,    33.7577),
    8:   (201.0115,   9.4422,    2.3665,    28.8792),
    9:   (239.4957,   21.2898,   3.4027,    24.8517),
    10:  (222.1163,   26.3513,   3.6162,    30.9323),
    11:  (209.9749,   -17.0459,  -5.1299,   5.9550),
    12:  (250.9584,   51.7974,   4.1421,    29.5901),
    13:  (291.2291,   18.9549,   0.8765,    20.2325),
    14:  (244.3581,   -29.1568,  -0.1320,   10.5669),
    15:  (235.3462,   16.1154,   2.1064,    19.4020),
    16:  (315.4128,   68.2045,   4.1826,    25.0434),
    17:  (348.2779,   68.1923,   3.5500,    5.6704),
    18:  (367.9649,   29.8039,   1.0997,    16.4118),
    19:  (106.5492,   15.6068,   0.7328,    -5.0331),
    20:  (49.2701,    11.0757,   4.3757,    1.5646),
    21:  (53.1871,    18.1302,   3.4933,    3.3646),
    22:  (78.7578,    19.1772,   2.6558,    1.0897),
    23:  (103.5672,   20.8519,   1.6547,    1.1084),
    24:  (-19.5575,   -24.0220,  0.5236,    19.3190),
    25:  (330.9117,   -1.3329,   -2.2611,   22.0457),
    26:  (287.1863,   2.6113,    -1.4992,   23.9279),
    27:  (267.4170,   15.5010,   0.4883,    26.2582),
    28:  (205.7363,   -16.1905,  -0.9280,   36.7624),
    29:  (292.5816,   60.1907,   11.8687,   34.4110),
    30:  (419.4959,   5.2621,    -4.3170,   36.0223),
    31:  (377.6775,   -21.5199,  -2.2409,   30.7004),
    32:  (556.3944,   -8.6881,   -4.7841,   48.2989),
    33:  (349.9409,   84.8567,   -7.4244,   10.6790),
    34:  (390.2446,   79.3047,   -4.4735,   5.6645),
    35:  (443.8712,   49.5968,   -1.8153,   2.0869),
    36:  (488.0819,   130.1320,  -6.8991,   3.7778),
    37:  (361.4775,   14.0159,   -12.1664,  25.6584),
    38:  (146.4836,   12.5082,   2.0592,    11.6284),
    39:  (820.7118,   41.3490,   0.1759,    46.7680),
    40:  (321.1759,   18.3404,   -4.4164,   13.2571),
    41:  (441.4388,   -50.6419,  -9.0065,   73.7444),
    42:  (223.0992,   17.1780,   -0.4086,   20.5722),
    43:  (126.2952,   -0.5820,   2.3625,    6.0178),
    44:  (1080.3139,  199.9042,  3.9873,    40.3909),
    45:  (636.2020,   75.7089,   4.3592,    42.6733),
    46:  (642.0427,   58.0782,   1.0266,    36.1286),
    47:  (1142.6119,  109.1930,  0.4329,    None),
    48:  (1052.6072,  102.1024,  0.5172,    64.3506),
    49:  (1364.5333,  None,      None,      None),
    50:  (1487.4109,  None,      None,      None),
    51:  (618.9782,   56.1572,   0.1190,    30.9229),
    52:  (553.8090,   44.2000,   -2.3615,   25.5034),
    53:  (434.0811,   -7.1070,   -9.4154,   34.7699),
    54:  (461.5784,   0.5887,    -8.2595,   38.0185),
    55:  (864.5074,   None,      None,      None),
    56:  (304.3321,   -7.7181,   -4.9259,   20.3127),
    57:  (719.2462,   117.1330,  5.1666,    43.7983),
    58:  (475.7958,   45.1531,   7.1581,    None),
    59:  (586.1413,   None,      None,      None),
    60:  (500.2434,   67.9821,   -6.2791,   51.0710),
    61:  (412.6276,   45.4406,   9.6413,    48.1957),
    62:  (475.9623,   56.4059,   3.4731,    34.1240),
    63:  (512.2893,   -19.9737,  -2.2718,   40.9263),
    64:  (422.2307,   36.0883,   2.4489,    29.8612),
    65:  (37.1936,    10.4146,   -0.5403,   4.7476),
    66:  (453.3397,   18.9903,   8.3052,    -25.3680),
    67:  (306.7139,   10.9495,   -4.7101,   23.6094),
    68:  (866.5843,   82.6239,   -5.0929,   34.8472),
    69:  (821.4141,   None,      None,      None),
    70:  (282.0181,   25.4209,   5.7270,    75.7193),
    71:  (207.9312,   72.5587,   2.7602,    69.5645),
    72:  (920.3617,   None,      None,      None),
    73:  (1153.1344,  None,      None,      None),
    74:  (494.2668,   None,      None,      None),
    75:  (1041.0851,  None,      None,      None),
    76:  (1251.2675,  164.3355,  4.0458,    None),
    77:  (778.9151,   None,      None,      None),
    78:  (540.0895,   157.3401,  12.6786,   None),
    79:  (879.7062,   97.2830,   0.2822,    52.8789),
    80:  (660.4645,   153.7225,  None,      27.1026),
    81:  (1018.4865,  None,      None,      None),
    82:  (1559.9840,  90.9726,   -23.9221,  68.0701),
    83:  (510.4223,   62.3642,   0.7043,    None),
    84:  (1149.9670,  None,      None,      None),
    85:  (1209.2972,  None,      None,      None),
    86:  (347.7717,   None,      None,      None),
    87:  (664.0903,   53.6350,   12.6128,   None),
    88:  (957.6388,   24.7302,   -10.2451,  64.4616),
    89:  (928.9954,   None,      None,      None),
    90:  (560.1024,   38.4681,   -4.0133,   20.0440),
    91:  (229.2288,   None,      None,      None),
    92:  (606.1797,   63.6504,   -5.0403,   28.7127),
    93:  (215.3416,   34.2058,   3.2023,    55.3822),
    94:  (273.1755,   None,      None,      None),
    95:  (1218.1878,  None,      None,      None),
    96:  (2082.3288,  None,      None,      None),
    97:  (201.3224,   27.3441,   -4.3834,   29.3068),
    # 98 (aromatic secondary amine) is new in [2]; its Tb value falls back to
    # group 42, which is exactly how [1] fragmented these amines.
    98:  (223.0992,   34.5325,   -13.6078,  None),
    99:  (886.7613,   None,      None,      None),
    100: (1045.0343,  None,      None,      None),
    101: (-109.6269,  None,      None,      None),
    102: (111.0590,   1.3231,    3.3971,    1.3597),
    103: (1573.3769,  764.9595,  58.9190,   None),
    104: (1483.1289,  None,      None,      None),
    105: (1506.8136,  None,      None,      None),
    106: (484.6371,   None,      None,      None),
    107: (1379.4485,  None,      None,      None),
    108: (659.7336,   None,      None,      None),
    109: (492.0707,   None,      None,      None),
    # 110 is never defined in either paper, but Table 10 of [1] introduces a
    # "phosphite P(O-)3" group without an ID and 110 is the only unused slot
    # consistent with the published Tc/Pc values (2 components each).
    110: (None,       23.4707,   0.1812,    None),
    111: (971.0365,   None,      None,      None),
    113: (428.8911,   None,      None,      None),
    115: (612.9506,   36.0361,   -5.1116,   16.2688),
    116: (562.1791,   None,      None,      None),
    117: (761.6006,   None,      None,      None),
    # 214 (biphenyl bridge carbon) is new in [2]; its Tb value falls back to
    # group 18, which is exactly how [1] fragmented biphenyls.
    214: (367.9649,   48.1680,   1.5574,    16.3122),
    215: (None,       0.2842,    3.8751,    37.0423),
    216: (None,       -0.6536,   4.4882,    55.7432),
    # 219 is NOT part of the published method: local extension for cyclic
    # anhydrides without sp2/aromatic ring closure (succinic-type), which fall
    # in neither published anhydride group's training set (76 gives ~-120 K,
    # 96 ~-60 K).  Tb fitted 2026-07-09 to succinic (534.1 K), glutaric
    # (560.1 K) and methylsuccinic (532.2 K) anhydride from Perry Table 2-10;
    # fit residuals -0.4/-3.1/+3.5 K.  Diglycolic anhydride is assigned to
    # this group but was excluded from the fit (extra ring ether oxygen;
    # +45 K vs its corrected Tb of 513.6 K).  No critical-property data was
    # available, so Tc/Pc/Vc remain unavailable for this group.
    219: (1936.7436,  None,      None,      None),
}

# 1,2-diketone: "do not fragment / do not estimate" in [2].
DO_NOT_ESTIMATE_GROUP = 118

# ---------------------------------------------------------------------------
# Second-order corrections.  Sources: Table 4 of [1] (Tb, with old ID 118 ->
# new ID 134 per [2]); Tables 8, 11, 14 of [2].
# ---------------------------------------------------------------------------

CORRECTION_CONTRIBUTIONS: dict[int, tuple[float | None, ...]] = {
    #     Tb          Tc(e-3)    Pc(e-4)    Vc
    119: (-82.2328,   32.1829,   7.3149,    -3.8033),   # (C=O)-C(hal>=2)
    120: (-247.8893,  11.4437,   4.1439,    27.5326),   # (C=O)-(C(hal>=2))2
    121: (-20.3996,   -1.3023,   0.4387,    1.5807),    # C-[F,Cl]3
    122: (15.4720,    -34.3037,  -4.2678,   -2.6235),   # (C)2-C-[F,Cl]2
    123: (-172.4201,  -1.3798,   4.8944,    -5.3091),   # no hydrogen
    124: (-99.8035,   -2.7180,   2.8103,    -6.1909),   # one hydrogen
    125: (-62.3740,   11.3251,   -0.3035,   3.2219),    # 3/4-membered ring
    126: (-40.0058,   -4.7516,   0.0930,    -6.3900),   # 5-membered ring
    127: (-27.2705,   1.2823,    0.7061,    -3.5964),   # ortho pair(s)
    128: (-3.5075,    6.7099,    -0.7246,   1.5196),    # meta pair(s)
    129: (16.1061,    None,      None,      None),      # para pair(s)
    130: (25.8348,    -33.8201,  -8.8457,   -4.6483),   # (C=)(C)C-CC3
    131: (35.8330,    -18.4815,  -2.2542,   -5.0563),   # C2C-CC2
    132: (51.9098,    -23.6024,  -3.2460,   -6.3267),   # C3C-CC2
    133: (111.8372,   -24.5802,  -5.3113,   4.9392),    # C3C-CC3
    134: (40.4205,    -35.6113,  1.0934,    2.8889),    # C=C-C=O conjugation
    217: (None,       62.0286,   8.6126,    19.4348),   # Si attached to halogen
}

# ---------------------------------------------------------------------------
# Group interactions.  Classes per Table 9 of [1] / Table 4 of [2]:
#   A alkanol(34,35,36)  B phenol(37)      C COOH(44)      D ether(38)
#   E epoxide(39)        F ester(45,46,47) G ketone(51,92) H aldehyde(52,90)
#   I aromatic O(65)     J thioether(54)   K aromatic S(56) L thiol(53)
#   M prim.amine(40,41)  N sec.amine(42,97) O isocyanate(80) P nitrile(57)
#   Q aromatic nitro(69) R aromatic N r5(66) S aromatic N r6(67)
# ---------------------------------------------------------------------------

INTERACTION_CLASS: dict[int, str] = {
    34: "A", 35: "A", 36: "A",
    37: "B",
    44: "C",
    38: "D",
    39: "E",
    45: "F", 46: "F", 47: "F",
    51: "G", 92: "G",
    52: "H", 90: "H",
    65: "I",
    54: "J",
    56: "K",
    53: "L",
    40: "M", 41: "M",
    42: "N", 97: "N",
    80: "O",
    57: "P",
    69: "Q",
    66: "R",
    67: "S",
}

def _pairs(table: dict[str, tuple[float | None, ...]]) -> dict[frozenset, tuple]:
    return {frozenset(key): val for key, val in table.items()}

# key "XY": (tb [K], tc [*1e-3], pc [*1e-4], vc [cm3/mol]).
# Tb from Table 5 of [1]; Tc/Pc/Vc from Tables 9, 12, 15 of [2].
INTERACTION_CONTRIBUTIONS: dict[frozenset, tuple[float | None, ...]] = _pairs({
    #      Tb           Tc(e-3)     Pc(e-4)     Vc
    "AA": (291.7985,   -434.8568,  -5.6023,    None),
    "AM": (314.6126,   120.9166,   69.8200,    None),
    "AN": (286.9698,   -30.4354,   6.1331,     -8.0423),
    "AL": (38.6974,    None,       None,       None),
    "AC": (146.7286,   None,       None,       None),
    "AD": (135.3991,   -146.7881,  7.3373,     19.7707),
    "AE": (226.4980,   None,       None,       None),
    "AF": (211.6814,   None,       None,       None),
    "AG": (46.3754,    None,       None,       None),
    "AJ": (-74.0193,   None,       None,       None),
    "AP": (306.3979,   None,       None,       None),
    "AI": (435.0923,   None,       None,       None),
    "AS": (1334.6747,  None,       None,       None),
    "BB": (288.6155,   144.4697,   57.8350,    97.5425),
    "BM": (797.4327,   None,       None,       None),
    "BC": (-1477.9671, None,       None,       None),
    "BD": (130.3742,   None,       None,       None),
    "BF": (-1184.9784, None,       None,       None),
    "BG": (None,       None,       None,       None),   # listed, no value
    "BH": (43.9722,    None,       None,       None),
    "BQ": (-1048.1236, None,       None,       None),
    "BS": (-614.3624,  None,       None,       None),
    "MM": (174.0258,   -60.9217,   -0.6754,    None),
    "MN": (510.3473,   None,       None,       None),
    "MD": (124.3549,   -738.0515,  -125.5983,  None),
    "MF": (182.6291,   None,       None,       None),
    "MJ": (-562.3061,  None,       None,       None),
    "MQ": (663.8009,   None,       None,       None),
    "MI": (395.4093,   None,       None,       None),
    "MS": (27.2735,    None,       None,       None),
    "NN": (239.8076,   -49.7641,   22.1871,    -57.1233),
    "ND": (101.8475,   None,       None,       None),
    "NF": (317.0200,   None,       None,       None),
    "NG": (-215.3532,  None,       None,       None),
    "NS": (758.9855,   None,       None,       None),
    "LL": (217.6360,   None,       None,       None),
    # Printed "L AF" in Table 5 of [1]; read as thiol-ester (see module doc).
    "LF": (501.2778,   None,       None,       None),
    # COOH-COOH: only the Tb value is published.  The Tc/Pc values are NOT
    # part of the published method: local extension fitted 2026-07-09 to the
    # pulse-heating measurements for HOOC-(CH2)n-COOH (n = 3..12, succinic
    # excluded per that paper's own convention): Nikitin, Popov,
    # Bogatishcheva, Yatluk, J. Chem. Eng. Data 49 (2004) 1515-1520
    # (data/reference/pure-component-properties/2004-nikitin-dicarboxylic-acid-critical-properties.pdf;
    # Tables 2-3 also photographed in data/reference/pure-component-properties/2004-nikitin-dicarboxylic-acid-critical-property-tables.jpeg).  Fit residuals: Tc AAD 10.0 K (data
    # uncertainty +-13 K; the 1/n interaction form cannot fully reproduce the
    # nearly flat Tc series), Pc AAD 2.7 %.
    # NOTE: because diacids have no true experimental Tb (they decompose),
    # the Tc constant is conditioned on the Tb convention used in the fit:
    # here the NIST-Antoine-extrapolated Tb column (adipic 610.5 K etc.),
    # matching the Tb style of this repo's databases (e.g. Perry 2-10).
    # Refitting against Clapeyron/vacuum-distillation Tbs gives -463.8e-3;
    # against internally estimated Tb, -1152.0e-3 (AAD 1.8 K).  The Pc value
    # is Tb-independent.
    "CC": (117.2044,   -967.4511,  -47.7547,   None),
    "CD": (612.8821,   None,       None,       None),
    "CF": (-183.2986,  None,       None,       None),
    "CG": (-55.9871,   None,       None,       None),
    "OO": (-356.5017,  -1866.0970, None,       44.1062),
    "OQ": (-263.0807,  None,       None,       None),
    "DD": (91.4997,    162.6878,   2.6751,     -23.6366),
    "DE": (178.7845,   707.4116,   88.8752,    -329.5074),
    "DF": (322.5671,   128.2740,   -1.0295,    -55.5112),
    "DG": (15.6980,    None,       None,       None),
    "DH": (17.0400,    None,       None,       None),
    "DJ": (394.5505,   -654.1363,  25.8246,    -37.2468),
    "DQ": (963.6518,   None,       None,       None),
    "DP": (293.5974,   741.8565,   None,       None),
    "DI": (329.0050,   None,       None,       None),
    "EE": (1006.3880,  None,       None,       None),
    "EH": (163.5475,   None,       None,       None),
    "FF": (431.0990,   366.2663,   0.5195,     -74.8680),
    "FG": (22.5208,    None,       None,       None),
    "FQ": (-205.6165,  None,       None,       None),
    "FP": (517.0677,   None,       None,       None),
    "FI": (707.9404,   None,       None,       None),
    "GG": (-303.9653,  1605.5640,  -78.2743,   -413.3976),
    "GH": (-391.3690,  None,       None,       None),
    "GQ": (-3628.9026, None,       None,       None),
    "GK": (381.0107,   None,       None,       None),
    "GP": (-574.2230,  None,       None,       None),
    "GI": (176.5481,   None,       None,       None),
    "GS": (124.1943,   None,       None,       None),
    "HH": (582.1763,   None,       None,       None),
    "HQ": (140.9644,   None,       None,       None),
    "HK": (397.5750,   None,       None,       None),
    "HI": (674.6858,   None,       None,       None),
    "JJ": (-11.9406,   -861.1528,  43.9001,    -403.1196),
    "QQ": (65.1432,    None,       None,       None),
    "KP": (-101.2319,  None,       None,       None),
    "KR": (-348.7400,  131.7924,   -19.7033,   164.2930),
    "PS": (-370.9729,  None,       None,       None),
    "IR": (-888.6123,  24.0243,    -35.1998,   217.9243),
    "RR": (None,       None,       None,       None),   # listed, no value
    "SS": (-271.9449,  -32.3208,   12.5371,    -26.4556),
})

# COOH + NH2 -> zwitterion; interaction 218 of [2]: do not estimate.
_ZWITTERION_PAIR = frozenset("CM")

# ---------------------------------------------------------------------------
# Vapor pressure curve slope (dB) contributions of [3].
#   log10(Ps/atm) = (4.1012 + dB) (Trb - 1)/(Trb - 1/8),  Trb = T/Tb  (eq. 6)
#   dB = sum(N_i C_i) + GI - 0.176055                                (eq. 7)
# Group values from Table 4, second-order from Table 5, interactions from
# Table 6 (all printed *10^3).  [3] shares the fragmentation scheme of
# [1]/[2], so the same fragmenter and interaction classes are reused; the
# per-pair interaction IDs of Table 6 (135-218) are mapped to class-letter
# pairs below.  None = no published dB (vapor pressure not estimable for
# first-order groups; contributes zero for corrections, per [3]).
# ---------------------------------------------------------------------------

PSAT_B0 = 4.1012            # eq. (6) of [3]
PSAT_DB_OFFSET = 0.176055   # eq. (7) of [3]
_DB_SCALE = 1.0e-3          # Tables 4-6 of [3] list contributions * 10^3
_P_ATM_KPA = 101.325
_R_GAS = 8.314462618        # J/(mol K)

PSAT_DB_GROUPS: dict[int, float | None] = {
    1: 13.3063,     2: 91.8000,     3: 50.1939,     4: 54.6564,
    5: 45.7437,     6: -31.7531,    7: 37.8485,     8: 96.1386,
    9: 22.2573,     10: 32.8162,    11: 4.8500,     12: 23.6411,
    13: 49.8237,    14: -3.6950,    15: 32.7177,    16: 69.8796,
    17: 41.5534,    18: 43.7191,    19: 79.5429,    20: 51.2880,
    21: 42.0887,    22: 56.9998,    23: 142.1060,   24: 45.9652,
    25: 93.6679,    26: 67.8082,    27: 55.9304,    28: 46.0435,
    29: 84.9162,    30: 104.9291,   31: -40.1837,   32: 134.3501,
    33: 719.3666,   34: 758.4218,   35: 700.7226,   36: 756.0824,
    37: 441.8437,   38: 108.4964,   39: 286.9731,   40: 251.9212,
    41: 361.7760,   42: 193.7667,   43: -102.7252,  44: 1074.1000,
    45: 355.7381,   46: 350.5184,   47: 292.8046,   48: 269.2471,
    49: 736.9540,   50: 1216.0700,  51: 255.8480,   52: 252.9059,
    53: 123.2143,   54: 127.3380,   55: 222.2789,   56: 20.1604,
    57: 226.1873,   58: 86.4601,    59: 224.2238,   60: 134.9382,
    61: 34.2541,    62: 97.4210,    63: 206.6665,   64: 128.0247,
    65: 48.8839,    66: 375.0486,   67: 126.3340,   68: 375.8217,
    69: 238.2066,   70: 2.8992,     71: 9.3624,     72: 603.5347,
    73: 662.0582,   74: 510.9666,   75: 1317.4360,  76: 681.3525,
    77: 564.1116,   78: 391.3697,   79: 318.2350,   80: 435.8446,
    81: 218.6012,   82: 381.5945,   83: 80.2735,    84: 231.3919,
    85: 186.9204,   86: 48.5026,    87: 168.3007,   88: 108.5277,
    89: 213.7165,   90: 183.1130,   91: 1178.1950,  # questionable per [3]
    92: 158.3258,   93: -47.3420,   94: 186.7950,   95: 392.2006,
    96: 595.1778,   97: 268.7081,   98: 183.3467,   99: 612.9546,
    100: 258.9924,  101: -316.4392, 102: 64.0566,   103: 660.2247,
    104: 554.7941,  105: 420.7591,  107: -237.2124, 110: 37.0590,
    111: 319.4879,  113: 118.8412,  115: 305.1341,  116: 191.5058,
    117: 423.5251,
    # 106, 108, 109, 114, 118: no dB published (118: do not estimate).
    214: 86.9450,   215: -66.5670,  216: -81.1543,
}

PSAT_DB_CORRECTIONS: dict[int, float | None] = {
    119: 34.3545,   120: 2.5030,    121: -83.3326,  122: -64.4854,
    123: -125.9208, 124: -47.2962,  125: 33.9765,   126: -7.0982,
    127: -45.0531,  128: -3.2036,
    129: None,      # para pairs: no dB value in Table 5 of [3]
    130: -20.6706,  131: -36.3170,  132: -1.1994,   133: 123.7433,
    134: -15.9694,
    217: 36.7574,   # Si-halogen: exists for dB (unlike Tb)
}

# Table 6 of [3]; per-pair IDs 135-218 in the trailing comments.
PSAT_DB_INTERACTIONS: dict[frozenset, float | None] = _pairs({
    "AA": -561.5153,    # 135 OH-OH
    "AM": 1067.6660,    # 136 OH-NH2
    "AN": 42.4825,      # 137 OH-NH
    "AD": -799.5332,    # 140 OH-etherO
    "AE": -618.2760,    # 141 OH-epoxide
    "AF": -1797.6930,   # 142 OH-ester
    "AG": -1181.5990,   # 143 OH-ketone
    "AP": 1431.2430,    # 145 OH-CN
    "AI": 1132.0400,    # 146 OH-aromatic O
    "BB": -97.6205,     # 148 OH(a)-OH(a)
    "BD": -751.6676,    # 151 OH(a)-etherO
    "BF": 548.4352,     # 152 OH(a)-ester
    "MM": 1085.8320,    # 157 NH2-NH2
    "MN": -206.7811,    # 158 NH2-NH
    "MD": -198.2791,    # 159 NH2-etherO
    "MF": -1676.4770,   # 160 NH2-ester
    "MQ": 1659.0340,    # 162 NH2-nitro
    "NN": -307.1018,    # 165 NH-NH
    "ND": 65.4421,      # 166 NH-etherO
    "LL": 240.3037,     # 170 SH-SH
    "CC": -2601.7090,   # 172 COOH-COOH (published here, unlike Tc/Pc!)
    "CG": -787.8563,    # 175 COOH-ketone
    "OO": -3929.1300,   # 176 OCN-OCN
    "DD": 144.6074,     # 178 etherO-etherO
    "DE": 1118.9580,    # 179 etherO-epoxide
    "DF": -225.7802,    # 180 etherO-ester
    "DG": 1981.2980,    # 181 etherO-ketone
    "DH": 362.7540,     # 182 etherO-aldehyde
    "DJ": -1425.0170,   # 183 etherO-thioether
    "DP": 743.3353,     # 185 etherO-CN
    "EE": -3748.8180,   # 187 epoxide-epoxide: questionable per [3]
    "FF": 920.3138,     # 189 ester-ester
    "FG": 1594.1640,    # 190 ester-ketone
    "FP": 108.1305,     # 192 ester-CN
    "FI": 1590.3210,    # 193 ester-aromatic O
    "GG": -1270.0830,   # 194 ketone-ketone
    "HH": 946.7309,     # 201 aldehyde-aldehyde
    "HI": 705.3049,     # 204 aldehyde-aromatic O
    "JJ": 838.3372,     # 205 thioether-thioether
    "QQ": -1501.3550,   # 206 nitro-nitro
    "KR": 675.0414,     # 208 aromatic S-aromatic N r5
    "PS": 994.4996,     # 209 CN-aromatic N r6
    "IR": 135.5896,     # 210 aromatic O-aromatic N r5
    "SS": -29.6785,     # 212 AN6-AN6
    # 218 COOH-NH2: do not estimate (zwitterion; flagged at fragmentation)
})

# ---------------------------------------------------------------------------
# Saturated liquid viscosity of [4] (Part 4).
#   ln(eta / 1.3 cP) = -dBv (T - Tv)/(T - Tv/16)                      (eq. 6)
#   dBv = (sum(N_i C_i) + GI) / (n^-2.5635 + 0.0685) + 3.7777         (eq. 7)
#   Tv  = 21.8444 Tb^0.5 + (sum(N_i C_i) + GI)^0.9315
#                          / (n^0.6577 + 4.9259) - 231.1361           (eq. 8)
# Tv is the temperature where eta = 1.3 cP; n = heavy-atom count.  The GI
# prefactor f is 1 for dBv and 2 for Tv (eq. 10), with the ordered-pair
# double count -- all conventions pinned by worked examples 28a-28c of [4].
# Same fragmentation scheme and interaction classes as [1]-[3].  Group
# values from Tables 5/8, second-order from Tables 6/9, interactions from
# Tables 7/10 (dBv tables printed *10^3; Tv tables plain).  All 278 values
# cross-checked against the ch. 9 tables of the 2006 thesis: 100 % digit
# agreement (the sole thesis mismatch, group 81 dBv, is a text-extraction
# artifact: "^6.5613" for "-46.5613").  Validity: ~triple point to
# Tr 0.75-0.8; larger deviations for high rotational symmetry (cyclohexane),
# first members of series, and small carboxylic acids (dimerization).
# ---------------------------------------------------------------------------

VISC_ETA_REF_CP = 1.3       # eq. (6) reference viscosity [cP = mPa s]
VISC_S = 16.0               # eq. (5): C ~ Tv/s with s = 16
VISC_DBV_A = -2.5635        # eq. (7) constants
VISC_DBV_B = 0.0685
VISC_DBV_C = 3.7777
VISC_TV_A = 21.8444         # eq. (8) constants [K^0.5]
VISC_TV_EXP = 0.9315
VISC_TV_C = 0.6577
VISC_TV_D = 4.9259
VISC_TV_E = 231.1361        # [K]

VISC_DBV_GROUPS: dict[int, float | None] = {
    1: 13.9133,     2: 11.7002,     3: -11.0660,    4: 2.1727,
    5: 4.5878,      6: 37.0296,     7: 21.3473,     8: 5.9452,
    9: 10.8799,     10: -7.2202,    11: 142.1976,   12: 61.0811,
    13: 28.7351,    14: -12.3456,   15: -2.7840,    16: 45.9403,
    17: 80.5124,    18: 37.4124,    19: 5.1640,     21: 2.8323,
    22: 0.7129,     23: -36.3189,   24: -61.9434,   25: 4.7579,
    26: 5.8228,     27: 4.6555,     28: -67.3989,   29: -9.6209,
    30: 0.5164,     31: -18.1984,   32: -17.3110,   33: 336.8834,
    34: 365.8067,   35: 249.0118,   36: 218.8000,   37: 160.8315,
    38: -35.3055,   39: 85.3693,    40: 58.9131,    41: 44.0698,
    42: 13.6479,    43: -58.7354,   44: 54.7891,    45: 17.6757,
    46: 0.4267,     47: 47.6109,    48: 12.6717,    49: 129.8293,
    50: 202.2864,   51: 24.2524,    52: 18.4961,    53: 30.5022,
    54: -0.0276,    55: -13.4614,   56: 18.5507,    57: 23.1459,
    58: 9.5809,     59: 152.2693,   60: 18.5983,    61: 21.4560,
    62: 19.7836,    63: -165.0071,  64: 13.3585,    65: 42.7958,
    66: 151.9493,   67: 52.5900,    68: -34.3948,   69: -6.5626,
    70: -25.5950,   71: -28.3943,   72: -30.6156,   73: 45.9972,
    74: -7.3298,    75: 369.8367,   76: 16.3525,    77: -2.6553,
    78: -61.2368,   79: -7.5067,    80: 4.1408,     81: -46.5613,
    82: 122.6902,   86: -70.9713,   88: 29.0985,    89: 125.0861,
    90: -14.2823,   92: -8.5352,    93: 9.9037,     96: 102.0816,
    97: 74.0520,    98: 43.6079,    100: 54.4769,   102: 5.7765,
    103: 95.6531,   104: 56.9133,   105: 64.7133,   107: 22.9969,
    108: 23.2473,   110: -45.7263,
    # 20, 83-85, 87, 91, 94-95, 99, 101, 106, 109, 111-117: no dBv
    # published (118: do not estimate).
    214: 37.5669,   215: 64.6600,   216: 68.4952,
}

VISC_DBV_CORRECTIONS: dict[int, float | None] = {
    119: 0.3041,    121: -6.1420,   122: -26.4635,  123: -14.9636,
    124: -25.9017,  125: -57.3789,  126: -21.2204,  127: -20.1917,
    128: -34.5860,  130: -110.7391, 131: 2.4859,    132: -59.3670,
    134: 13.1413,
    217: -76.1631,
}

# Table 7 of [4]; Table 10 prints the etherO-nitro pair as ID 183 where
# Table 7 prints 184 -- an ID-label misprint only, the class pair is what
# matters and both are mapped to "DQ" here.
VISC_DBV_INTERACTIONS: dict[frozenset, float | None] = _pairs({
    "AA": -112.4939,    # 135 OH-OH
    "AM": 1031.5920,    # 136 OH-NH2
    "AN": 853.2318,     # 137 OH-NH
    "AD": -423.9834,    # 140 OH-etherO
    "AP": -683.0189,    # 145 OH-CN
    "AI": -557.5079,    # 146 OH-aromatic O
    "BB": -1186.0500,   # 148 OH(a)-OH(a)
    "BD": -333.5638,    # 151 OH(a)-etherO
    "BQ": -878.0615,    # 155 OH(a)-nitro
    "MM": 135.3183,     # 157 NH2-NH2
    "MD": 219.9701,     # 159 NH2-etherO
    "ND": -134.4625,    # 166 NH-etherO
    "DD": 132.0275,     # 178 etherO-etherO
    "DF": 44.8702,      # 180 etherO-ester
    "DG": -219.5265,    # 181 etherO-ketone
    "DH": 546.5846,     # 182 etherO-aldehyde
    "DQ": -59.3635,     # 184 etherO-nitro
    "FF": 964.0840,     # 189 ester-ester
    "FG": 126.0380,     # 190 ester-ketone
    "FP": 539.2401,     # 192 ester-CN
    "GG": 3705.4400,    # 194 ketone-ketone: questionable per [4]
    "HI": 50.1063,      # 204 aldehyde-aromatic O
    "QQ": 896.3606,     # 206 nitro-nitro
    "PS": -196.6361,    # 209 CN-aromatic N r6
    # 218 COOH-NH2: do not estimate (zwitterion; flagged at fragmentation)
})

VISC_TV_GROUPS: dict[int, float | None] = {
    1: 89.0803,     2: 216.0226,    3: 80.9698,     4: 60.3316,
    5: 24.2637,     6: 244.4643,    7: 103.4109,    8: -16.5212,
    9: 174.1316,    10: -37.7584,   11: 252.0190,   12: 251.9299,
    13: 330.7100,   14: 294.3323,   15: 113.9028,   16: -26.6195,
    17: 133.5499,   18: 128.4739,   19: 208.3258,   21: 35.2688,
    22: 207.3562,   23: -15.8544,   24: 112.1172,   25: 329.0113,
    26: 313.1106,   27: 194.6060,   28: -8.6247,    29: 182.7067,
    30: 456.3713,   31: 391.6060,   32: 499.2149,   33: 1199.4010,
    34: 1198.1040,  35: 1078.0840,  36: 1284.7450,  37: 1134.1640,
    38: -34.9892,   39: 612.7222,   40: 458.7425,   41: 705.1250,
    42: 159.5146,   43: -284.4707,  44: 1446.0240,  45: 325.5736,
    46: 454.1671,   47: 374.6477,   48: 289.9690,   49: 1150.8290,
    50: 1619.1650,  51: 304.5982,   52: 394.7932,   53: 294.7319,
    54: 206.6432,   55: 292.3613,   56: 302.2321,   57: 346.9998,
    58: -23.9801,   59: 238.3242,   60: 137.5408,   61: 74.4489,
    62: 304.9257,   63: -32.4179,   64: 57.8131,    65: 279.2114,
    66: 662.0051,   67: 277.5038,   68: 369.4221,   69: 488.1136,
    70: -10.6146,   71: -181.7627,  72: 351.0623,   73: -15.2801,
    74: 174.3672,   75: 1098.1570,  76: 549.1481,   77: 394.5776,
    78: 10.3752,    79: 365.8081,   80: 164.8904,   81: 197.1806,
    82: 1297.7560,  86: 35.4672,    88: 495.5141,   89: 551.9254,
    90: 490.7224,   92: 669.0158,   93: 256.5078,   96: 1787.0390,
    97: 220.0803,   98: 229.4135,   100: 131.2253,  102: 53.2507,
    103: 288.4140,  104: 542.6641,  105: 714.0494,  107: 797.2271,
    108: 253.5303,  110: -237.2545,
    214: 192.1303,  215: 377.7146,  216: 806.8125,
}

VISC_TV_CORRECTIONS: dict[int, float | None] = {
    119: -180.3686, 121: 241.8968,  122: 138.6555,  123: -71.1647,
    124: -115.0418, 125: -96.7544,  126: -153.8442, 127: -22.1041,
    128: 24.7835,   130: 224.2439,  131: 24.2539,   132: 137.8708,
    134: -54.1782,
    217: -726.4291,
}

VISC_TV_INTERACTIONS: dict[frozenset, float | None] = _pairs({
    "AA": -1313.5690,   # 135 OH-OH
    "AM": -41.9608,     # 136 OH-NH2
    "AN": -1868.6060,   # 137 OH-NH
    "AD": -643.4378,    # 140 OH-etherO
    "AP": -345.7844,    # 145 OH-CN
    "AI": 50.2582,      # 146 OH-aromatic O
    "BB": -1146.1070,   # 148 OH(a)-OH(a)
    "BD": -229.2406,    # 151 OH(a)-etherO
    "BQ": 515.1511,     # 155 OH(a)-nitro
    "MM": 86.7249,      # 157 NH2-NH2
    "MD": -57.1437,     # 159 NH2-etherO
    "ND": 54.2025,      # 166 NH-etherO
    "DD": 156.7495,     # 178 etherO-etherO
    "DF": 273.6616,     # 180 etherO-ester
    "DG": -339.6071,    # 181 etherO-ketone
    "DH": 1050.3190,    # 182 etherO-aldehyde
    "DQ": 355.0508,     # 183(sic) etherO-nitro
    "FF": 167.7204,     # 189 ester-ester
    "FG": 244.0583,     # 190 ester-ketone
    "FP": 334.4856,     # 192 ester-CN
    "GG": 1985.8270,    # 194 ketone-ketone: questionable per [4]
    "HI": 161.7447,     # 204 aldehyde-aromatic O
    "QQ": 1839.2630,    # 206 nitro-nitro
    "PS": 718.1262,     # 209 CN-aromatic N r6
    # 218 COOH-NH2: do not estimate (zwitterion; flagged at fragmentation)
})

# ---------------------------------------------------------------------------
# Local refits (NOT part of the published method).
# Unlike groups 219 / the COOH-COOH interaction (which fill published *holes*),
# these OVERRIDE published values that proved systematically biased.  They are
# applied by default; pass local_refits=False to estimate()/estimate_from_mol()
# for byte-faithful published behavior.  Every use is reported in warnings.
# Fitted 2026-07-09 against Ambrose/Tsonopoulos/Nikitin/Morton/Marsh, J. Chem.
# Eng. Data 61 (2016) (Part 12 recommended values,
# data/reference/pure-component-properties/2016-ambrose-vapor-liquid-critical-properties-review.pdf) and Perry 9th ed. Table 2-106:
#   44/pc: linear saturated acids C3-C18 from ATN-12 (ethanoic excluded:
#          dimerization; C20/C22 excluded: alkyl-tail extrapolation dominates).
#   53/pc: thiol series C3-C10 (Perry) + 1-dodecanethiol (ATN-12); the
#          published value fits only methyl/ethyl thiol, its own 2-compound
#          training set, which consequently degrade under the refit.
#   37/vc: phenol + o/p-cresol (Perry 2-106); published value (2 unknown
#          compounds, "0.0" error) gave +20..25 % for the whole family.
#          m-cresol excluded from the fit: its listed (Pc, Vc) pair is a
#          correlated outlier -- the family Zc is conserved to +-0.001 while
#          Pc spreads 10 % (i.e. Vc back-calculated from Pc), and Rackett
#          inversion of the measured liquid densities (Vs at 323 K: 106.0 /
#          107.0 / 107.0 cm3/mol for o/m/p) shows the three cresols must
#          have near-equal Vc.  The refit predicts m-cresol Vc ~286, not
#          the listed 312.
# Post-refit accuracy: acids C3-C18 AAD 1.6 % (ethanoic -7.6 %: dimerization;
# C20/C22 -4.5/-8.1 %: alkyl-tail); thiols class AAD 13.8 % -> 4.9 %
# (methanethiol -20 % and ethanethiol -8 % degrade -- for those two the
# published value is better, use local_refits=False); phenol/o-/p-cresol
# within +-2.3 %.
LOCAL_REFITS: dict[int, dict[str, float]] = {
    44: {"pc": 0.5750},     # published 3.9873 (*1e-4)
    53: {"pc": -1.9044},    # published -9.4154 (*1e-4)
    37: {"vc": -11.9568},   # published 25.6584 (cm3/mol)
}

# Local extensions that fill holes in the published first-order table.  These
# are deliberately separate from LOCAL_REFITS: local_refits=False must retain
# the paper's original "not estimable" result rather than silently treating a
# locally fitted value as published.
#
# Group 69 (aryl-NO2), fitted 2026-08-11; full inverse diagnostic and source
# records: scripts/nannoolal/refit_group69_aromatic_nitro.py
#
# Tc: median/central value 85.41 (*1e-3) from seven uncomplicated mononitro
#     experimental compounds plus Perry 1,3,5-trinitrobenzene and TNT.  At
#     85.0, those nine Tc residuals are -2.3..+3.3 K.  EOS-effective DNTs are
#     excluded from the Tc center because their decomposition/extrapolated Tb
#     choices move the inferred contribution by 40-70 table units.
# Pc: robust median 2.621 (*1e-4) across eleven available Perry,
#     uncertainty-qualified experimental, and q>0.90 EOS-effective Pc values.
#     This is the weakest extension: most predictions are useful, but
#     nitrobenzene/nitrotoluenes differ by 13-22 % and q=0.94 3,5-DNT by 52 %.
# Vc: median 48.527 cm3/mol from seven Perry or q>0.90 EOS-effective values
#     (nitrobenzene; o/m/p-nitrotoluene; 1,3,5-TNB; TNT; 2,4-DNT).  Individual
#     implied contributions are 38.27..54.27; q=0.90/0.70 DNT holdouts straddle
#     the center at 27.12/66.97 and were not fitted.
LOCAL_GROUP_EXTENSIONS: dict[int, dict[str, float]] = {
    AROMATIC_NITRO_GROUP: {"tc": 85.41, "pc": 2.621, "vc": 48.527},
}

# The COOH-COOH Pc interaction is calibrated jointly with group 44: the value
# in INTERACTION_CONTRIBUTIONS (-47.7547) is consistent with the PUBLISHED
# group 44 and is used when local_refits=False; with the refitted group 44
# most of the diacid deviation is explained additively and the consistent
# interaction shrinks to -3.8213 (Nikitin diacids n=3..12, AAD 2.2 %).
LOCAL_INTERACTION_REFITS: dict[frozenset, dict[str, float]] = {
    frozenset("CC"): {"pc": -3.8213},
}

# OH-OH Tc interaction, subtype-aware split (local extension, revised
# 2026-07-10).  After the 35/36 chain-rule correction (see module notes), the
# per-diol exact-fit interaction values cluster by the OH GROUP SUBTYPE of the
# pair, not by vicinality (ATN Part-12 Tc data + Nikitin-verified MEG 720 K,
# PFDSim-validated / literature Tbs):
#   * 36+36 pairs (both OH short-chain primary): MEG -398, 1,3-PD -405,
#     2-Me-1,3-PD -475, neopentyl glycol -372, 1,4-BD -425 -- the PUBLISHED
#     -434.8568 sits inside the cluster and is used as-is (residuals +-7 K).
#     The old "vicinal" split was an artifact of the chain-rule bug, which
#     misassigned every non-vicinal C3/C4 diol to group 35.
#   * pairs involving a secondary OH (34): -241 (1,2-PD), -259 (1,3-BD);
#     fitted constant -252.  Holdout 1,2-butanediol (Tc 694 K after +14 K
#     same-source bias correction) comes out -20 K -- the weakest point of
#     the scheme, on the shakiest datum (a Psat-paper Tc).
#   * pairs involving a long-chain OH (35): C5-C10 terminal diols give
#     +70/-45/+105/+2/-182 (LSQ +14, indistinguishable from zero at the
#     +-8..12 K data uncertainty) -> treated as additive (0.0), i.e. the
#     interaction is dropped.  Validated: 4-oxa-1,7-heptanediol +3.8,
#     tetraethylene glycol -0.3.  Known misses of the published parameter
#     set: DEG -27 K, TEG -22 K (both 2-3 sigma vs. Nikitin; documented,
#     not locally patched).
# With local_refits=False the published constant applies to ALL pairs.
_AA_TC_34 = -252.0              # *1e-3, fit: 1,2-propanediol + 1,3-butanediol
_AA_TC_35 = 0.0                 # additive; LSQ +14 == 0 within data noise


def _aa_tc_value(frag, i: int, j: int) -> float:
    """Local Tc value (*1e-3) for an OH-OH pair, keyed on group subtype."""
    gids = {frag.interaction_gids[i], frag.interaction_gids[j]}
    if 35 in gids:
        return _AA_TC_35
    if 34 in gids:
        return _AA_TC_34
    return INTERACTION_CONTRIBUTIONS[frozenset("A")][1]   # published, 36+36


_PROP_INDEX = {"tb": 0, "tc": 1, "pc": 2, "vc": 3}
_PROP_SCALE = {"tb": 1.0, "tc": _TC_SCALE, "pc": _PC_SCALE, "vc": 1.0}

# ---------------------------------------------------------------------------
# Fragmentation
# ---------------------------------------------------------------------------

_HALOGENS = {9, 17}          # F, Cl (Br/I are separate groups, never "e")
_E_ELEMENTS = {7, 8, 9, 17}  # "very electronegative" N, O, F, Cl
_C_OR_SI = {6, 14}


def _is_e_neighbor(atom: Chem.Atom) -> bool:
    """Non-aromatic N/O/F/Cl neighbor ("e" in the papers' abbreviations)."""
    return any(nb.GetAtomicNum() in _E_ELEMENTS and not nb.GetIsAromatic()
               for nb in atom.GetNeighbors())


def _has_aromatic_c_neighbor(atom: Chem.Atom) -> bool:
    return any(nb.GetAtomicNum() == 6 and nb.GetIsAromatic()
               for nb in atom.GetNeighbors())


def _has_aromatic_neighbor(atom: Chem.Atom) -> bool:
    return any(nb.GetIsAromatic() for nb in atom.GetNeighbors())


def _halogen_env(halogen: Chem.Atom):
    """For a halogen bonded to C/Si: (host, n_other_F, n_other_Cl, n_other_heavy)."""
    host = halogen.GetNeighbors()[0]
    n_f = n_cl = n_heavy = 0
    for nb in host.GetNeighbors():
        if nb.GetIdx() == halogen.GetIdx():
            continue
        n_heavy += 1
        if nb.GetAtomicNum() == 9:
            n_f += 1
        elif nb.GetAtomicNum() == 17:
            n_cl += 1
    return host, n_f, n_cl, n_heavy


def _host_is_sp3_c_or_si(host: Chem.Atom) -> bool:
    if host.GetAtomicNum() == 14:
        return True
    return (host.GetAtomicNum() == 6 and not host.GetIsAromatic()
            and host.GetHybridization() == Chem.HybridizationType.SP3)


def _host_is_vinyl_c(host: Chem.Atom) -> bool:
    """Non-aromatic sp2 carbon participating in a C=C double bond."""
    if host.GetAtomicNum() != 6 or host.GetIsAromatic():
        return False
    for bond in host.GetBonds():
        if (bond.GetBondType() == Chem.BondType.DOUBLE
                and bond.GetOtherAtom(host).GetAtomicNum() == 6):
            return True
    return False


def _count_c_si_neighbors(atom: Chem.Atom) -> int:
    return sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() in _C_OR_SI)


def _c_si_chain_reaches(atom: Chem.Atom, target: int) -> bool:
    """True if a simple heavy-atom path from `atom` contains >= target C/Si.

    Only C/Si atoms on the path count toward `target`; the path may traverse
    heteroatoms (DEGME reaches its 5th carbon across two ether oxygens) and
    `atom` itself is not counted.
    """
    def dfs(a, visited, count):
        if count >= target:
            return True
        for nb in a.GetNeighbors():
            if nb.GetIdx() not in visited:
                visited.add(nb.GetIdx())
                nxt = count + (1 if nb.GetAtomicNum() in _C_OR_SI else 0)
                if dfs(nb, visited, nxt):
                    return True
                visited.discard(nb.GetIdx())
        return False
    return dfs(atom, {atom.GetIdx()}, 0)


# --- predicates for single-atom (environment-classified) groups -------------

def _pred_f(gid):
    def pred(mol, match):
        f = mol.GetAtomWithIdx(match[0])
        if f.GetDegree() != 1 or f.GetNeighbors()[0].GetAtomicNum() not in _C_OR_SI:
            return False
        host, n_f, n_cl, n_heavy = _halogen_env(f)
        sp3 = _host_is_sp3_c_or_si(host)
        if gid == 21:
            return sp3 and n_f >= 1 and n_heavy >= 3
        if gid == 102:
            return sp3 and n_cl >= 1 and n_heavy >= 3
        if gid == 23:
            return sp3 and (n_f + n_cl) == 2
        if gid == 22:
            return sp3 and (n_f + n_cl) >= 1
        if gid == 20:
            return _host_is_vinyl_c(host)
        if gid == 24:
            return host.GetIsAromatic()
        return True  # 19: generic F-(C, Si)
    return pred


def _pred_cl(gid):
    def pred(mol, match):
        cl = mol.GetAtomWithIdx(match[0])
        if cl.GetDegree() != 1 or cl.GetNeighbors()[0].GetAtomicNum() not in _C_OR_SI:
            return False
        host, n_f, n_cl, _ = _halogen_env(cl)
        sp3 = _host_is_sp3_c_or_si(host)
        if gid == 27:
            return sp3 and (n_f + n_cl) >= 2
        if gid == 29:
            return _host_is_vinyl_c(host)
        if gid == 26:
            return sp3 and (n_f + n_cl) == 1
        if gid == 25:
            return sp3 and (n_f + n_cl) == 0
        return host.GetIsAromatic()  # 28
    return pred


def _pred_oh(gid):
    def pred(mol, match):
        o = mol.GetAtomWithIdx(match[0])
        if o.GetDegree() != 1:
            return False
        host = o.GetNeighbors()[0]
        if host.GetAtomicNum() not in _C_OR_SI:
            return False
        if gid == 37:
            return host.GetIsAromatic()
        if host.GetIsAromatic():
            return False
        n_csi = _count_c_si_neighbors(host)
        if gid == 35:   # primary, >= 5 C/Si in a chain from the hydroxyl O
            return n_csi == 1 and _c_si_chain_reaches(o, 5)
        if gid == 34:   # secondary
            return n_csi == 2
        if gid == 33:   # tertiary: host has four non-hydrogen neighbors
            return host.GetTotalNumHs() == 0 and host.GetDegree() == 4
        return True     # 36: short-chain / remaining aliphatic OH
    return pred


def _pred_e_carbon(mol, match):
    return _is_e_neighbor(mol.GetAtomWithIdx(match[0]))


def _pred_aromatic_nb(mol, match):
    return _has_aromatic_neighbor(mol.GetAtomWithIdx(match[0]))


def _pred_aromatic_c_nb(mol, match):
    return _has_aromatic_c_neighbor(mol.GetAtomWithIdx(match[0]))


def _pred_ring_c_exo_e(mol, match):
    """Group 12: ring C with N/O not part of a ring, or any Cl/F neighbor."""
    atom = mol.GetAtomWithIdx(match[0])
    for nb in atom.GetNeighbors():
        z = nb.GetAtomicNum()
        if z in (9, 17):
            return True
        if z in (7, 8) and not nb.GetIsAromatic() and not nb.IsInRing():
            return True
    return False


def _pred_ring_c_ring_e(mol, match):
    """Group 13: ring C with a ring-member (non-aromatic) N/O neighbor."""
    atom = mol.GetAtomWithIdx(match[0])
    return any(nb.GetAtomicNum() in (7, 8) and not nb.GetIsAromatic()
               and nb.IsInRing() for nb in atom.GetNeighbors())


def _aromatic_bond_count(atom: Chem.Atom) -> int:
    return sum(1 for b in atom.GetBonds() if b.GetIsAromatic())


def _pred_ar_c_e(mol, match):
    """Group 17: aromatic C with an exocyclic (non-aromatic bond) N/O/F/Cl."""
    atom = mol.GetAtomWithIdx(match[0])
    for bond in atom.GetBonds():
        nb = bond.GetOtherAtom(atom)
        if nb.GetAtomicNum() in _E_ELEMENTS and not bond.GetIsAromatic():
            return True
    return False


def _pred_ar_c_fused(mol, match):
    return _aromatic_bond_count(mol.GetAtomWithIdx(match[0])) == 3


def _pred_ar_c_bridge(mol, match):
    """Group 214: aromatic C single-bonded to another aromatic atom (biphenyl)."""
    atom = mol.GetAtomWithIdx(match[0])
    for bond in atom.GetBonds():
        if not bond.GetIsAromatic() and bond.GetOtherAtom(atom).GetIsAromatic():
            return True
    return False


def _pred_olefin_e(mol, match):
    return any(_is_e_neighbor(mol.GetAtomWithIdx(i)) for i in match)


def _pred_olefin_ar(mol, match):
    return any(_has_aromatic_c_neighbor(mol.GetAtomWithIdx(i)) for i in match)


def _anhydride_ring_atoms(mol, match):
    """Smallest ring containing the C(=O)-O-C(=O) core of an anhydride match."""
    core = {match[0], match[2], match[3]}
    rings = [set(r) for r in mol.GetRingInfo().AtomRings() if core <= set(r)]
    return min(rings, key=len) if rings else None


def _pred_cyclic_anhydride_sp2(mol, match):
    """Group 96: the anhydride ring closes through a C=C or aromatic bond."""
    ring = _anhydride_ring_atoms(mol, match)
    if ring is None:
        return False
    for bond in mol.GetBonds():
        if {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} <= ring and (
                bond.GetIsAromatic()
                or (bond.GetBondType() == Chem.BondType.DOUBLE
                    and bond.GetBeginAtom().GetAtomicNum() == 6
                    and bond.GetEndAtom().GetAtomicNum() == 6)):
            return True
    return False


def _pred_cyclic_anhydride_sp3(mol, match):
    """Group 219 (extension): cyclic anhydride without sp2/aromatic closure."""
    return (_anhydride_ring_atoms(mol, match) is not None
            and not _pred_cyclic_anhydride_sp2(mol, match))


def _claim_anhydride_ring(mol, match):
    """Group 96 claims its whole ring plus the two carbonyl oxygens.  This is
    the reading of "(-C=O-O-C=O-)r" that reproduces the published accuracy for
    maleic/citraconic/phthalic anhydride (see module docstring)."""
    ring = _anhydride_ring_atoms(mol, match)
    atoms = set(match) | (ring or set())
    return tuple(sorted(atoms))


# --- assigner registry -------------------------------------------------------
# (priority, group id, SMARTS, claimed match indices or None for all,
#  optional extra predicate)
# Priorities are the PR numbers of Table 2 of [2] (Part-1 PRs for groups that
# only exist there); lower PR = matched first.

_RAW_ASSIGNERS: list[tuple] = [
    (1,   118, "[CX3;!$([CX3]([#8])[#8])](=[OX1])[CX3;!$([CX3]([#8])[#8])]=[OX1]", None, None),
    (2,   99,  "[OX2][CX3](=[OX1])[NX3]", None, None),
    (3,   100, "[NX3][CX3](=[OX1])[NX3]", None, None),
    (4,   51,  "[CX3](=[OX1])([#6;A])[#6;A]", (0, 1), None),
    (5,   87,  "[CX3]=[CX2]=[CX3]", None, None),
    (6,   88,  "[CX3;R]=[CX3;R][CX3;R]=[CX3;R]", None, None),
    (7,   89,  "[CX3]=[CX3][CX3]=[CX3]", None, None),
    (8,   95,  "[CX2]#[CX2][CX2]#[CX2]", None, None),
    (9,   73,  "[PX4](=[OX1])([OX2])([OX2])[OX2]", None, None),
    (10,  96,  "[CX3;R](=[OX1])[OX2;R][CX3;R]=[OX1]", _claim_anhydride_ring,
     _pred_cyclic_anhydride_sp2),
    (10.5, 219, "[CX3;R](=[OX1])[OX2;R][CX3;R]=[OX1]", None,
     _pred_cyclic_anhydride_sp3),
    (11,  76,  "[CX3](=[OX1])[OX2][CX3]=[OX1]", None, None),
    (12,  85,  "[Ge](Cl)(Cl)Cl", None, None),
    (13,  72,  "[OX2][NX3,NX3+](~[OX1])~[OX1]", None, None),
    (14,  79,  "[OX2;!R][CX3;!R](=[OX1])[OX2;!R]", None, None),
    (15,  78,  "[B]([OX2])([OX2])[OX2]", None, None),
    (16,  84,  "[As](Cl)Cl", None, None),
    (17,  82,  "[#6][SX4](=[OX1])=[OX1]", (1, 2, 3), None),
    (18,  77,  "[CX3](=[OX1])[Cl]", None, None),
    (19,  81,  "[NX2]=[CX2]=[SX1]", None, None),
    (20,  68,  "[#6;A][NX3,NX3+](~[OX1])~[OX1]", (1, 2, 3), None),
    (21,  69,  "[c][NX3,NX3+](~[OX1])~[OX1]", (1, 2, 3), None),
    (22,  74,  "[OX2;$([OX2][#6])][NX2]=[OX1]", None, None),
    (23,  44,  "[#6][CX3](=[OX1])[OX2H1]", (1, 2, 3), None),
    (24,  45,  "[#6][CX3;!R](=[OX1])[OX2H0;!R][#6]", (1, 2, 3), None),
    (25,  47,  "[#6][CX3;R](=[OX1])[OX2;R]", (1, 2, 3), None),
    (26,  46,  "[CX3H1](=[OX1])[OX2][#6]", (0, 1, 2), None),
    (27,  50,  "[CX3](=[OX1])[NX3H2]", None, None),
    (28,  80,  "[NX2]=[CX2]=[OX1]", None, None),
    (29,  75,  "[CX3]=[NX2][OX2H1]", None, None),
    (31,  94,  "[OX2][OX2]", None, None),
    (32,  101, "[NX4,NX4+]", None, None),
    (33,  103, "[OX2;R][CX3;R](=[OX1])[OX2;R]", None, None),
    (34,  104, "[SX4](=[OX1])(=[OX1])([OX2])[OX2]", None, None),
    (35,  105, "[SX4](=[OX1])(=[OX1])[NX3]", None, None),
    (36,  106, "[nX3]1[cX3][nX2][cX3][cX3]1", None, None),
    (37,  107, "[SX3](=[OX1])", None, None),
    (38,  108, "[SX2][CX2]#[NX1]", (1, 2), None),
    (39,  109, "[CX3](=[OX1])[SX2]", (0, 1), None),
    (41,  111, "[NX3][CX2]#[NX1]", (1, 2), None),
    (42,  110, "[PX3]([OX2])([OX2])[OX2]", None, None),
    (43,  113, "[PX3]", None, None),
    (45,  115, "[oX2][nX2]", None, None),
    (46,  116, "[Se]", None, None),
    (47,  117, "[Al]", None, None),
    (48,  49,  "[CX3](=[OX1])[NX3H1]", None, None),
    (49,  48,  "[CX3](=[OX1])[NX3H0]", None, None),
    (50,  39,  "[#6]1[OX2][#6]1", None, None),
    (51,  55,  "[#6][SX2][SX2][#6]", (1, 2), None),
    (52,  90,  "[c][CX3H1]=[OX1]", (1, 2), None),
    (53,  52,  "[#6;A][CX3H1]=[OX1]", (1, 2), None),
    (54,  92,  "[#6][CX3](=[OX1])[c]", (1, 2), None),
    (55,  57,  "[#6][CX2]#[NX1]", (1, 2), None),
    (56,  64,  "[CX2H1]#[CX2]", None, None),
    (57,  61,  "[CX3H2]=[CX3]", None, None),
    (58,  60,  "[CX3;!R]=[CX3;!R]", None, _pred_olefin_e),
    (59,  59,  "[CX3;!R]=[CX3;!R]", None, _pred_olefin_ar),
    (60,  62,  "[CX3;R]=[CX3;R]", None, None),
    (61,  63,  "[CX2]#[CX2]", None, None),
    (62,  58,  "[CX3]=[CX3]", None, None),
    (63,  32,  "[I;$(I[#6,#14])]", None, None),
    (64,  83,  "[Sn]", None, None),
    (65,  30,  "[Br;$(Br[#6;A]),$(Br[#14])]", None, None),
    (66,  31,  "[Br;$(Br[c])]", None, None),
    (67,  86,  "[Ge]", None, None),
    (68,  27,  "[Cl]", None, _pred_cl(27)),
    (69,  29,  "[Cl]", None, _pred_cl(29)),
    (70,  26,  "[Cl]", None, _pred_cl(26)),
    (71,  25,  "[Cl]", None, _pred_cl(25)),
    (72,  28,  "[Cl]", None, _pred_cl(28)),
    (73,  53,  "[SX2H1]", None, None),
    (74,  54,  "[SX2H0]([#6])[#6]", (0,), None),
    (75,  56,  "[sX2]", None, None),
    (76,  "Si", "[#14]", None, None),
    (81,  21,  "[F]", None, _pred_f(21)),
    (82,  102, "[F]", None, _pred_f(102)),
    (83,  23,  "[F]", None, _pred_f(23)),
    (84,  22,  "[F]", None, _pred_f(22)),
    (85,  20,  "[F]", None, _pred_f(20)),
    (86,  24,  "[F]", None, _pred_f(24)),
    (87,  19,  "[F]", None, _pred_f(19)),
    (88,  35,  "[OX2H1]", None, _pred_oh(35)),
    (89,  37,  "[OX2H1]", None, _pred_oh(37)),
    (90,  34,  "[OX2H1]", None, _pred_oh(34)),
    (91,  33,  "[OX2H1]", None, _pred_oh(33)),
    (92,  36,  "[OX2H1]", None, _pred_oh(36)),
    (93,  65,  "[oX2]", None, None),
    (94,  38,  "[OX2H0]([#6,#14])[#6,#14]", (0,), None),
    (95,  41,  "[NX3H2][c]", (0,), None),
    (96,  40,  "[NX3H2][#6,#14]", (0,), None),
    (97,  67,  "[nX2;r6]", None, None),
    (98,  66,  "[nX2;r5]", None, None),
    (99,  97,  "[$([NX3H1;R]([#6,#14])[#6,#14]),nX3H1]", None, None),
    (99.5, 98, "[NX3H1;!R;$([NX3H1][c])]([#6,#14])[#6,#14]", (0,), None),
    (100, 42,  "[NX3H1;!R]([#6,#14])[#6,#14]", (0,), None),
    (101, 43,  "[$([NX3H0;!$([NX3]=*)]([#6,#14])([#6,#14])[#6,#14]),$([nX3H0](:*)(:*)[#6])]", (0,), None),
    (102, 91,  "[NX2]=[#6;A]", None, None),
    (103, 2,   "[CX4H3]", None, _pred_e_carbon),
    (104, 3,   "[CX4H3]", None, _pred_aromatic_nb),
    (105, 1,   "[CX4H3]", None, None),
    (106, 15,  "[cX3H1]", None, None),
    (107, 14,  "[CX4;R]", None, _pred_aromatic_c_nb),
    (108, 7,   "[CX4;!R]", None, _pred_e_carbon),
    (109, 8,   "[CX4;!R]", None, _pred_aromatic_c_nb),
    (110, 12,  "[CX4;R]", None, _pred_ring_c_exo_e),
    (111, 13,  "[CX4;R]", None, _pred_ring_c_ring_e),
    (112, 4,   "[CX4H2;!R]", None, None),
    (113, 9,   "[CX4H2;R]", None, None),
    (114, 17,  "[c]", None, _pred_ar_c_e),
    (115, 18,  "[c]", None, _pred_ar_c_fused),
    (116, 214, "[c]", None, _pred_ar_c_bridge),
    (117, 16,  "[c]", None, None),
    (118, 10,  "[CX4H1;R]", None, None),
    (119, 5,   "[CX4H1;!R]", None, None),
    (120, 11,  "[CX4H0;R]", None, None),
    (121, 6,   "[CX4H0;!R]", None, None),
]

_ASSIGNERS = sorted(
    ((prio, gid, Chem.MolFromSmarts(smarts), claim_idx, pred)
     for prio, gid, smarts, claim_idx, pred in _RAW_ASSIGNERS),
    key=lambda entry: entry[0],
)
assert all(patt is not None for _, _, patt, _, _ in _ASSIGNERS), \
    "invalid SMARTS in assigner registry"


@dataclass
class Fragmentation:
    """Result of assigning every heavy atom to a Nannoolal structural group."""
    mol: Chem.Mol
    n_atoms: int
    n_hydrogens: int
    groups: Counter = field(default_factory=Counter)          # id -> frequency
    instances: list = field(default_factory=list)             # (id, atom idx tuple)
    si_atoms: list = field(default_factory=list)              # per-Si env dicts
    corrections: Counter = field(default_factory=Counter)     # id -> frequency
    interaction_classes: list = field(default_factory=list)   # e.g. ['A','A','N']
    interaction_atoms: list = field(default_factory=list)     # claimed atoms per instance
    interaction_gids: list = field(default_factory=list)      # group id per instance
    do_not_estimate: str | None = None
    notes: list = field(default_factory=list)


_EXO_CARBONYL_AROMATIC = Chem.MolFromSmarts("[c]=[OX1,SX1]")


def _demote_carbonyl_aromatic_rings(mol: Chem.Mol) -> Chem.Mol:
    """Return a copy where "aromatic" rings bearing an exocyclic C=O (2-pyranones
    such as coumarin, 2-pyridones, ...) are demoted to their Kekule form, so the
    lactone/amide/ketone groups can claim the carbonyl.  Atoms shared with a
    genuinely aromatic fused ring stay aromatic."""
    carbonyl_carbons = {m[0] for m in mol.GetSubstructMatches(_EXO_CARBONYL_AROMATIC)}
    if not carbonyl_carbons:
        return mol
    mol = Chem.Mol(mol)
    try:
        Chem.Kekulize(mol, clearAromaticFlags=False)
    except Exception:
        return mol
    rings = mol.GetRingInfo().AtomRings()
    affected = [set(r) for r in rings if carbonyl_carbons & set(r)]
    untouched_aromatic = set().union(*(
        set(r) for r in rings
        if not (carbonyl_carbons & set(r))
        and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r))) \
        if len(rings) > len(affected) else set()
    demote = set().union(*affected) - untouched_aromatic
    for idx in demote:
        mol.GetAtomWithIdx(idx).SetIsAromatic(False)
    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() in demote or bond.GetEndAtomIdx() in demote:
            bond.SetIsAromatic(False)
    return mol


def fragment(mol: Chem.Mol) -> Fragmentation:
    """Assign every heavy atom of `mol` to a structural group (PR order)."""
    mol = _demote_carbonyl_aromatic_rings(mol)
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < 2:
        raise FragmentationError(
            "the Nannoolal method needs at least two heavy atoms")
    n_h = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    frag = Fragmentation(mol=mol, n_atoms=n_heavy, n_hydrogens=n_h)

    claimed: set[int] = set()
    for _prio, gid, patt, claim_idx, pred in _ASSIGNERS:
        seen_claims: set[frozenset] = set()
        for match in mol.GetSubstructMatches(patt):
            if claim_idx is None:
                atoms = match
            elif callable(claim_idx):
                atoms = claim_idx(mol, match)
            else:
                atoms = tuple(match[i] for i in claim_idx)
            key = frozenset(atoms)
            if key in seen_claims or key & claimed:
                continue
            if pred is not None and not pred(mol, match):
                continue
            seen_claims.add(key)
            claimed.update(atoms)
            if gid == "Si":
                si = mol.GetAtomWithIdx(atoms[0])
                n_ch = (sum(1 for nb in si.GetNeighbors() if nb.GetAtomicNum() == 6)
                        + si.GetTotalNumHs())
                frag.si_atoms.append({
                    "n_ch": n_ch,
                    "has_halogen": any(nb.GetAtomicNum() in (9, 17, 35, 53)
                                       for nb in si.GetNeighbors()),
                    "has_o": any(nb.GetAtomicNum() == 8 for nb in si.GetNeighbors()),
                })
            else:
                frag.groups[gid] += 1
                frag.instances.append((gid, atoms))
                cls = INTERACTION_CLASS.get(gid)
                if cls:
                    frag.interaction_classes.append(cls)
                    frag.interaction_atoms.append(atoms)
                    frag.interaction_gids.append(gid)
            if gid == DO_NOT_ESTIMATE_GROUP:
                frag.do_not_estimate = (
                    "1,2-diketone (group 118): 'do not fragment' per [2]")

    unclaimed = [a for a in mol.GetAtoms() if a.GetIdx() not in claimed]
    if unclaimed:
        detail = ", ".join(f"{a.GetSymbol()}({a.GetIdx()})" for a in unclaimed[:8])
        raise FragmentationError(
            f"could not assign {len(unclaimed)} atom(s) to any published "
            f"structural group: {detail}")

    if 219 in frag.groups:
        frag.notes.append(
            "group 219 (saturated cyclic anhydride) is a local extension "
            "fitted to 3 compounds, not part of the published method")
    if frag.interaction_classes.count("C") >= 2:
        frag.notes.append(
            "COOH-COOH Tc/Pc interaction is a local extension fitted to "
            "pulse-heating diacid data, not part of the published method")
    if frag.interaction_classes.count("A") >= 2:
        frag.notes.append(
            "OH-OH Tc interaction uses a group-subtype split (34/35/36) "
            "fitted to pulse-heating diol data (local extension; the "
            "published single constant applies with local_refits=False)")

    if _ZWITTERION_PAIR <= set(frag.interaction_classes):
        frag.do_not_estimate = (
            "molecule contains both COOH and NH2 (zwitterion, interaction "
            "218): 'do not estimate' per [2]")

    _add_corrections(frag)
    return frag


# --- second-order corrections ------------------------------------------------

def _add_corrections(frag: Fragmentation) -> None:
    mol = frag.mol
    corr = frag.corrections

    # 123/124: molecules with no / exactly one hydrogen.
    if frag.n_hydrogens == 0:
        corr[123] += 1
    elif frag.n_hydrogens == 1:
        corr[124] += 1

    # 121/122: halogenated carbons.
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetIsAromatic():
            continue
        n_hal = sum(1 for nb in atom.GetNeighbors()
                    if nb.GetAtomicNum() in _HALOGENS)
        n_c = sum(1 for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 6)
        if n_hal == 3:
            corr[121] += 1
        elif n_hal == 2 and n_c == 2:
            corr[122] += 1

    # 119/120: carbonyl carbon next to one / two carbons bearing >= 2 F/Cl.
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or not any(
                b.GetBondType() == Chem.BondType.DOUBLE
                and b.GetOtherAtom(atom).GetAtomicNum() == 8
                for b in atom.GetBonds()):
            continue
        n_dihalo = sum(
            1 for nb in atom.GetNeighbors()
            if nb.GetAtomicNum() == 6
            and sum(1 for nn in nb.GetNeighbors()
                    if nn.GetAtomicNum() in _HALOGENS) >= 2)
        if n_dihalo >= 2:
            corr[120] += 1
        elif n_dihalo == 1:
            corr[119] += 1

    # 125/126: small non-aromatic rings (per SSSR ring).
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        if len(ring) in (3, 4):
            corr[125] += 1
        elif len(ring) == 5:
            corr[126] += 1

    # 127/128/129: ortho/meta/para pairs on six-membered aromatic rings,
    # each kind counted once and only if it is the only kind present.
    n_ortho = n_meta = n_para = 0
    for ring in ring_info.AtomRings():
        if len(ring) != 6:
            continue
        atoms = [mol.GetAtomWithIdx(i) for i in ring]
        if not all(a.GetIsAromatic() for a in atoms):
            continue
        substituted = [pos for pos, a in enumerate(atoms)
                       if _aromatic_bond_count(a) < 3
                       and any(nb.GetIdx() not in ring for nb in a.GetNeighbors())]
        for i, pi in enumerate(substituted):
            for pj in substituted[i + 1:]:
                dist = min(abs(pi - pj), 6 - abs(pi - pj))
                if dist == 1:
                    n_ortho += 1
                elif dist == 2:
                    n_meta += 1
                else:
                    n_para += 1
    if n_ortho and not n_meta and not n_para:
        corr[127] += 1
    elif n_meta and not n_ortho and not n_para:
        corr[128] += 1
    elif n_para and not n_ortho and not n_meta:
        corr[129] += 1

    # 130-133: steric corrections around C-C bonds.
    def _further_c(atom, partner):
        return sum(1 for nb in atom.GetNeighbors()
                   if nb.GetAtomicNum() == 6 and nb.GetIdx() != partner.GetIdx())

    def _is_sp3_c(atom):
        return (atom.GetAtomicNum() == 6 and not atom.GetIsAromatic()
                and atom.GetHybridization() == Chem.HybridizationType.SP3)

    def _is_sp2_c(atom):
        return atom.GetAtomicNum() == 6 and (
            atom.GetIsAromatic() or _host_is_vinyl_c(atom))

    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        if _is_sp3_c(a1) and _is_sp3_c(a2):
            counts = sorted((_further_c(a1, a2), _further_c(a2, a1)))
            if counts == [3, 3]:
                corr[133] += 1
            elif counts == [2, 3]:
                corr[132] += 1
            elif counts == [2, 2]:
                corr[131] += 1
        else:
            for x, y in ((a1, a2), (a2, a1)):
                if _is_sp2_c(x) and _is_sp3_c(y) and _further_c(y, x) == 3:
                    corr[130] += 1
                    break

    # 134: C=C-C=O conjugation, once per aldehyde/ketone/acid/ester carbonyl
    # with an sp2-carbon neighbor (aromatic or olefinic).
    for gid, atoms in frag.instances:
        if gid not in (44, 45, 46, 47, 51, 52, 90, 92):
            continue
        carbonyl = next(
            (mol.GetAtomWithIdx(i) for i in atoms
             if mol.GetAtomWithIdx(i).GetAtomicNum() == 6), None)
        if carbonyl is not None and any(
                _is_sp2_c(nb) for nb in carbonyl.GetNeighbors()):
            corr[134] += 1

    # 217: silicon attached to a halogen (Tc/Pc/Vc only; Tb covers this via
    # the Part-1 silicon group 93).
    n_si_hal = sum(1 for si in frag.si_atoms if si["has_halogen"])
    if n_si_hal:
        corr[217] += n_si_hal


# ---------------------------------------------------------------------------
# Property evaluation
# ---------------------------------------------------------------------------

def _si_group_id(si_env: dict, prop: str) -> int:
    if prop == "tb":     # Part-1 scheme: neighbor type
        if si_env["has_halogen"]:
            return 93
        return 71 if si_env["has_o"] else 70
    # Part-2 scheme: number of C/H neighbors
    return {0: 216, 1: 215, 2: 93, 3: 71}.get(si_env["n_ch"], 70)


def _group_sum(frag: Fragmentation, prop: str, warnings: list,
               local_refits: bool = True) -> float | None:
    """sum(N_i C_i) over first-order groups + corrections, or None."""
    idx, scale = _PROP_INDEX[prop], _PROP_SCALE[prop]
    total = 0.0
    for gid, freq in frag.groups.items():
        value = GROUP_CONTRIBUTIONS[gid][idx]
        if local_refits and prop in LOCAL_GROUP_EXTENSIONS.get(gid, {}):
            value = LOCAL_GROUP_EXTENSIONS[gid][prop]
            warnings.append(
                f"{prop}: group {gid} uses a locally fitted extension; "
                "no contribution was published for this property "
                "(pass local_refits=False to disable)")
        elif local_refits and prop in LOCAL_REFITS.get(gid, {}):
            value = LOCAL_REFITS[gid][prop]
            warnings.append(
                f"{prop}: group {gid} uses a locally refitted value, "
                f"not the published one (pass local_refits=False to disable)")
        if value is None:
            warnings.append(
                f"{prop}: no published contribution for group {gid}; "
                f"property not estimable")
            return None
        total += freq * value * scale
    for si_env in frag.si_atoms:
        gid = _si_group_id(si_env, prop)
        value = GROUP_CONTRIBUTIONS[gid][idx]
        if value is None:
            warnings.append(
                f"{prop}: no published contribution for silicon group {gid}; "
                f"property not estimable")
            return None
        total += value * scale
    for cid, freq in frag.corrections.items():
        if prop == "tb" and cid == 217:
            continue
        value = CORRECTION_CONTRIBUTIONS[cid][idx]
        if value is None:
            warnings.append(
                f"{prop}: correction {cid} has no published value; ignored")
            continue
        total += freq * value * scale
    return total


def _interaction_sum(frag: Fragmentation, prop: str,
                     local_refits: bool = True) -> float:
    """GI = (1/n) * sum over ordered pairs of C(i-j) / (m - 1)."""
    classes = frag.interaction_classes
    m = len(classes)
    if m < 2:
        return 0.0
    idx, scale = _PROP_INDEX[prop], _PROP_SCALE[prop]
    num = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            pair = frozenset(classes[i] + classes[j])
            entry = INTERACTION_CONTRIBUTIONS.get(pair)
            value = entry[idx] if entry is not None else None
            if local_refits and prop in LOCAL_INTERACTION_REFITS.get(pair, {}):
                # keeps the pair calibration consistent with LOCAL_REFITS
                value = LOCAL_INTERACTION_REFITS[pair][prop]
            if local_refits and prop == "tc" and pair == frozenset("A"):
                # subtype-aware OH-OH split (see _AA_TC_* above)
                value = _aa_tc_value(frag, i, j)
            if value is None:
                continue  # missing interactions contribute zero, per [2]
            num += 2.0 * value * scale
    return num / (frag.n_atoms * (m - 1))


def _critical_interaction_refusal(frag: Fragmentation) -> str | None:
    """Return a local critical-only domain refusal for group 69 mixtures."""
    if AROMATIC_NITRO_GROUP not in frag.groups:
        return None
    refused = []
    if 41 in frag.groups:
        refused.append("aromatic primary amine (group 41-Q)")
    if 37 in frag.groups:
        refused.append("phenolic hydroxyl (group 37-Q)")
    if not refused:
        return None
    return (
        "critical properties: aromatic nitro directly combined with "
        + ", ".join(refused)
        + "; do not estimate because the corresponding critical interaction "
        "contribution is unpublished and not locally calibrated"
    )


@dataclass
class NannoolalResult:
    """Estimation result; unavailable properties are None (see warnings)."""
    smiles: str
    n_atoms: int
    n_hydrogens: int
    molar_mass: float
    tb_K: float | None
    tc_K: float | None
    pc_kPa: float | None
    vc_cm3_mol: float | None
    tb_source: str                       # "estimated" | "provided"
    groups: dict
    corrections: dict
    interaction_classes: list
    warnings: list

    def summary(self) -> str:
        def fmt(v, unit, nd=1):
            return f"{v:.{nd}f} {unit}" if v is not None else "n/a"
        lines = [
            f"SMILES        : {self.smiles}",
            f"heavy atoms   : {self.n_atoms}   M = {self.molar_mass:.2f} g/mol",
            f"groups        : {dict(sorted(self.groups.items(), key=lambda kv: str(kv[0])))}",
            f"corrections   : {dict(sorted(self.corrections.items()))}",
            f"interactions  : {''.join(sorted(self.interaction_classes)) or '-'}",
            f"Tb ({self.tb_source})  : {fmt(self.tb_K, 'K')}",
            f"Tc            : {fmt(self.tc_K, 'K')}",
            f"Pc            : {fmt(self.pc_kPa, 'kPa')}",
            f"Vc            : {fmt(self.vc_cm3_mol, 'cm3/mol')}",
        ]
        lines += [f"warning       : {w}" for w in self.warnings]
        return "\n".join(lines)


def estimate(smiles: str, tb: float | None = None,
             local_refits: bool = True) -> NannoolalResult:
    """Estimate Tb, Tc, Pc and Vc for a molecule given as SMILES.

    If an experimental normal boiling point `tb` [K] is supplied it is used
    for the Tc estimation (recommended by the papers); otherwise the
    internally estimated Tb is used.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise NannoolalError(f"could not parse SMILES {smiles!r}")
    return estimate_from_mol(mol, tb=tb, smiles=smiles,
                             local_refits=local_refits)


def estimate_from_mol(mol: Chem.Mol, tb: float | None = None,
                      smiles: str | None = None,
                      local_refits: bool = True) -> NannoolalResult:
    frag = fragment(mol)
    warnings: list = list(frag.notes)
    n = frag.n_atoms

    tb_est = tc = pc = vc = None
    if frag.do_not_estimate:
        warnings.append(frag.do_not_estimate)
    else:
        s = _group_sum(frag, "tb", warnings, local_refits)
        if s is not None:
            s += _interaction_sum(frag, "tb", local_refits)
            tb_est = s / (n ** TB_A + TB_B) + TB_C

        critical_refusal = _critical_interaction_refusal(frag)
        if critical_refusal:
            warnings.append(critical_refusal)
        else:
            tb_for_tc = tb if tb is not None else tb_est
            s = _group_sum(frag, "tc", warnings, local_refits)
            if s is not None and tb_for_tc is not None:
                s += _interaction_sum(frag, "tc", local_refits)
                if s > 0:
                    tc = tb_for_tc * (TC_B + 1.0 / (TC_A + s ** TC_C))
                else:
                    warnings.append("tc: non-positive group sum; outside model range")

            s = _group_sum(frag, "pc", warnings, local_refits)
            if s is not None:
                s += _interaction_sum(frag, "pc", local_refits)
                denom = PC_A + s
                if denom > 0:
                    pc = Descriptors.MolWt(mol) ** PC_B / denom ** 2
                else:
                    warnings.append("pc: non-positive denominator; outside model range")

            s = _group_sum(frag, "vc", warnings, local_refits)
            if s is not None:
                s += _interaction_sum(frag, "vc", local_refits)
                vc = s * n ** (-VC_A) + VC_B

    return NannoolalResult(
        smiles=smiles if smiles is not None else Chem.MolToSmiles(mol),
        n_atoms=n,
        n_hydrogens=frag.n_hydrogens,
        molar_mass=Descriptors.MolWt(mol),
        tb_K=tb_est,
        tc_K=tc,
        pc_kPa=pc,
        vc_cm3_mol=vc,
        tb_source="provided" if tb is not None else "estimated",
        groups=dict(frag.groups),
        corrections=dict(frag.corrections),
        interaction_classes=list(frag.interaction_classes),
        warnings=warnings,
    )


def boiling_point(smiles: str) -> float | None:
    """Normal boiling point [K]."""
    return estimate(smiles).tb_K


def critical_temperature(smiles: str, tb: float | None = None) -> float | None:
    """Critical temperature [K]; pass an experimental Tb [K] if available."""
    return estimate(smiles, tb=tb).tc_K


def critical_pressure(smiles: str) -> float | None:
    """Critical pressure [kPa]."""
    return estimate(smiles).pc_kPa


def critical_volume(smiles: str) -> float | None:
    """Critical volume [cm3/mol]."""
    return estimate(smiles).vc_cm3_mol


# ---------------------------------------------------------------------------
# Vapor pressure ([3])
# ---------------------------------------------------------------------------

# Pure-carbon skeleton groups (chain/ring/aromatic/unsaturated C); used to
# recognize "mono-functional alcohols" in the sense of eq. (9) of [3].
_HC_SKELETON_GROUPS = frozenset(
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
     61, 62, 63, 64, 87, 88, 89, 95, 214})
_ALKANOL_GROUPS = frozenset({33, 34, 35, 36})


def _is_monofunctional_alkanol(frag: Fragmentation) -> bool:
    n_oh = sum(f for g, f in frag.groups.items() if g in _ALKANOL_GROUPS)
    others = set(frag.groups) - _ALKANOL_GROUPS
    return n_oh == 1 and not frag.si_atoms and others <= _HC_SKELETON_GROUPS


_ALCOHOL_F_A = 0.37704   # eq. (8-6) of the thesis (= eq. (9) of [3], deprinted)


def _alcohol_correction(t_K: float) -> float:
    """Mono-alcohol low-pressure bowing term added to log10(Ps/atm)."""
    if t_K <= 92.0:              # saturated limit; alcohols are solid here
        return -_ALCOHOL_F_A
    u = (201.0 / (t_K - 91.0)) ** 7
    if u > 700.0:                # exp overflow guard; same limit
        return -_ALCOHOL_F_A
    return _ALCOHOL_F_A * (2.0 / (1.0 + math.exp(u)) - 1.0)


def _alcohol_correction_dT(t_K: float) -> float:
    """d f / dT of the bowing term (for the heat of vaporization)."""
    if t_K <= 92.0:
        return 0.0
    u = (201.0 / (t_K - 91.0)) ** 7
    if u > 700.0:
        return 0.0
    eu = math.exp(u)
    return _ALCOHOL_F_A * 2.0 * eu * 7.0 * u / ((t_K - 91.0) * (1.0 + eu) ** 2)


def _db_group_sum(frag: Fragmentation, warnings: list) -> float | None:
    """sum(N_i C_i) over first-order groups + corrections for dB, or None."""
    total = 0.0
    for gid, freq in frag.groups.items():
        value = PSAT_DB_GROUPS.get(gid)
        if gid == 219:
            # local extension group; borrow the nearest published relative
            value = PSAT_DB_GROUPS[96]
            warnings.append(
                "psat: group 219 (saturated cyclic anhydride, local "
                "extension) has no published dB; using the dB of group 96")
        if value is None:
            warnings.append(
                f"psat: no published dB contribution for group {gid}; "
                f"vapor pressure not estimable")
            return None
        if gid == 91:
            warnings.append(
                "psat: the dB of group 91 is flagged questionable in [3]")
        total += freq * value * _DB_SCALE
    for si_env in frag.si_atoms:
        gid = _si_group_id(si_env, "db")   # Part-2 style C/H-count mapping
        value = PSAT_DB_GROUPS.get(gid)
        if value is None:
            warnings.append(
                f"psat: no published dB contribution for silicon group "
                f"{gid}; vapor pressure not estimable")
            return None
        total += value * _DB_SCALE
    for cid, freq in frag.corrections.items():
        value = PSAT_DB_CORRECTIONS.get(cid)
        if value is None:
            warnings.append(
                f"psat: correction {cid} has no published dB value; ignored")
            continue
        total += freq * value * _DB_SCALE
    return total


def _db_interaction_sum(frag: Fragmentation, warnings: list) -> float:
    """GI for dB = (1/n) * sum over ordered pairs of C(i-j) / (m - 1)."""
    classes = frag.interaction_classes
    m = len(classes)
    if m < 2:
        return 0.0
    num = 0.0
    warned_ee = False
    for i in range(m):
        for j in range(i + 1, m):
            pair = frozenset(classes[i] + classes[j])
            value = PSAT_DB_INTERACTIONS.get(pair)
            if value is None:
                continue  # missing interactions contribute zero, per [3]
            if pair == frozenset("E") and not warned_ee:
                warnings.append("psat: the epoxide-epoxide dB interaction "
                                "is flagged questionable in [3]")
                warned_ee = True
            num += 2.0 * value * _DB_SCALE
    return num / (frag.n_atoms * (m - 1))


@dataclass
class NannoolalPsatResult:
    """Vapor pressure curve from eq. (6) of [3]; db is the slope parameter.

    The curve is anchored at (tb_K, 1 atm); tb_source records whether the
    anchor was user-provided, back-calculated from a single (T, Ps) point,
    or estimated internally.  db/tb_K are None when not estimable.
    """
    smiles: str
    n_atoms: int
    molar_mass: float
    db: float | None
    tb_K: float | None
    tb_source: str                 # "provided" | "from psat point" | "estimated"
    groups: dict
    corrections: dict
    interaction_classes: list
    warnings: list
    alcohol_correction: bool = False   # mono-alcohol bowing term active

    def psat_kPa(self, t_K: float) -> float | None:
        """Saturation pressure [kPa] at t_K [K] (valid ~triple point..0.8 Tc)."""
        if self.db is None or self.tb_K is None or t_K <= 0.125 * self.tb_K:
            return None
        trb = t_K / self.tb_K
        exponent = (PSAT_B0 + self.db) * (trb - 1.0) / (trb - 0.125)
        if self.alcohol_correction:
            exponent += _alcohol_correction(t_K)
        return _P_ATM_KPA * 10.0 ** exponent

    def temperature_K(self, p_kPa: float) -> float | None:
        """Saturation temperature [K] at p_kPa [kPa] (inverse of psat_kPa)."""
        if self.db is None or self.tb_K is None or p_kPa <= 0:
            return None
        a = PSAT_B0 + self.db
        lg = math.log10(p_kPa / _P_ATM_KPA)
        if lg >= a:            # pressure beyond the pole of eq. (6)
            return None
        if not self.alcohol_correction:
            return self.tb_K * (lg / 8.0 - a) / (lg - a)
        # With the bowing term there is no closed form, but f(T) is a small
        # perturbation: iterate the exact closed-form inverse with f frozen
        # at the previous iterate.  |f'| <= ~0.006/K vs a base log10-slope
        # of 0.02-0.05/K, so the map contracts (factor < ~0.3) and reaches
        # machine precision in a handful of steps.
        t = self.tb_K * (lg / 8.0 - a) / (lg - a)     # uncorrected start
        for _ in range(30):
            lg_eff = lg - _alcohol_correction(t)
            if lg_eff >= a:        # beyond the pole of eq. (6)
                return None
            t_new = self.tb_K * (lg_eff / 8.0 - a) / (lg_eff - a)
            if abs(t_new - t) < 1e-10 * self.tb_K:
                return t_new
            t = t_new
        return t

    def dhvap_J_mol(self, t_K: float, dz_vap: float = 1.0) -> float | None:
        """Heat of vaporization [J/mol] at t_K (Appendix A of [3]).

        dz_vap is the compressibility-factor difference of the coexisting
        phases (~0.95-1 well below Tb, ~0.9 at Tb); the default 1.0 is the
        ideal-gas/zero-liquid-volume limit and overestimates by ~5-10 %.
        """
        if self.db is None or self.tb_K is None or t_K <= 0:
            return None
        a = PSAT_B0 + self.db
        dh = (56.0 * math.log(10.0) * _R_GAS * a * self.tb_K * dz_vap
              / (self.tb_K / t_K - 8.0) ** 2)
        if self.alcohol_correction:
            # d(ln Ps)/dT gains ln(10) f'(T); via Clausius-Clapeyron this
            # adds R T^2 dz ln(10) f'(T) (raises dHvap at low T, as H-bonded
            # liquids should)
            dh += (_R_GAS * t_K ** 2 * dz_vap * math.log(10.0)
                   * _alcohol_correction_dT(t_K))
        return dh

    def summary(self) -> str:
        def fmt(v, unit, nd=4):
            return f"{v:.{nd}f} {unit}" if v is not None else "n/a"
        lines = [
            f"SMILES        : {self.smiles}",
            f"heavy atoms   : {self.n_atoms}   M = {self.molar_mass:.2f} g/mol",
            f"groups        : {dict(sorted(self.groups.items(), key=lambda kv: str(kv[0])))}",
            f"corrections   : {dict(sorted(self.corrections.items()))}",
            f"interactions  : {''.join(sorted(self.interaction_classes)) or '-'}",
            f"dB            : {fmt(self.db, '')}",
            f"Tb anchor ({self.tb_source}): {fmt(self.tb_K, 'K', 2)}",
        ]
        lines += [f"warning       : {w}" for w in self.warnings]
        return "\n".join(lines)


def estimate_psat(smiles: str, tb: float | None = None,
                  psat_point: tuple[float, float] | None = None,
                  ) -> NannoolalPsatResult:
    """Estimate the vapor pressure curve of a molecule given as SMILES.

    Anchor priority: experimental `tb` [K] if given; else back-calculation
    from `psat_point` = (T [K], Ps [kPa]) per Appendix A of [3]; else the
    internal Tb estimate (warned: slope and anchor are then both estimated).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise NannoolalError(f"could not parse SMILES {smiles!r}")
    return estimate_psat_from_mol(mol, tb=tb, psat_point=psat_point,
                                  smiles=smiles)


def estimate_psat_from_mol(mol: Chem.Mol, tb: float | None = None,
                           psat_point: tuple[float, float] | None = None,
                           smiles: str | None = None) -> NannoolalPsatResult:
    frag = fragment(mol)
    warnings: list = list(frag.notes)

    db = None
    alcohol = False
    if frag.do_not_estimate:
        warnings.append(frag.do_not_estimate)
    else:
        s = _db_group_sum(frag, warnings)
        if s is not None:
            db = s + _db_interaction_sum(frag, warnings) - PSAT_DB_OFFSET
            if _is_monofunctional_alkanol(frag):
                alcohol = True
                warnings.append(
                    "psat: mono-functional alcohol; low-pressure bowing "
                    "correction applied per eq. (8-6) of the 2006 thesis "
                    "(eq. (9) of [3] is a misprint of it; see module notes)")

    tb_source = "provided"
    if tb is None:
        if db is not None and psat_point is not None:
            t_pt, p_pt = psat_point
            a = PSAT_B0 + db
            lg = math.log10(p_pt / _P_ATM_KPA)
            if alcohol:
                lg -= _alcohol_correction(t_pt)
            tb = t_pt * (lg - a) / (lg / 8.0 - a)
            tb_source = "from psat point"
        else:
            tb_source = "estimated"
            if db is not None:
                s = _group_sum(frag, "tb", warnings)
                if s is not None:
                    s += _interaction_sum(frag, "tb")
                    tb = s / (frag.n_atoms ** TB_A + TB_B) + TB_C
                    warnings.append(
                        "psat: anchored at the internally estimated Tb; "
                        "slope and anchor errors compound -- prefer an "
                        "experimental tb or psat_point")

    return NannoolalPsatResult(
        smiles=smiles if smiles is not None else Chem.MolToSmiles(mol),
        n_atoms=frag.n_atoms,
        molar_mass=Descriptors.MolWt(mol),
        db=db,
        tb_K=tb,
        tb_source=tb_source,
        groups=dict(frag.groups),
        corrections=dict(frag.corrections),
        interaction_classes=list(frag.interaction_classes),
        warnings=warnings,
        alcohol_correction=alcohol,
    )


def vapor_pressure(smiles: str, t_K: float,
                   tb: float | None = None) -> float | None:
    """Saturation pressure [kPa] at t_K; pass an experimental Tb [K] if available."""
    return estimate_psat(smiles, tb=tb).psat_kPa(t_K)


def acentric_factor(smiles: str, tb: float | None = None,
                    tc: float | None = None,
                    pc_kPa: float | None = None) -> float | None:
    """Acentric factor omega = -log10(Ps(0.7 Tc)/Pc) - 1.

    Experimental values are used where supplied (tb [K] anchors the slope,
    tc [K]/pc_kPa fix the evaluation point); anything missing falls back to
    this module's own estimates.  Tr = 0.7 lies just above the Tb anchor for
    typical organics, well inside the Psat model's validity (Tr <= 0.75-0.8).
    """
    if tc is None or pc_kPa is None:
        r = estimate(smiles, tb=tb)
        tc = tc if tc is not None else r.tc_K
        pc_kPa = pc_kPa if pc_kPa is not None else r.pc_kPa
    if tc is None or pc_kPa is None or pc_kPa <= 0:
        return None
    ps = estimate_psat(smiles, tb=tb).psat_kPa(0.7 * tc)
    if ps is None or ps <= 0:
        return None
    return -math.log10(ps / pc_kPa) - 1.0


def _visc_sum(frag: Fragmentation, groups_table: dict, corrections_table: dict,
              scale: float, label: str, warnings: list) -> float | None:
    """sum(N_i C_i) over first-order groups + corrections for dBv or Tv."""
    total = 0.0
    for gid, freq in frag.groups.items():
        value = groups_table.get(gid)
        if gid == 219:
            # local extension group; borrow the nearest published relative
            value = groups_table.get(96)
            warnings.append(
                f"visc: group 219 (saturated cyclic anhydride, local "
                f"extension) has no published {label}; using group 96")
        if value is None:
            warnings.append(
                f"visc: no published {label} contribution for group {gid}; "
                f"viscosity not estimable")
            return None
        total += freq * value * scale
    for si_env in frag.si_atoms:
        gid = _si_group_id(si_env, "db")   # Part-2 style C/H-count mapping
        value = groups_table.get(gid)
        if value is None:
            warnings.append(
                f"visc: no published {label} contribution for silicon group "
                f"{gid}; viscosity not estimable")
            return None
        total += value * scale
    for cid, freq in frag.corrections.items():
        value = corrections_table.get(cid)
        if value is None:
            warnings.append(
                f"visc: correction {cid} has no published {label} value; "
                f"ignored")
            continue
        total += freq * value * scale
    return total


def _visc_interaction_sum(frag: Fragmentation, table: dict, f_gi: float,
                          scale: float, label: str, warnings: list) -> float:
    """GI = (f/n) * sum over ordered pairs of C(i-j)/(m-1); f=1 dBv, f=2 Tv."""
    classes = frag.interaction_classes
    m = len(classes)
    if m < 2:
        return 0.0
    num = 0.0
    warned_gg = False
    for i in range(m):
        for j in range(i + 1, m):
            pair = frozenset(classes[i] + classes[j])
            value = table.get(pair)
            if value is None:
                continue  # missing interactions contribute zero, per [4]
            if pair == frozenset("G") and not warned_gg:
                warnings.append(f"visc: the ketone-ketone {label} interaction "
                                f"is flagged questionable in [4]")
                warned_gg = True
            num += 2.0 * value * scale
    return f_gi * num / (frag.n_atoms * (m - 1))


@dataclass
class NannoolalViscosityResult:
    """Saturated liquid viscosity curve from eq. (6) of [4].

    ln(eta/1.3 cP) = -dBv (T - Tv)/(T - Tv/16); Tv is the temperature where
    eta = 1.3 cP.  tv_source records the anchor pedigree: back-calculated
    from one experimental (T, eta) point (best, ~3.4 % in [4]), estimated
    from a provided experimental Tb (~15 % in [4]), or estimated from the
    internal Tb estimate (slope and anchor errors compound).  Validity is
    roughly the triple point to Tr 0.75-0.8; expect larger deviations for
    high rotational symmetry, first members of a series, and small
    carboxylic acids (vapor-phase dimerization).
    """
    smiles: str
    n_atoms: int
    molar_mass: float
    dbv: float | None
    tv_K: float | None
    tv_source: str        # "from viscosity point" | "estimated (provided Tb)"
                          # | "estimated (internal Tb)"
    tb_K: float | None
    groups: dict
    corrections: dict
    interaction_classes: list
    warnings: list

    def viscosity_mPa_s(self, t_K: float) -> float | None:
        """Saturated liquid viscosity [mPa s] at t_K [K]."""
        if (self.dbv is None or self.tv_K is None
                or t_K <= self.tv_K / VISC_S):
            return None
        return VISC_ETA_REF_CP * math.exp(
            -self.dbv * (t_K - self.tv_K) / (t_K - self.tv_K / VISC_S))

    def viscosity_Pa_s(self, t_K: float) -> float | None:
        eta = self.viscosity_mPa_s(t_K)
        return None if eta is None else eta * 1.0e-3

    def temperature_K(self, eta_mPa_s: float) -> float | None:
        """Temperature [K] where the liquid viscosity equals eta_mPa_s."""
        if self.dbv is None or self.tv_K is None or eta_mPa_s <= 0:
            return None
        lg = math.log(eta_mPa_s / VISC_ETA_REF_CP)
        if lg + self.dbv <= 0:      # beyond the eta -> exp(-dBv) floor
            return None
        return self.tv_K * (self.dbv + lg / VISC_S) / (lg + self.dbv)

    def summary(self) -> str:
        def fmt(v, unit, nd=4):
            return f"{v:.{nd}f} {unit}" if v is not None else "n/a"
        lines = [
            f"SMILES        : {self.smiles}",
            f"heavy atoms   : {self.n_atoms}   M = {self.molar_mass:.2f} g/mol",
            f"groups        : {dict(sorted(self.groups.items(), key=lambda kv: str(kv[0])))}",
            f"corrections   : {dict(sorted(self.corrections.items()))}",
            f"interactions  : {''.join(sorted(self.interaction_classes)) or '-'}",
            f"dBv           : {fmt(self.dbv, '')}",
            f"Tv anchor ({self.tv_source}): {fmt(self.tv_K, 'K', 2)}",
        ]
        lines += [f"warning       : {w}" for w in self.warnings]
        return "\n".join(lines)


def estimate_viscosity(smiles: str, tb: float | None = None,
                       visc_point: tuple[float, float] | None = None,
                       ) -> NannoolalViscosityResult:
    """Estimate the saturated liquid viscosity curve from SMILES (Part 4).

    Anchor priority: `visc_point` = (T [K], eta [mPa s]) back-calculates the
    reference temperature Tv from one experimental point (the anchored mode
    of [4], ~3.4 % AAD; Tb is then not needed); else Tv is estimated from
    eq. (8) using the experimental `tb` [K] if given, or the internal Tb
    estimate as a last resort (warned: slope and anchor errors compound).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise NannoolalError(f"could not parse SMILES {smiles!r}")
    return estimate_viscosity_from_mol(mol, tb=tb, visc_point=visc_point,
                                       smiles=smiles)


_VISC_ORTHO_CHELATE = None


def _visc_domain_refusal(mol: Chem.Mol, frag: Fragmentation) -> str | None:
    """Structure classes where eq. (6) fails and no anchor can save it
    (benchmarked 2026-07-12 on the VDI-PPDS set; rejections mirrored in
    hsu_method where applicable)."""
    global _VISC_ORTHO_CHELATE
    if _VISC_ORTHO_CHELATE is None:
        _VISC_ORTHO_CHELATE = [Chem.MolFromSmarts(s) for s in (
            '[OX2H1][c]:[c][CX3]=[OX1]',
            '[OX2H1][c]:[c][NX3](=[OX1])=[OX1]',
            '[OX2H1][c]:[c][NX3+](=[OX1])[OX1-]')]
    total_c = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    if 44 in frag.groups and total_c <= 6:
        return ("visc: carboxylic acid with <= 6 carbons refused: "
                "dimerization; [4] removed small acids from its own "
                "regression and reports large deviations up to hexanoic "
                "(trichloroacetic acid measured 780 %, anchored worse)")
    if any(mol.HasSubstructMatch(p) for p in _VISC_ORTHO_CHELATE):
        return ("visc: ortho-chelating aromatic refused (salicylate/"
                "o-nitrophenol type): the intramolecular H-bond removes the "
                "association the OH(a) contributions assume (200-260 % "
                "measured, anchoring does not recover the curve shape)")
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        hal = [nb.GetSymbol() for nb in atom.GetNeighbors()
               if nb.GetSymbol() in ('F', 'Cl', 'Br', 'I')]
        if len(hal) >= 3 and any(h in ('Br', 'I') for h in hal):
            return ("visc: heavy perhalomethane carbon (>= 3 halogens "
                    "incl. Br/I) refused per the caution in [4] "
                    "(CBr4 measured 642 %)")
    return None


def estimate_viscosity_from_mol(mol: Chem.Mol, tb: float | None = None,
                                visc_point: tuple[float, float] | None = None,
                                smiles: str | None = None,
                                ) -> NannoolalViscosityResult:
    frag = fragment(mol)
    warnings: list = list(frag.notes)

    refusal = _visc_domain_refusal(mol, frag)
    dbv = None
    if frag.do_not_estimate:
        warnings.append(frag.do_not_estimate)
    elif refusal:
        warnings.append(refusal)
    else:
        s = _visc_sum(frag, VISC_DBV_GROUPS, VISC_DBV_CORRECTIONS,
                      _DB_SCALE, "dBv", warnings)
        if s is not None:
            s += _visc_interaction_sum(frag, VISC_DBV_INTERACTIONS, 1.0,
                                       _DB_SCALE, "dBv", warnings)
            dbv = (s / (frag.n_atoms ** VISC_DBV_A + VISC_DBV_B)
                   + VISC_DBV_C)

    tv = None
    tv_source = "estimated (provided Tb)"
    if dbv is not None and visc_point is not None:
        t_pt, eta_pt = visc_point
        lg = math.log(eta_pt / VISC_ETA_REF_CP)
        # eq. (6) floor: eta(T->inf) = 1.3 exp(-dBv); the anchor viscosity
        # must lie above it for the inversion to have a solution.
        if lg + dbv > 0:
            tv = t_pt * (lg + dbv) / (dbv + lg / VISC_S)
            tv_source = "from viscosity point"
        else:
            warnings.append("visc: viscosity point below the exp(-dBv) "
                            "floor of eq. (6); anchor ignored")
    if tv is None and dbv is not None:
        if tb is None:
            tv_source = "estimated (internal Tb)"
            s = _group_sum(frag, "tb", warnings)
            if s is not None:
                s += _interaction_sum(frag, "tb")
                tb = s / (frag.n_atoms ** TB_A + TB_B) + TB_C
                warnings.append(
                    "visc: Tv anchored at the internally estimated Tb; "
                    "slope and anchor errors compound -- prefer an "
                    "experimental tb or visc_point")
        if tb is not None:
            s = _visc_sum(frag, VISC_TV_GROUPS, VISC_TV_CORRECTIONS,
                          1.0, "Tv", warnings)
            if s is not None:
                s += _visc_interaction_sum(frag, VISC_TV_INTERACTIONS, 2.0,
                                           1.0, "Tv", warnings)
                if s <= 0.0:
                    warnings.append(
                        "visc: negative Tv group sum; eq. (8) not evaluable")
                else:
                    tv = (VISC_TV_A * math.sqrt(tb)
                          + s ** VISC_TV_EXP
                          / (frag.n_atoms ** VISC_TV_C + VISC_TV_D)
                          - VISC_TV_E)

    return NannoolalViscosityResult(
        smiles=smiles if smiles is not None else Chem.MolToSmiles(mol),
        n_atoms=frag.n_atoms,
        molar_mass=Descriptors.MolWt(mol),
        dbv=dbv,
        tv_K=tv,
        tv_source=tv_source,
        tb_K=tb,
        groups=dict(frag.groups),
        corrections=dict(frag.corrections),
        interaction_classes=list(frag.interaction_classes),
        warnings=warnings,
    )


def liquid_viscosity(smiles: str, t_K: float, tb: float | None = None,
                     visc_point: tuple[float, float] | None = None,
                     ) -> float | None:
    """Saturated liquid viscosity [mPa s] at t_K [K] (convenience wrapper)."""
    return estimate_viscosity(smiles, tb=tb,
                              visc_point=visc_point).viscosity_mPa_s(t_K)


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Nannoolal group-contribution Tb/Tc/Pc/Vc estimation")
    parser.add_argument("smiles", help="molecule as SMILES")
    parser.add_argument("--tb", type=float, default=None,
                        help="experimental normal boiling point [K] "
                             "(used for the Tc and vapor pressure estimation)")
    parser.add_argument("--psat", type=float, default=None, metavar="T",
                        help="also print the vapor pressure [kPa] at T [K]")
    parser.add_argument("--visc", type=float, default=None, metavar="T",
                        help="also print the liquid viscosity [mPa s] at T [K]")
    parser.add_argument("--visc-point", type=float, nargs=2, default=None,
                        metavar=("T", "ETA"),
                        help="anchor the viscosity curve at one experimental "
                             "point (T [K], eta [mPa s])")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show fragmentation details")
    args = parser.parse_args(argv)

    try:
        result = estimate(args.smiles, tb=args.tb)
        psat_result = (estimate_psat(args.smiles, tb=args.tb)
                       if args.psat is not None else None)
        visc_result = (estimate_viscosity(
                           args.smiles, tb=args.tb,
                           visc_point=(tuple(args.visc_point)
                                       if args.visc_point else None))
                       if args.visc is not None else None)
    except NannoolalError as exc:
        parser.exit(1, f"error: {exc}\n")
    if args.verbose:
        print(result.summary())
    else:
        def fmt(v, unit, nd=1):
            return f"{v:.{nd}f} {unit}" if v is not None else "n/a"
        print(f"Tb = {fmt(result.tb_K, 'K')} ({result.tb_source})")
        print(f"Tc = {fmt(result.tc_K, 'K')}")
        print(f"Pc = {fmt(result.pc_kPa, 'kPa')}")
        print(f"Vc = {fmt(result.vc_cm3_mol, 'cm3/mol')}")
        for w in result.warnings:
            print(f"warning: {w}")
    if psat_result is not None:
        db = psat_result.db
        print(f"dB = {db:.6f}" if db is not None else "dB = n/a")
        p = psat_result.psat_kPa(args.psat)
        p_str = f"{p:.4g} kPa" if p is not None else "n/a"
        print(f"Psat({args.psat:g} K) = {p_str} "
              f"(Tb anchor: {psat_result.tb_source})")
        for w in psat_result.warnings:
            if w.startswith("psat:"):
                print(f"warning: {w}")
    if visc_result is not None:
        dbv = visc_result.dbv
        print(f"dBv = {dbv:.6f}" if dbv is not None else "dBv = n/a")
        eta = visc_result.viscosity_mPa_s(args.visc)
        eta_str = f"{eta:.4g} mPa s" if eta is not None else "n/a"
        print(f"eta({args.visc:g} K) = {eta_str} "
              f"(Tv anchor: {visc_result.tv_source})")
        for w in visc_result.warnings:
            if w.startswith("visc:"):
                print(f"warning: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
