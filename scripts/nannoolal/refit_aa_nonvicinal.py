"""Refit _AA_TC_NONVICINAL after the 35/36 chain-rule fix.

Per-compound: find the exact per-pair AA value x* that reproduces the ATN
Tc, then least-squares over the set.  Data: ATN Part-12 review Tc (corrected
JSON), PFDSim-validated / literature Tbs.
"""
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import nannoolal_method as nm

# name, smiles, Tb[K], Tc[K], unc[K]   (Tb: E = PFDSim-validated, L = lit)
DIOLS = [
    ("1,3-propanediol",   "OCCCO",        487.6, 720.0, 3.0),   # L
    ("1,3-butanediol",    "CC(O)CCO",     480.1, 690.0, 8.0),   # L
    ("2-Me-1,3-PD",       "OCC(C)CO",     486.6, 708.0, 2.0),   # L
    ("neopentyl glycol",  "OCC(C)(C)CO",  483.2, 687.0, 10.0),  # L
    ("1,4-butanediol",    "OCCCCO",       503.2, 725.0, 2.0),   # E
    ("1,5-pentanediol",   "OCCCCCO",      515.2, 731.0, 11.0),  # E
    ("1,6-hexanediol",    "OCCCCCCO",     523.2, 740.0, 3.0),   # E
    ("1,8-octanediol",    "OCCCCCCCCO",   552.0, 752.0, 11.0),  # E
    ("1,9-nonanediol",    "OCCCCCCCCCO",  561.2, 760.0, 11.0),  # E
    ("1,10-decanediol",   "OCCCCCCCCCCO", 570.2, 770.0, 12.0),  # E
]

GLYCOL_ETHERS = [  # validation only (A-D / D-D published pairwise + AA split)
    ("2-methoxyethanol",  "COCCO",        397.5, 579.6),
    ("1-MeO-2-propanol",  "COCC(C)O",     393.2, 579.8),
    ("DEG",               "OCCOCCO",      518.0, 753.0),
    ("DEGEE",             "CCOCCOCCO",    475.2, 670.0),
    ("TEG",               "OCCOCCOCCO",   559.0, 797.0),
    ("tetraEG",           "OCCOCCOCCOCCO",600.6, 800.0),
    ("4-oxa-1,7-heptanediol", "OCCCOCCCO",504.1, 705.0),
]

def tc_with_x(smiles, tb, x):
    nm._AA_TC_NONVICINAL = x
    return nm.estimate(smiles, tb=tb).tc_K

print("Per-diol exact x* (bisection) and residual at candidate values:")
xs = []
for name, smi, tb, tc, unc in DIOLS:
    lo, hi = -800.0, 800.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if tc_with_x(smi, tb, mid) < tc:   # larger x -> larger S -> LOWER Tc?
            hi = mid
        else:
            lo = mid
    # check direction: evaluate at ends
    t_lo, t_hi = tc_with_x(smi, tb, -800), tc_with_x(smi, tb, 800)
    xstar = 0.5 * (lo + hi)
    # bisection direction may be inverted; redo properly
    if not (min(t_lo, t_hi) <= tc <= max(t_lo, t_hi)):
        print(f"  {name:20s} Tc outside reachable range [{t_hi:.1f},{t_lo:.1f}]")
        continue
    lo, hi = -800.0, 800.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        tmid = tc_with_x(smi, tb, mid)
        if (tmid - tc) * (t_lo - tc) > 0:
            lo, t_lo = mid, tmid
        else:
            hi = mid
    xstar = 0.5 * (lo + hi)
    xs.append(xstar)
    grp = sorted(nm.estimate(smi, tb=tb).groups.items())
    print(f"  {name:20s} x* = {xstar:8.1f}   groups {grp}")

# least squares over the set: scan
best = None
for x in [i * 2.0 for i in range(-200, 201)]:
    sse = sum((tc_with_x(s, tb, x) - tc) ** 2 for _, s, tb, tc, _ in DIOLS)
    if best is None or sse < best[1]:
        best = (x, sse)
x0 = best[0]
for x in [x0 + i * 0.1 for i in range(-25, 26)]:
    sse = sum((tc_with_x(s, tb, x) - tc) ** 2 for _, s, tb, tc, _ in DIOLS)
    if sse < best[1]:
        best = (x, sse)
xfit = best[0]
print(f"\nLeast-squares x = {xfit:.2f}  (old fit, old fragmentation: 23.48)")
print(f"{'diol':20s} {'calc':>7s} {'ref':>7s} {'err':>7s}")
for name, smi, tb, tc, unc in DIOLS:
    t = tc_with_x(smi, tb, xfit)
    print(f"{name:20s} {t:7.1f} {tc:7.1f} {t - tc:+7.1f}  (+-{unc:.0f})")

print(f"\nGlycol ethers with x = {xfit:.2f} (validation):")
for name, smi, tb, tc in GLYCOL_ETHERS:
    t = tc_with_x(smi, tb, xfit)
    print(f"{name:22s} {t:7.1f} {tc:7.1f} {t - tc:+7.1f}")

# and the vicinal pairs must be untouched:
nm._AA_TC_NONVICINAL = xfit
for name, smi, tb, tc in [("MEG", "OCCO", 470.5, 720.0),
                          ("1,2-propanediol", "CC(O)CO", 460.8, 676.4)]:
    t = nm.estimate(smi, tb=tb).tc_K
    print(f"vicinal {name:18s} {t:7.1f} {tc:7.1f} {t - tc:+7.1f}")
