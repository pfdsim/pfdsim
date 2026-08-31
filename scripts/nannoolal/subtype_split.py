"""Test the 3-regime AA Tc scheme keyed on OH group subtype (36/34/35)."""
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import nannoolal_method as nm

# Monkeypatch _interaction_sum's AA handling via the module constants is not
# possible (subtype scheme not implemented yet), so compute Tc manually here.
def tc_manual(smiles, tb, aa_value):
    """Tc with the AA tc pair forced to aa_value (*1e-3); other pairs published."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    frag = nm.fragment(mol)
    w = []
    s = nm._group_sum(frag, "tc", w, local_refits=False)
    classes = frag.interaction_classes
    m = len(classes)
    gi = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            pair = frozenset(classes[i] + classes[j])
            if pair == frozenset("A"):
                v = aa_value
            else:
                e = nm.INTERACTION_CONTRIBUTIONS.get(pair)
                v = e[1] if e else None
            if v is None:
                continue
            gi += 2.0 * v * 1e-3
    s += gi / (frag.n_atoms * (m - 1)) if m >= 2 else 0.0
    return tb * (nm.TC_B + 1.0 / (nm.TC_A + s ** nm.TC_C))

PUB = -434.8568
X34 = -250.0
X35 = 0.0

SETS = {
    "36+36 -> published -434.9": [
        ("MEG",              "OCCO",         470.5, 720.0),
        ("1,3-propanediol",  "OCCCO",        487.6, 720.0),
        ("2-Me-1,3-PD",      "OCC(C)CO",     486.6, 708.0),
        ("neopentyl glycol", "OCC(C)(C)CO",  483.2, 687.0),
        ("1,4-butanediol",   "OCCCCO",       503.2, 725.0),
        ("DEG (has AD too)", "OCCOCCO",      518.0, 753.0),
    ],
    "34-involved -> -250": [
        ("1,2-propanediol",  "CC(O)CO",      460.8, 676.4),
        ("1,3-butanediol",   "CC(O)CCO",     480.1, 690.0),
        ("1,2-butanediol",   "CCC(O)CO",     469.2, 694.0),  # holdout
    ],
    "35-involved -> 0": [
        ("1,5-pentanediol",  "OCCCCCO",      515.2, 731.0),
        ("1,6-hexanediol",   "OCCCCCCO",     523.2, 740.0),
        ("1,8-octanediol",   "OCCCCCCCCO",   552.0, 752.0),
        ("1,9-nonanediol",   "OCCCCCCCCCO",  561.2, 760.0),
        ("1,10-decanediol",  "OCCCCCCCCCCO", 570.2, 770.0),
        ("4-oxa-1,7-heptanediol", "OCCCOCCCO", 504.1, 705.0),
        ("TEG",              "OCCOCCOCCO",   559.0, 797.0),
        ("tetraEG",          "OCCOCCOCCOCCO", 600.6, 800.0),
    ],
}
XV = {"36+36 -> published -434.9": PUB, "34-involved -> -250": X34,
      "35-involved -> 0": X35}

for label, rows in SETS.items():
    x = XV[label]
    errs = []
    print(f"\n{label}:")
    for name, smi, tb, tc in rows:
        t = tc_manual(smi, tb, x)
        errs.append(t - tc)
        print(f"  {name:22s} {t:7.1f} vs {tc:7.1f}  {t - tc:+7.1f}")
    aad = sum(abs(e) for e in errs) / len(errs)
    print(f"  AAD {aad:.1f} K, bias {sum(errs)/len(errs):+.1f} K")

# LSQ refine X34 and X35 on their fit members (holdouts excluded)
def lsq(rows, lo, hi):
    best = None
    x = lo
    while x <= hi:
        sse = sum((tc_manual(s, tb, x) - tc) ** 2 for _, s, tb, tc in rows)
        if best is None or sse < best[1]:
            best = (x, sse)
        x += 2.0
    return best[0]

fit34 = lsq(SETS["34-involved -> -250"][:2], -500, 100)
fit35 = lsq(SETS["35-involved -> 0"][:5], -400, 400)   # terminal diols only
print(f"\nLSQ: x(34) = {fit34:.1f} (on 1,2-PD + 1,3-BD)   "
      f"x(35) = {fit35:.1f} (on C5-C10 terminal diols)")
