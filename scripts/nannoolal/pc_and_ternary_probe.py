"""Probe A: mono-alcohol Pc errors (is group 36-pc itself biased?)
Probe B: per-diol exact-fit AA-pc values, clustered by OH subtype
Probe C: ternary AAD Tc correction -- needed dS for the DEG family
"""
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import nannoolal_method as nm
from rdkit import Chem

# ---------- A: mono-alcohols (no AA interaction; isolates the group value)
MONO = [  # name, smiles, Pc_ref[kPa]  (Perry 2-106 / ATN)
    ("methanol",        "CO",         8215.0),
    ("ethanol",         "CCO",        6148.0),
    ("1-propanol",      "CCCO",       5175.0),
    ("2-methyl-1-propanol", "CC(C)CO", 4295.0),
    ("1-butanol",       "CCCCO",      4414.0),   # newly 36
    ("3-methyl-1-butanol", "CC(C)CCO", 3930.0),  # newly 36
    ("1-pentanol",      "CCCCCO",     3897.0),   # 35
    ("1-hexanol",       "CCCCCCO",    3417.0),   # 35
    ("1-octanol",       "CCCCCCCCO",  2777.0),   # 35
]
print("A) mono-alcohol Pc errors (group value only, no AA interaction):")
for name, smi, ref in MONO:
    r = nm.estimate(smi)
    g = [g for g in r.groups if g in (33, 34, 35, 36)]
    print(f"  {name:22s} grp{g}  {100*(r.pc_kPa-ref)/ref:+6.1f} %")

# ---------- B: per-diol exact-fit AA-pc
def pc_with_aa(smiles, x):
    """Pc [kPa] with the AA pc pair forced to x (*1e-4); rest published,
    subtype-aware Tc irrelevant here."""
    mol = Chem.MolFromSmiles(smiles)
    frag = nm.fragment(mol)
    w = []
    s = nm._group_sum(frag, "pc", w, local_refits=False)
    classes = frag.interaction_classes
    m = len(classes)
    gi = 0.0
    for i in range(m):
        for j in range(i + 1, m):
            pair = frozenset(classes[i] + classes[j])
            if pair == frozenset("A"):
                v = x
            else:
                e = nm.INTERACTION_CONTRIBUTIONS.get(pair)
                v = e[2] if e else None
            if v is None:
                continue
            gi += 2.0 * v * 1e-4
    if m >= 2:
        s += gi / (frag.n_atoms * (m - 1))
    from rdkit.Chem import Descriptors
    denom = nm.PC_A + s
    return Descriptors.MolWt(mol) ** nm.PC_B / denom ** 2

DIOLS_PC = [  # name, smiles, Pc_ref[kPa], subtype tag
    ("MEG",             "OCCO",         8200.0, "36+36"),   # Nikitin
    ("1,3-PD",          "OCCCO",        6400.0, "36+36"),
    ("1,4-BD",          "OCCCCO",       5800.0, "36+36"),
    ("2-Me-1,3-PD",     "OCC(C)CO",     5400.0, "36+36"),
    ("neopentyl glycol","OCC(C)(C)CO",  4200.0, "36+36"),
    ("1,2-PD",          "CC(O)CO",      6000.0, "34+36"),
    ("1,3-BD",          "CC(O)CCO",     5000.0, "34+36"),
    ("1,5-PD",          "OCCCCCO",      4000.0, "35+35"),
    ("1,6-HD",          "OCCCCCCO",     4000.0, "35+35"),
    ("1,8-OD",          "OCCCCCCCCO",   2700.0, "35+35"),
    ("1,9-ND",          "OCCCCCCCCCO",  2400.0, "35+35"),
    ("1,10-DD",         "OCCCCCCCCCCO", 2200.0, "35+35"),
    ("DEG",             "OCCOCCO",      4770.0, "36+36+D"),
]
print("\nB) per-diol exact-fit AA-pc (*1e-4; published single constant -5.60):")
for name, smi, ref, tag in DIOLS_PC:
    lo, hi = -400.0, 400.0
    p_lo, p_hi = pc_with_aa(smi, lo), pc_with_aa(smi, hi)
    if not (min(p_lo, p_hi) <= ref <= max(p_lo, p_hi)):
        print(f"  {name:18s} [{tag}] unreachable "
              f"(range {p_hi:.0f}..{p_lo:.0f} vs {ref:.0f})")
        continue
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (pc_with_aa(smi, mid) - ref) * (p_lo - ref) > 0:
            lo, p_lo = mid, pc_with_aa(smi, mid)
        else:
            hi = mid
    err_pub = 100 * (pc_with_aa(smi, -5.6023) - ref) / ref
    print(f"  {name:18s} [{tag}]  x* = {0.5*(lo+hi):+8.1f}   "
          f"(err at published: {err_pub:+6.1f} %)")

# ---------- C: ternary AAD Tc -- needed extra S and triple-count scaling
def tc_parts(smiles, tb):
    mol = Chem.MolFromSmiles(smiles)
    frag = nm.fragment(mol)
    w = []
    s = nm._group_sum(frag, "tc", w) + nm._interaction_sum(frag, "tc")
    return frag, s

def tc_from_s(s, tb):
    return tb * (nm.TC_B + 1.0 / (nm.TC_A + s ** nm.TC_C))

FAMILY = [  # name, smiles, Tb, Tc_ref, n, m, AAD_triples
    ("DEG",     "OCCOCCO",        518.0, 753.0, 7, 3, 1),
    ("4-oxa",   "OCCCOCCCO",      504.1, 705.0, 9, 3, 1),
    ("TEG",     "OCCOCCOCCO",     559.0, 797.0, 10, 4, 2),
    ("tetraEG", "OCCOCCOCCOCCO",  600.6, 800.0, 13, 5, 3),
]
print("\nC) ternary AAD probe (needed dS to hit reference Tc):")
for name, smi, tb, tcref, n, m, ntri in FAMILY:
    frag, s = tc_parts(smi, tb)
    lo, hi = -1.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if tc_from_s(s + mid, tb) > tcref:   # larger S -> lower Tc
            lo = mid
        else:
            hi = mid
    ds = 0.5 * (lo + hi)
    # implied ternary constant under GI-style norm: dS = c * ntri/(n*(m-1))
    c = ds * n * (m - 1) / ntri
    print(f"  {name:8s} dS needed = {ds:+8.4f}   "
          f"triples={ntri} n={n} m={m}  -> c = {c:+8.3f}")
