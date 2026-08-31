"""Before/after impact of the 35/36 chain-rule fix on the benchmark databases.

Only compounds whose fragmentation changed can move; everything else is
bit-identical.  Sweep Perry 2-106 + ATN Part-12 + Perry 2-10, find the flip
set, and compare errors under the old (heavy-atom) and new (C/Si) rules.
"""
from pathlib import Path
import json
import sys
import warnings as pywarn
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import nannoolal_method as nm
from chemicals.identifiers import search_chemical

NEW_RULE = nm._c_si_chain_reaches


def old_rule(atom, target):  # pre-fix behavior: heavy atoms, O included
    def dfs(a, visited, depth):
        if depth >= target:
            return True
        for nb in a.GetNeighbors():
            if nb.GetIdx() not in visited:
                visited.add(nb.GetIdx())
                if dfs(nb, visited, depth + 1):
                    return True
                visited.discard(nb.GetIdx())
        return False
    return dfs(atom, {atom.GetIdx()}, 1)


perry = json.load(open(str(REPO_ROOT / "data/perry_properties.json")))["chemicals"]
atn = json.load(open(str(REPO_ROOT / "data/acs_jced_5b00571_table1.json")))["chemicals"]
t210 = json.load(open(str(REPO_ROOT / "data/perry_table_2_10_vapor_pressure.json")))["chemicals"]

EXCLUDE = {"302-17-0"}  # chloral hydrate (decomposes) - excluded upstream too

refs = {}
for cas, row in perry.items():
    cc = row.get("critical_constants") or {}
    refs.setdefault(cas, {})["name"] = row.get("name")
    if cc.get("Tc_K"):
        refs[cas]["tc"] = cc["Tc_K"]
    if cc.get("Pc_MPa"):
        refs[cas]["pc"] = cc["Pc_MPa"] * 1e3
    if cc.get("Vc_m3_per_kmol"):
        refs[cas]["vc"] = cc["Vc_m3_per_kmol"] * 1e3
for cas, row in atn.items():  # ATN preferred on overlap
    cp = row.get("critical_properties") or {}
    refs.setdefault(cas, {})["name"] = row.get("article_name")
    if cp.get("Tc_K"):
        refs[cas]["tc"] = cp["Tc_K"]
    if cp.get("Pc_MPa"):
        refs[cas]["pc"] = cp["Pc_MPa"] * 1e3
for cas, row in t210.items():
    if row.get("Tb_K"):
        refs.setdefault(cas, {})["tb"] = row["Tb_K"]
        refs[cas].setdefault("name", row.get("name"))
        if row.get("Tm_K"):
            refs[cas]["tm"] = row["Tm_K"]

flips, n_ok = [], 0
pywarn.filterwarnings("ignore")
for cas, ref in sorted(refs.items()):
    if cas in EXCLUDE:
        continue
    try:
        smiles = search_chemical(cas).smiles
        if not smiles:
            continue
    except Exception:
        continue
    try:
        nm._c_si_chain_reaches = NEW_RULE
        r_new = nm.estimate(smiles, tb=ref.get("tb"))
        nm._c_si_chain_reaches = old_rule
        r_old = nm.estimate(smiles, tb=ref.get("tb"))
    except Exception:
        continue
    finally:
        nm._c_si_chain_reaches = NEW_RULE
    n_ok += 1
    if r_new.groups != r_old.groups:
        flips.append((cas, ref, r_old, r_new))

print(f"fragmentable compounds swept: {n_ok}; fragmentation changed: {len(flips)}\n")

def line(name, ref, old, new, prop, refkey, unit, pct=False):
    rv = ref.get(refkey)
    ov, nv = getattr(old, prop), getattr(new, prop)
    if rv is None or ov is None or nv is None:
        return None
    if pct:
        eo, en = 100 * (ov - rv) / rv, 100 * (nv - rv) / rv
    else:
        eo, en = ov - rv, nv - rv
    return eo, en

agg = {p: [] for p in ("tb", "tc", "pc", "vc")}
hdr = f"{'compound':28s} {'prop':>4s} {'old err':>9s} {'new err':>9s}"
print(hdr); print("-" * len(hdr))
for cas, ref, old, new in flips:
    name = (ref.get("name") or cas)[:28]
    # skip sublimers for Tb, mirroring the original benchmark filter
    tb_ok = not (ref.get("tm") and ref.get("tb") and ref["tm"] >= ref["tb"] - 2)
    for prop, attr, key, pct in (("tb", "tb_K", "tb", False),
                                 ("tc", "tc_K", "tc", False),
                                 ("pc", "pc_kPa", "pc", True),
                                 ("vc", "vc_cm3_mol", "vc", True)):
        if prop == "tb" and not tb_ok:
            continue
        out = line(name, ref, old, new, attr, key, "", pct)
        if out is None:
            continue
        eo, en = out
        agg[prop].append((eo, en))
        u = "%" if pct else "K"
        print(f"{name:28s} {prop:>4s} {eo:+8.1f}{u} {en:+8.1f}{u}")

print()
for prop, pairs in agg.items():
    if not pairs:
        continue
    aad_o = sum(abs(a) for a, _ in pairs) / len(pairs)
    aad_n = sum(abs(b) for _, b in pairs) / len(pairs)
    bias_o = sum(a for a, _ in pairs) / len(pairs)
    bias_n = sum(b for _, b in pairs) / len(pairs)
    u = "%" if prop in ("pc", "vc") else "K"
    print(f"{prop}: flip-set n={len(pairs):2d}  AAD {aad_o:6.2f} -> {aad_n:6.2f} {u}"
          f"   bias {bias_o:+6.2f} -> {bias_n:+6.2f} {u}")
