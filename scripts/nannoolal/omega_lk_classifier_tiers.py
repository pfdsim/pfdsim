"""Acentric factor: NN vs LK vs mean vs polarity-classifier, tiers A/B/C.

The script behind the omega-rung policy in compaction_notes.md:
  - tier A (real Tb/Tc/Pc): classifier best (MAD 0.0042), LK 0.0043, NN 0.0069
  - tiers B/C (estimated criticals): all within noise of LK (~0.044);
    classifier-flagged associators hit MAD ~0.2 -> quality floor 0.3
Classifier: score = HBA + 2*HBD (Lipinski); polar if score > 3 and
score/n_heavy > 0.3 (ratio cut separates nitroaromatics from diesters).
References: Perry 2-106 omega + omega derived from chemicals Wagner fits
at 0.7 Tc; combined and deduped by canonical SMILES.
"""
from pathlib import Path
import json
import math
import sys
import warnings

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import nannoolal_method as nm
from chemicals.identifiers import search_chemical
from chemicals.acentric import LK_omega
from chemicals.vapor_pressure import (Psat_data_WagnerMcGarry,
    Psat_data_WagnerPoling, Wagner, Wagner_original)
from rdkit import Chem
from rdkit.Chem import Lipinski

P_ATM = 101.325

def bisect_T(f, target, lo, hi):
    flo, fhi = f(lo), f(hi)
    if not (flo <= target <= fhi):
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

_sc = {}
def smi(k):
    if k not in _sc:
        try:
            _sc[k] = search_chemical(k).smiles
        except Exception:
            _sc[k] = None
    return _sc[k]

rows = {}
def add(s, tb, tc, pc, w_ref):
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return
    key = Chem.MolToSmiles(mol)
    if key in rows:
        return
    try:
        wA_nn = nm.acentric_factor(s, tb=tb, tc=tc, pc_kPa=pc)
        wA_lk = LK_omega(tb, tc, pc * 1000.0)
        eB = nm.estimate(s, tb=tb)
        wB_nn = nm.acentric_factor(s, tb=tb)
        wB_lk = LK_omega(tb, eB.tc_K, eB.pc_kPa * 1000.0)
        eC = nm.estimate(s)
        wC_nn = nm.acentric_factor(s)
        wC_lk = LK_omega(eC.tb_K, eC.tc_K, eC.pc_kPa * 1000.0)
    except Exception:
        return
    if None in (wA_nn, wB_nn, wC_nn):
        return
    score = Lipinski.NumHAcceptors(mol) + 2 * Lipinski.NumHDonors(mol)
    polar = score > 3 and score / mol.GetNumHeavyAtoms() > 0.3
    rows[key] = (w_ref, (wA_nn, wA_lk), (wB_nn, wB_lk), (wC_nn, wC_lk), polar)

perry = json.load(open(str(REPO_ROOT / "data/perry_properties.json")))["chemicals"]
for cas, row in perry.items():
    if cas == "302-17-0":
        continue
    cc = row.get("critical_constants") or {}
    vps = row.get("vapor_pressure") or []
    if not vps or not cc.get("Tc_K") or not cc.get("Pc_MPa") \
            or cc.get("omega") is None:
        continue
    rec = vps[0]
    c = (list(rec["coefficients"]) + [0.0] * 5)[:5]
    if c[4] == 0:
        c[4] = 1.0
    tlo, thi = rec["T_min_K"], rec["T_max_K"]
    if not tlo or not thi or thi <= tlo:
        continue
    s = smi(cas)
    if not s:
        continue
    def pref(T, c=c):
        return math.exp(c[0] + c[1]/T + c[2]*math.log(T)
                        + c[3]*T**c[4]) / 1000.0
    tb = bisect_T(pref, P_ATM, tlo, thi)
    if tb:
        add(s, tb, cc["Tc_K"], cc["Pc_MPa"] * 1e3, cc["omega"])

def wag(bank, fn):
    for cas, r in bank.iterrows():
        s = smi(cas)
        if not s:
            continue
        def pref(T, r=r, fn=fn):
            return fn(T, r["Tc"], r["Pc"], r["A"], r["B"],
                      r["C"], r["D"]) / 1000.0
        thi = (min(r["Tmax"], r["Tc"] - 0.1) if "Tmax" in r else r["Tc"] - 0.1)
        tb = bisect_T(pref, P_ATM, r["Tmin"], thi)
        t07 = 0.7 * r["Tc"]
        if tb and r["Tmin"] <= t07 <= thi:
            w_ref = -math.log10(pref(t07) / (r["Pc"] / 1000.0)) - 1.0
            add(s, tb, r["Tc"], r["Pc"] / 1000.0, w_ref)
wag(Psat_data_WagnerMcGarry, Wagner_original)
wag(Psat_data_WagnerPoling, Wagner)

data = list(rows.values())
print(f"combined deduped set, all tiers computable: n = {len(data)}\n")

def stats(errs):
    ae = sorted(abs(x) for x in errs)
    return sum(ae)/len(ae), ae[len(ae)//2], ae[int(0.9*len(ae))]

for tier, idx in (("A: real Tb,Tc,Pc", 1), ("B: real Tb only", 2),
                  ("C: all estimated", 3)):
    print(f"tier {tier}:")
    pol = {
        "pure LK": [d[idx][1] - d[0] for d in data],
        "pure NN": [d[idx][0] - d[0] for d in data],
        "mean": [0.5*(d[idx][0] + d[idx][1]) - d[0] for d in data],
        "classifier": [(d[idx][0] if d[4] else d[idx][1]) - d[0]
                       for d in data],
    }
    for lbl, e in pol.items():
        m, md, p90 = stats(e)
        print(f"  {lbl:12s} MAD {m:.4f}  median {md:.4f}  p90 {p90:.4f}")
    esub_c = [(d[idx][0] if d[4] else d[idx][1]) - d[0] for d in data if d[4]]
    esub_l = [d[idx][1] - d[0] for d in data if d[4]]
    if esub_c:
        print(f"  routed subset (n={len(esub_c)}): classifier MAD "
              f"{sum(abs(x) for x in esub_c)/len(esub_c):.4f} vs "
              f"LK-there {sum(abs(x) for x in esub_l)/len(esub_l):.4f}")
    print()
