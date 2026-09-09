"""Full NN x Ambrose-Walton Psat matrix across information tiers.

Arms:
 1a NN  | est Tb anchor                     (nothing real)
 1b AW  | est Tc,Pc + LK(est) omega         (nothing real)
 2a NN  | real Tb anchor
 2b AW  | est Tc,Pc(real Tb) + LK omega     (real Tb only)
 3a AW  | real Tc,Pc + NN omega (tier A)
 3b AW  | real Tc,Pc + LK omega (real Tb)
 4  AW  | real Tc,Pc + experimental omega
 5  NN  | anchored at omega point (0.7Tc, Pc 10^-(1+w)); real Tc,Pc,w, no Tb
 6a NN  | anchored at self-consistent Tb from AW<->LK; real Tc,Pc, no Tb/w
 6b AW  | real Tc,Pc + iterated LK omega    (same loop as 6a)
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
from chemicals.vapor_pressure import (Ambrose_Walton, Psat_data_WagnerMcGarry,
    Psat_data_WagnerPoling, Wagner, Wagner_original)

P_ATM = 101.325
TARGETS = [0.001, 0.005, 0.05, 0.5, 5.0, 50.0, 300.0, 1500.0]

def bisect_T(f, target, lo, hi):
    flo, fhi = f(lo), f(hi)
    if not (flo <= target <= fhi): return None
    for _ in range(80):
        mid = 0.5*(lo+hi)
        if f(mid) < target: lo = mid
        else: hi = mid
    return 0.5*(lo+hi)

_sc = {}
def smi(k):
    if k not in _sc:
        try: _sc[k] = search_chemical(k).smiles
        except Exception: _sc[k] = None
    return _sc[k]

def aw(T, tc, pc_kpa, w):
    try:
        v = Ambrose_Walton(T, tc, pc_kpa*1000.0, w) / 1000.0
        return v if v > 0 else None
    except Exception:
        return None

def aw_tsat(p_kpa, tc, pc_kpa, w):
    return bisect_T(lambda T: aw(T, tc, pc_kpa, w) or 0.0, p_kpa,
                    0.30*tc, 0.999*tc)

# NOTE: an earlier design iterated Tb <-> LK omega <-> AW Tsat(1 atm) from
# real (Tc, Pc) alone.  That system is SINGULAR: LK and AW are the same
# Pitzer expansion solved in opposite directions, so their composition is
# ~the identity map and (Tc, Pc) cannot determine (Tb, omega).  Case 6 now
# closes the system with structure instead: NN supplies omega (estimated-Tb
# slope evaluated against the real criticals), AW carries the curve.

ARMS = ["1a NN  est Tb", "1b AW  est all + LK w",
        "2a NN  real Tb", "2b AW  real Tb, est crit + LK w",
        "3a AW  real crit + NN w", "3b AW  real crit + LK w",
        "4  AW  real crit + real w",
        "5  NN  anchor from real crit+w (no Tb)",
        "6a AW  real crit + NN w (est Tb; no real Tb/w)",
        "6b NN  re-anchored at 6a curve's 1 atm point",
        "6c NN  Tb back-calc from real Tc via Tc/Tb ratio",
        "6d AW  same ratio-Tb -> LK w -> AW (real crit)"]

def records_perry():
    perry = json.load(open(str(REPO_ROOT / "data/perry_properties.json")))["chemicals"]
    for cas, row in perry.items():
        if cas == "302-17-0": continue
        cc = row.get("critical_constants") or {}
        vps = row.get("vapor_pressure") or []
        if not vps or not cc.get("Tc_K") or not cc.get("Pc_MPa") or cc.get("omega") is None:
            continue
        rec = vps[0]
        c = (list(rec["coefficients"]) + [0.0]*5)[:5]
        if c[4] == 0: c[4] = 1.0
        tlo, thi = rec["T_min_K"], rec["T_max_K"]
        if not tlo or not thi or thi <= tlo: continue
        s = smi(cas)
        if not s: continue
        def pref(T, c=c):
            return math.exp(c[0]+c[1]/T+c[2]*math.log(T)+c[3]*T**c[4])/1000.0
        yield s, pref, tlo, thi, cc["Tc_K"], cc["Pc_MPa"]*1e3, cc["omega"]

def records_wagner():
    for cas, r in Psat_data_WagnerMcGarry.iterrows():
        s = smi(cas)
        if s:
            def pref(T, r=r):
                return Wagner_original(T, r["Tc"], r["Pc"], r["A"], r["B"], r["C"], r["D"])/1000.0
            yield s, pref, r["Tmin"], r["Tc"]-0.1, r["Tc"], r["Pc"]/1000.0, None
    for cas, r in Psat_data_WagnerPoling.iterrows():
        if cas in Psat_data_WagnerMcGarry.index: continue
        s = smi(cas)
        if s:
            def pref(T, r=r):
                return Wagner(T, r["Tc"], r["Pc"], r["A"], r["B"], r["C"], r["D"])/1000.0
            yield s, pref, r["Tmin"], min(r["Tmax"], r["Tc"]-0.1), r["Tc"], r["Pc"]/1000.0, None

def run(tag, records):
    bins = {a: {} for a in ARMS}
    n_comp = 0
    for s, pref, tlo, thi, tc, pc, w_ref in records:
        tb = bisect_T(pref, P_ATM, tlo, thi)
        if tb is None: continue
        if w_ref is None:
            t07 = 0.7*tc
            if not (tlo <= t07 <= thi): continue
            w_ref = -math.log10(pref(t07)/pc) - 1.0
        try:
            eC = nm.estimate(s)                       # nothing real
            eB = nm.estimate(s, tb=tb)                # real Tb
            nn1 = nm.estimate_psat(s)                 # est Tb anchor
            nn2 = nm.estimate_psat(s, tb=tb)
            w_nnA = nm.acentric_factor(s, tb=tb, tc=tc, pc_kPa=pc)
            nn5 = nm.estimate_psat(
                s, psat_point=(0.7*tc, pc*10.0**(-(1.0+w_ref))))
            w6 = nm.acentric_factor(s, tc=tc, pc_kPa=pc)   # est-Tb slope
            tb6 = aw_tsat(P_ATM, tc, pc, w6) if w6 is not None else None
            nn6 = nm.estimate_psat(s, tb=tb6) if tb6 else None
            # 6c: Tc = Tb * g(structure) -> Tb = Tc_real / g
            g = nm.estimate(s, tb=100.0).tc_K
            tb_ratio = tc / (g / 100.0) if g else None
            nn6c = nm.estimate_psat(s, tb=tb_ratio) if tb_ratio else None
            # 6d: same anchor carried by AW via self-anchoring LK omega
            w6d = (LK_omega(tb_ratio, tc, pc*1000.0) if tb_ratio else None)
            w_lk1 = LK_omega(eC.tb_K, eC.tc_K, eC.pc_kPa*1000.0)
            w_lk2 = LK_omega(tb, eB.tc_K, eB.pc_kPa*1000.0)
            w_lk3 = LK_omega(tb, tc, pc*1000.0)
        except Exception:
            continue
        if (nn1.db is None or nn1.tb_K is None or nn2.db is None
                or nn5.tb_K is None or nn6 is None or nn6.tb_K is None
                or w_nnA is None or w6 is None or w6d is None
                or nn6c is None or nn6c.tb_K is None
                or eC.tc_K is None or eC.pc_kPa is None
                or eB.tc_K is None or eB.pc_kPa is None):
            continue
        got = False
        for pt in TARGETS:
            t = bisect_T(pref, pt, tlo, thi)
            if t is None or abs(t - tb) < 0.5 or t > 0.85*tc:
                continue
            vals = {
                ARMS[0]: nn1.psat_kPa(t),
                ARMS[1]: aw(t, eC.tc_K, eC.pc_kPa, w_lk1),
                ARMS[2]: nn2.psat_kPa(t),
                ARMS[3]: aw(t, eB.tc_K, eB.pc_kPa, w_lk2),
                ARMS[4]: aw(t, tc, pc, w_nnA),
                ARMS[5]: aw(t, tc, pc, w_lk3),
                ARMS[6]: aw(t, tc, pc, w_ref),
                ARMS[7]: nn5.psat_kPa(t),
                ARMS[8]: aw(t, tc, pc, w6),
                ARMS[9]: nn6.psat_kPa(t),
                ARMS[10]: nn6c.psat_kPa(t),
                ARMS[11]: aw(t, tc, pc, w6d),
            }
            if any(v is None or v <= 0 for v in vals.values()):
                continue
            b = ("ELP" if pt < 0.01 else "LP" if pt < 10 else
                 "MP" if pt < 500 else "HP")
            for a, v in vals.items():
                bins[a].setdefault(b, []).append(100.0*abs(v/pt - 1.0))
            got = True
        n_comp += got
    out = [f"== {tag}: {n_comp} compounds, common points ==",
           f"{'arm':44s} {'ELP':>7s} {'LP':>7s} {'MP':>7s} {'HP':>7s} {'ALL':>7s}"]
    for a in ARMS:
        cells, allv = [], []
        for b in ("ELP", "LP", "MP", "HP"):
            v = bins[a].get(b, [])
            cells.append(f"{sum(v)/len(v):7.2f}" if v else f"{'-':>7s}")
            allv += v
        cells.append(f"{sum(allv)/len(allv):7.2f}")
        out.append(f"{a:44s} " + " ".join(cells))
    return "\n".join(out)

parts = [run("Perry 2-8", records_perry()),
         run("Wagner fits", records_wagner())]
report = "\n\n".join(parts)
print(report)
with open(str(REPO_ROOT / "outputs/nannoolal_aw_matrix.txt"), "w") as f:
    f.write("NN x Ambrose-Walton Psat matrix (MARD %). Bins: ELP<0.01 kPa, "
            "LP 0.01-10, MP 10-500, HP>500; Tr<=0.85; common subset.\n"
            "Case 5 anchors NN at (0.7Tc, Pc*10^-(1+w)); case 6 iterates "
            "Tb<->LK omega<->AW Tsat(1 atm) from real Tc,Pc only.\n\n"
            + report + "\n")
