"""1) Acentric factor accuracy in three information tiers.
2) Psat: Nannoolal slope model vs Ambrose-Walton CSP at matching tiers.

References: Perry (2-8 correlation + 2-106 criticals incl. omega) and
chemicals Wagner banks (omega derived definitionally from the fit at 0.7 Tc).
Common-subset evaluation within each source so tiers are comparable.
"""
from pathlib import Path
import json
import math
import sys
import warnings as pywarn
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import nannoolal_method as nm
from chemicals.identifiers import search_chemical
from chemicals.vapor_pressure import (
    Ambrose_Walton, Psat_data_WagnerMcGarry, Psat_data_WagnerPoling,
    Wagner, Wagner_original)

pywarn.filterwarnings("ignore")
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
def smiles_for(key):
    if key not in _sc:
        try:
            _sc[key] = search_chemical(key).smiles
        except Exception:
            _sc[key] = None
    return _sc[key]

# ---------------- build compound records --------------------------------
# each: dict(name, smiles, pref(T)->kPa, tlo, thi, tc, pc_kPa, omega_ref)
def perry_records():
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
        smi = smiles_for(cas)
        if not smi:
            continue
        def pref(T, c=c):
            return math.exp(c[0] + c[1]/T + c[2]*math.log(T)
                            + c[3]*T**c[4]) / 1000.0
        yield dict(name=row.get("name") or cas, smiles=smi, pref=pref,
                   tlo=tlo, thi=thi, tc=cc["Tc_K"], pc=cc["Pc_MPa"] * 1e3,
                   omega=cc["omega"])

def wagner_records():
    for cas, r in Psat_data_WagnerMcGarry.iterrows():
        smi = smiles_for(cas)
        if not smi:
            continue
        def pref(T, r=r):
            return Wagner_original(T, r["Tc"], r["Pc"], r["A"], r["B"],
                                   r["C"], r["D"]) / 1000.0
        yield dict(name=r["Name"], smiles=smi, pref=pref, tlo=r["Tmin"],
                   thi=r["Tc"] - 0.1, tc=r["Tc"], pc=r["Pc"] / 1000.0,
                   omega=None)
    for cas, r in Psat_data_WagnerPoling.iterrows():
        if cas in Psat_data_WagnerMcGarry.index:
            continue
        smi = smiles_for(cas)
        if not smi:
            continue
        def pref(T, r=r):
            return Wagner(T, r["Tc"], r["Pc"], r["A"], r["B"],
                          r["C"], r["D"]) / 1000.0
        yield dict(name=r["Name"], smiles=smi, pref=pref, tlo=r["Tmin"],
                   thi=min(r["Tmax"], r["Tc"] - 0.1), tc=r["Tc"],
                   pc=r["Pc"] / 1000.0, omega=None)

# ---------------- part 1: omega tiers ------------------------------------
def omega_bench(tag, records):
    rows = []
    for rec in records:
        tb = bisect_T(rec["pref"], P_ATM, rec["tlo"], rec["thi"])
        if tb is None:
            continue
        w_ref = rec["omega"]
        if w_ref is None:  # derive from the Wagner fit itself
            t07 = 0.7 * rec["tc"]
            if not (rec["tlo"] <= t07 <= rec["thi"]):
                continue
            w_ref = -math.log10(rec["pref"](t07) / rec["pc"]) - 1.0
        try:
            wA = nm.acentric_factor(rec["smiles"], tb=tb, tc=rec["tc"],
                                    pc_kPa=rec["pc"])
            wB = nm.acentric_factor(rec["smiles"], tb=tb)
            wC = nm.acentric_factor(rec["smiles"])
        except Exception:
            continue
        if None in (wA, wB, wC):
            continue
        rows.append((w_ref, wA, wB, wC))
    out = [f"omega vs {tag}: n = {len(rows)} (common subset)"]
    out.append(f"{'tier':34s} {'MAD':>7s} {'median':>7s} {'bias':>7s}")
    for label, idx in [("A: real Tb, Tc, Pc (slope only)", 1),
                       ("B: real Tb only", 2),
                       ("C: fully estimated", 3)]:
        e = [r[idx] - r[0] for r in rows]
        ae = sorted(abs(x) for x in e)
        out.append(f"{label:34s} {sum(ae)/len(ae):7.4f} "
                   f"{ae[len(ae)//2]:7.4f} {sum(e)/len(e):+7.4f}")
    return "\n".join(out)

# ---------------- part 2: Psat NN vs Ambrose-Walton -----------------------
TARGETS = [0.001, 0.005, 0.05, 0.5, 5.0, 50.0, 300.0, 1500.0]
def bin_of(p):
    if p < 0.01: return "ELP"
    if p < 10.0: return "LP"
    if p < 500.0: return "MP"
    return "HP"

def aw_kpa(T, tc, pc_kpa, w):
    try:
        return Ambrose_Walton(T, tc, pc_kpa * 1000.0, w) / 1000.0
    except Exception:
        return None

def psat_bench(tag, records):
    METH = ["NN (real Tb)", "AW (real Tc,Pc,w)", "AW (real Tc,Pc; NN w)",
            "AW (NN Tc,Pc,w from real Tb)"]
    bins = {m: {} for m in METH}
    n_comp = 0
    for rec in records:
        tb = bisect_T(rec["pref"], P_ATM, rec["tlo"], rec["thi"])
        if tb is None:
            continue
        w_ref = rec["omega"]
        if w_ref is None:
            t07 = 0.7 * rec["tc"]
            if not (rec["tlo"] <= t07 <= rec["thi"]):
                continue
            w_ref = -math.log10(rec["pref"](t07) / rec["pc"]) - 1.0
        try:
            r_nn = nm.estimate_psat(rec["smiles"], tb=tb)
            w_nn_exp = nm.acentric_factor(rec["smiles"], tb=tb,
                                          tc=rec["tc"], pc_kPa=rec["pc"])
            est = nm.estimate(rec["smiles"], tb=tb)
            w_nn_est = nm.acentric_factor(rec["smiles"], tb=tb)
        except Exception:
            continue
        if (r_nn is None or r_nn.db is None or w_nn_exp is None
                or w_nn_est is None or est.tc_K is None or est.pc_kPa is None):
            continue
        got = False
        for pt in TARGETS:
            t = bisect_T(rec["pref"], pt, rec["tlo"], rec["thi"])
            if t is None or abs(t - tb) < 0.5 or t > 0.85 * rec["tc"]:
                continue
            vals = {
                METH[0]: r_nn.psat_kPa(t),
                METH[1]: aw_kpa(t, rec["tc"], rec["pc"], w_ref),
                METH[2]: aw_kpa(t, rec["tc"], rec["pc"], w_nn_exp),
                METH[3]: aw_kpa(t, est.tc_K, est.pc_kPa, w_nn_est),
            }
            if any(v is None or v <= 0 for v in vals.values()):
                continue
            b = bin_of(pt)
            for m, v in vals.items():
                bins[m].setdefault(b, []).append(100.0 * abs(v / pt - 1.0))
            got = True
        n_comp += got
    out = [f"Psat vs {tag}: {n_comp} compounds, common evaluation points"]
    out.append(f"{'method':30s} {'ELP':>7s} {'LP':>7s} {'MP':>7s} "
               f"{'HP':>7s} {'ALL':>7s}")
    for m in METH:
        cells, allv = [], []
        for b in ("ELP", "LP", "MP", "HP"):
            v = bins[m].get(b, [])
            cells.append(f"{sum(v)/len(v):7.2f}" if v else f"{'-':>7s}")
            allv += v
        cells.append(f"{sum(allv)/len(allv):7.2f}")
        out.append(f"{m:30s} " + " ".join(cells))
    return "\n".join(out)

parts = []
parts.append(omega_bench("Perry 2-106 omega", perry_records()))
parts.append(omega_bench("Wagner-derived omega", wagner_records()))
parts.append(psat_bench("Perry 2-8", perry_records()))
parts.append(psat_bench("Wagner fits", wagner_records()))
report = "\n\n".join(parts)
print(report)
with open(str(REPO_ROOT / "outputs/nannoolal_omega_aw_benchmark.txt"), "w") as f:
    f.write("Acentric factor tiers + Nannoolal-vs-Ambrose-Walton Psat "
            "benchmark.\nMARD% bins: ELP<0.01 kPa, LP 0.01-10, MP 10-500, "
            "HP>500; Tr <= 0.85; common subsets.\n\n" + report + "\n")
