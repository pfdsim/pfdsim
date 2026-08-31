"""Benchmark the Nannoolal Psat layer against four reference sources.

Tier 1 (most reliable):
  A. Perry 2-8 DIPPR-101 correlations (perry_properties.json)
  B. chemicals: Wagner fits (McGarry + Poling banks)
Tier 2 (wider):
  C. Perry 2-10 point tables
  D. data/antoine.txt (classic Antoine, mmHg/degC)

Protocol: anchor the model at the reference's own 1-atm point (pure slope
test, like the paper); evaluate P%-errors at pressure targets, binned per
the paper: ELP < 0.01 kPa, LP 0.01-10, MP 10-500, HP > 500.  Cap Tr <= 0.85
where Tc is known.  Mono-functional alcohols (no eq. 9) reported separately.
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
    Psat_data_WagnerMcGarry, Psat_data_WagnerPoling, Wagner, Wagner_original)

pywarn.filterwarnings("ignore")
P_ATM = 101.325  # kPa

def bisect_T(f, target, lo, hi):
    """Solve f(T) = target (f monotone increasing) on [lo, hi]; None if unbracketed."""
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

_smiles_cache = {}
def smiles_for(key):
    if key not in _smiles_cache:
        try:
            _smiles_cache[key] = search_chemical(key).smiles
        except Exception:
            _smiles_cache[key] = None
    return _smiles_cache[key]

def model_for(smiles, tb):
    try:
        r = nm.estimate_psat(smiles, tb=tb)
    except Exception:
        return None
    return r if r.db is not None and r.tb_K is not None else None

TARGETS = [0.001, 0.005, 0.05, 0.5, 5.0, 50.0, 300.0, 1500.0]  # kPa

def bin_of(p):
    if p < 0.01: return "ELP"
    if p < 10.0: return "LP"
    if p < 500.0: return "MP"
    return "HP"

def run_source(tag, compounds):
    """compounds: iterable of (name, smiles, pref_fn(T)->kPa, Tlo, Thi, Tc_or_None)."""
    stats = {}          # bin -> list of |%err|
    t_errs = {}         # bin -> list of |dT|
    alco = {}           # same, mono-alcohols only
    n_comp = n_alco = 0
    for name, smi, pref, tlo, thi, tc in compounds:
        tb = bisect_T(pref, P_ATM, tlo, thi)
        if tb is None:
            continue
        r = model_for(smi, tb)
        if r is None:
            continue
        is_alc = any("eq. (9)" in w for w in r.warnings)
        got = False
        for pt in TARGETS:
            t = bisect_T(pref, pt, tlo, thi)
            if t is None or abs(t - tb) < 0.5:
                continue
            if tc is not None and t > 0.85 * tc:
                continue
            pc_ = r.psat_kPa(t)
            if pc_ is None or pc_ <= 0:
                continue
            e = 100.0 * abs(pc_ / pt - 1.0)
            b = bin_of(pt)
            stats.setdefault(b, []).append(e)
            tcalc = r.temperature_K(pt)
            if tcalc:
                t_errs.setdefault(b, []).append(abs(tcalc - t))
            if is_alc:
                alco.setdefault(b, []).append(e)
            got = True
        if got:
            n_comp += 1
            n_alco += is_alc
    out = [f"== {tag}: {n_comp} compounds ({n_alco} mono-alcohols) =="]
    out.append(f"{'bin':>4s} {'n':>6s} {'MARD%':>8s} {'median%':>8s} {'|dT| K':>7s}")
    allv = []
    for b in ("ELP", "LP", "MP", "HP"):
        v = stats.get(b, [])
        if not v:
            continue
        allv += v
        v_s = sorted(v)
        te = t_errs.get(b, [])
        out.append(f"{b:>4s} {len(v):6d} {sum(v)/len(v):8.2f} "
                   f"{v_s[len(v)//2]:8.2f} {sum(te)/len(te):7.2f}")
    if allv:
        a_s = sorted(allv)
        out.append(f"{'ALL':>4s} {len(allv):6d} {sum(allv)/len(allv):8.2f} "
                   f"{a_s[len(allv)//2]:8.2f}")
    for b in ("ELP", "LP", "MP"):
        v = alco.get(b, [])
        if v:
            out.append(f"  mono-alcohols {b}: n={len(v)} MARD {sum(v)/len(v):.1f} %")
    return "\n".join(out)

reports = []

# ---------- A: Perry 2-8 ----------------------------------------------------
perry = json.load(open(str(REPO_ROOT / "data/perry_properties.json")))["chemicals"]
def perry_compounds():
    for cas, row in perry.items():
        if cas == "302-17-0":
            continue
        vps = row.get("vapor_pressure") or []
        if not vps:
            continue
        rec = vps[0]
        c = (list(rec["coefficients"]) + [0.0, 0.0, 0.0, 0.0, 1.0])[:5]
        if c[4] == 0:
            c[4] = 1.0   # avoid 0**0 when C4 is absent
        tlo, thi = rec["T_min_K"], rec["T_max_K"]
        if not tlo or not thi or thi <= tlo:
            continue
        def pref(T, c=c):
            return math.exp(c[0] + c[1]/T + c[2]*math.log(T)
                            + c[3]*T**c[4]) / 1000.0
        smi = smiles_for(cas)
        if not smi:
            continue
        tc = (row.get("critical_constants") or {}).get("Tc_K")
        yield row.get("name") or cas, smi, pref, tlo, thi, tc
reports.append(run_source("A: Perry 2-8 correlations (tier 1)", perry_compounds()))

# ---------- B: chemicals Wagner banks ---------------------------------------
def wagner_compounds():
    for cas, r in Psat_data_WagnerMcGarry.iterrows():
        smi = smiles_for(cas)
        if not smi:
            continue
        def pref(T, r=r):
            return Wagner_original(T, r["Tc"], r["Pc"], r["A"], r["B"],
                                   r["C"], r["D"]) / 1000.0
        yield r["Name"], smi, pref, r["Tmin"], r["Tc"] - 0.1, r["Tc"]
    for cas, r in Psat_data_WagnerPoling.iterrows():
        if cas in Psat_data_WagnerMcGarry.index:
            continue
        smi = smiles_for(cas)
        if not smi:
            continue
        def pref(T, r=r):
            return Wagner(T, r["Tc"], r["Pc"], r["A"], r["B"],
                          r["C"], r["D"]) / 1000.0
        thi = min(r["Tmax"], r["Tc"] - 0.1)
        yield r["Name"], smi, pref, r["Tmin"], thi, r["Tc"]
reports.append(run_source("B: chemicals Wagner fits (tier 1)", wagner_compounds()))

# ---------- C: Perry 2-10 point tables --------------------------------------
t210 = json.load(open(
    str(REPO_ROOT / "data/perry_table_2_10_vapor_pressure.json")))["chemicals"]
def run_table_210():
    stats, alco = {}, {}
    n_comp = n_alco = 0
    for cas, row in t210.items():
        tb, tm = row.get("Tb_K"), row.get("Tm_K")
        pts = row.get("vapor_pressure") or []
        if not tb or not pts:
            continue
        smi = smiles_for(cas)
        if not smi:
            continue
        r = model_for(smi, tb)
        if r is None:
            continue
        is_alc = any("eq. (9)" in w for w in r.warnings)
        got = False
        for pt in pts:
            t, p_ref = pt.get("T_K"), (pt.get("P_bar") or 0) * 100.0
            if not t or p_ref <= 0 or abs(p_ref - P_ATM) < 2.0:
                continue
            if tm and t < tm + 2:          # solid-phase rows
                continue
            pc_ = r.psat_kPa(t)
            if pc_ is None or pc_ <= 0:
                continue
            e = 100.0 * abs(pc_ / p_ref - 1.0)
            b = bin_of(p_ref)
            stats.setdefault(b, []).append(e)
            if is_alc:
                alco.setdefault(b, []).append(e)
            got = True
        if got:
            n_comp += 1
            n_alco += is_alc
    out = [f"== C: Perry 2-10 point tables (tier 2): {n_comp} compounds "
           f"({n_alco} mono-alcohols) =="]
    out.append(f"{'bin':>4s} {'n':>6s} {'MARD%':>8s} {'median%':>8s}")
    allv = []
    for b in ("ELP", "LP", "MP", "HP"):
        v = stats.get(b, [])
        if not v:
            continue
        allv += v
        out.append(f"{b:>4s} {len(v):6d} {sum(v)/len(v):8.2f} "
                   f"{sorted(v)[len(v)//2]:8.2f}")
    if allv:
        out.append(f"{'ALL':>4s} {len(allv):6d} {sum(allv)/len(allv):8.2f} "
                   f"{sorted(allv)[len(allv)//2]:8.2f}")
    for b in ("ELP", "LP", "MP"):
        v = alco.get(b, [])
        if v:
            out.append(f"  mono-alcohols {b}: n={len(v)} MARD {sum(v)/len(v):.1f} %")
    return "\n".join(out)
reports.append(run_table_210())

# ---------- D: antoine.txt ---------------------------------------------------
def antoine_compounds():
    for ln in open(str(REPO_ROOT / "data/antoine.txt")):
        parts = ln.rstrip("\n").split("\t")
        if len(parts) < 8 or parts[0] == "ID":
            continue
        try:
            a, b_, c = float(parts[3]), float(parts[4]), float(parts[5])
            tlo, thi = float(parts[6]) + 273.15, float(parts[7]) + 273.15
        except ValueError:
            continue
        if thi - tlo < 20:
            continue
        name = parts[2]
        smi = smiles_for(name) or smiles_for(name.replace("-", " "))
        if not smi:
            continue
        def pref(T, a=a, b_=b_, c=c):   # mmHg, degC -> kPa
            return 10.0 ** (a - b_ / (c + T - 273.15)) * 0.13332237
        yield name, smi, pref, tlo, thi, None
reports.append(run_source("D: antoine.txt (tier 2)", antoine_compounds()))

report = "\n\n".join(reports)
print(report)
with open(str(REPO_ROOT / "outputs/nannoolal_psat_benchmark.txt"), "w") as f:
    f.write("Nannoolal Part-3 Psat benchmark -- anchored at each source's own "
            "1-atm point (2-10: its Tb column).\nP-error bins per the paper: "
            "ELP<0.01 kPa, LP 0.01-10, MP 10-500, HP>500. Tr capped at 0.85 "
            "where Tc known.\nPaper's own overall claim: 6.2 %kPa "
            "(vs 4.5 % for per-compound fitted Antoine).\n\n" + report + "\n")
