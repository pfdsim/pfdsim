"""Final combined benchmark: Perry 2-106 + ATN Part-12 (corrected) criticals.
Tc scored only where a high-quality experimental Tb exists (validated Tsat
solve, Perry 2-10, or PFDSim-validated values). Local refits active (default).
"""
from pathlib import Path
import json
import math
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import chemicals as chemlib
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
from chemical_properties import ChemicalDatabase
from nannoolal_method import estimate, NannoolalError

P_ATM = 101325.0
OUT = str(REPO_ROOT / "outputs/nannoolal_benchmark_final.txt")

perry = json.load(open(str(REPO_ROOT / "data/perry_properties.json")))["chemicals"]
t210 = json.load(open(str(REPO_ROOT / "data/perry_table_2_10_vapor_pressure.json")))["chemicals"]
atn = json.load(open(str(REPO_ROOT / "data/acs_jced_5b00571_table1.json")))["chemicals"]
smiles_db = ChemicalDatabase(enable_online=False)

EXCLUDE_CAS = {"302-17-0"}                 # chloral hydrate (decomposer)
PFDSim_TB = {                              # validated by PFDSim, 2026-07-09
    "1,4-butanediol": 503.2, "1,5-pentanediol": 515.2, "1,6-hexanediol": 523.2,
    "1,8-octanediol": 552.0, "1,9-nonanediol": 561.2, "1,10-decanediol": 570.2,
    "2-methoxyethanol": 397.5, "1-methoxy-2-propanol": 393.2,
    "2-(2-ethoxyethoxy)ethanol": 475.2, "triethylene glycol": 559.0,
    "tetraethylene glycol": 600.6, "adipic acid": 610.6,
    "1,2-butanediol": 469.2, "diglycolic anhydride": 513.6,
}
PFDSim_TB_CAS = {}
for name, tb in PFDSim_TB.items():
    try:
        PFDSim_TB_CAS[chemlib.search_chemical(name).CASs] = tb
    except Exception:
        pass


def solve_tsat(vp_records):
    for rec in vp_records or []:
        pmin, pmax = rec.get("P_at_T_min_Pa"), rec.get("P_at_T_max_Pa")
        if pmin is None or pmax is None or not (pmin <= P_ATM <= pmax):
            continue
        c1, c2, c3, c4, c5 = rec["coefficients"][:5]
        lo, hi = rec["T_min_K"], rec["T_max_K"]
        target = math.log(P_ATM)

        def f(t):
            return c1 + c2 / t + c3 * math.log(t) + c4 * t ** c5 - target

        try:
            if abs(f(lo) + target - math.log(pmin)) > 0.15 or \
               abs(f(hi) + target - math.log(pmax)) > 0.15:
                continue
        except (ValueError, OverflowError):
            continue
        flo = f(lo)
        if flo * f(hi) > 0:
            continue
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if f(mid) * flo <= 0:
                hi = mid
            else:
                lo, flo = mid, f(mid)
        return 0.5 * (lo + hi)
    return None


# ---- assemble reference set: ATN preferred over Perry -----------------------
compounds = {}
for cas, rec in perry.items():
    cc = rec.get("critical_constants") or {}
    if cc:
        compounds[cas] = {
            "name": rec.get("name", cas), "src": "Perry",
            "tc": cc.get("Tc_K"),
            "pc": cc["Pc_MPa"] * 1000 if cc.get("Pc_MPa") else None,
            "vc": cc["Vc_m3_per_kmol"] * 1000 if cc.get("Vc_m3_per_kmol") else None,
        }
for cas, rec in atn.items():
    cp = rec.get("critical_properties") or {}
    entry = compounds.setdefault(cas, {"name": rec.get("name", cas), "src": "ATN"})
    entry["src"] = "ATN" if cas not in perry else "ATN>Perry"
    if cp.get("Tc_K"):
        entry["tc"] = cp["Tc_K"]
    if cp.get("Pc_MPa"):
        entry["pc"] = cp["Pc_MPa"] * 1000
    if cp.get("Vc_cm3_mol"):
        entry["vc"] = cp["Vc_cm3_mol"]

# ---- high-quality experimental Tb -------------------------------------------
def hq_tb(cas):
    if cas in PFDSim_TB_CAS:
        return PFDSim_TB_CAS[cas], "PFDSim"
    prec = perry.get(cas)
    if prec:
        tsat = solve_tsat(prec.get("vapor_pressure"))
        if tsat:
            return tsat, "Tsat"
    trec = t210.get(cas)
    if trec and trec.get("Tb_K"):
        tm = trec.get("Tm_K")
        if tm and tm >= trec["Tb_K"] - 2:
            return None, None                 # sublimer/decomposer
        return trec["Tb_K"], "2-10"
    return None, None


# ---- run ---------------------------------------------------------------------
devs = {"tb": [], "tc": [], "pc": [], "vc": []}
rows_out = []
n_frag = 0
for cas, c in compounds.items():
    if cas in EXCLUDE_CAS:
        continue
    try:
        result = smiles_db.resolve_smiles_info(cas, fetch_online=False)
        smi = result.smiles if result else None
        if not smi:
            continue
    except Exception:
        continue
    tb_exp, tb_src = hq_tb(cas)
    try:
        r = estimate(smi, tb=tb_exp)
    except NannoolalError:
        n_frag += 1
        continue
    line = [c["name"], c["src"], tb_src or "-"]
    if tb_exp and r.tb_K:
        devs["tb"].append(r.tb_K - tb_exp)
    if c.get("tc") and r.tc_K and tb_exp:      # Tc gated on high-quality Tb
        devs["tc"].append(r.tc_K - c["tc"])
    if c.get("pc") and r.pc_kPa:
        devs["pc"].append(100 * (r.pc_kPa - c["pc"]) / c["pc"])
    if c.get("vc") and r.vc_cm3_mol:
        devs["vc"].append(100 * (r.vc_cm3_mol - c["vc"]) / c["vc"])

# separate Tb-only pass over full 2-10 (as before) for the Tb statistic
tb_devs = []
for cas, rec in t210.items():
    if cas in EXCLUDE_CAS or not rec.get("Tb_K"):
        continue
    tm = rec.get("Tm_K")
    tb_ref = PFDSim_TB_CAS.get(cas) or rec["Tb_K"]
    if tm and tm >= tb_ref - 2:
        continue
    try:
        result = smiles_db.resolve_smiles_info(cas, fetch_online=False)
        smi = result.smiles if result else None
        r = estimate(smi)
    except Exception:
        continue
    if r.tb_K:
        tb_devs.append(r.tb_K - tb_ref)

def stats(arr):
    a = sorted(abs(x) for x in arr)
    n = len(a)
    bias = sum(arr) / n
    return n, sum(a) / n, a[n // 2], a[min(n - 1, int(round(0.9 * (n - 1))))], bias

out = ["FINAL Nannoolal benchmark - Perry 2-106 + ATN Part 12 (corrected), "
       "local refits/extensions active", ""]
print(f"{'property':28s} {'n':>4s} {'AAD':>7s} {'median':>7s} {'p90':>7s} {'bias':>7s}")
for label, arr, unit in [
        ("Tb vs Perry 2-10 [K]", tb_devs, "K"),
        ("Tc (high-quality Tb) [K]", devs["tc"], "K"),
        ("Pc [%]", devs["pc"], "%"),
        ("Vc [%]", devs["vc"], "%")]:
    n, aad, med, p90, bias = stats(arr)
    line = f"{label:28s} {n:4d} {aad:7.2f} {med:7.2f} {p90:7.2f} {bias:+7.2f}"
    print(line)
    out.append(line)
out.append(f"\nnot fragmentable (out of scope): {n_frag}")
with open(OUT, "w") as fh:
    fh.write("\n".join(out) + "\n")
print("\nwrote", OUT)
