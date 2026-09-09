"""Hvap benchmark: slope source x dZ information tier.

Reference: Perry 2-150 Watson-type Hvap correlations (+ CRC HvapTb/Hvap298
experimental points from `chemicals`).  Hvap = dZ * R * T^2 * dlnPs/dT.

Slope arms:
  R1/R2  reference Psat slope (Perry 2-8), dZ = 1 / PTV(all real)  [ceiling]
  N1     NN slope (real Tb anchor), dZ = 1
Then the NN slope with a dZ information-tier ladder:
  Z1  no real criticals: est Tc/Pc (real-Tb chain) + LK omega  -> virial+Rackett
  Z2  real Tc,Pc; omega = LK(real Tb,Tc,Pc); no Vc             -> virial+Rackett
  Z3  real Tc,Pc,omega; no Vc                                  -> virial+Rackett
  Z4  real Tc,Pc,Vc; omega = LK(real Tb,Tc,Pc)                 -> PTV
  Z5  real Tc,Pc,Vc,omega                                      -> PTV

PTV = Patel-Teja-Valderrama (Zc-tuned cubic; needs Vc).
Carboxylic acids excluded (vapor dimerization: slope methods measure the
apparent Hvap, calorimetric correlations the real one).
"""
from pathlib import Path
import json
import math
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import nannoolal_method as nm
from chemicals.identifiers import search_chemical
from chemicals.acentric import LK_omega
from chemicals.phase_change import Hvap_data_CRC
from rdkit import Chem

from physical_constants import R_J_MOL_K

R = R_J_MOL_K
TR_TARGETS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
_COOH = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")

_sc = {}
def smiles_for(cas):
    if cas not in _sc:
        try:
            _sc[cas] = search_chemical(cas).smiles
        except Exception:
            _sc[cas] = None
    return _sc[cas]


def ptv_dz(T, p_kpa, tc, pc_kpa, vc_m3kmol, omega):
    """dZ = Zv - Zl from the Patel-Teja-Valderrama cubic; None if degenerate."""
    P, Pc = p_kpa * 1e3, pc_kpa * 1e3
    zc = Pc * (vc_m3kmol / 1000.0) / (R * tc)
    om_a = 0.66121 - 0.76105 * zc
    om_b = 0.02207 + 0.20868 * zc
    om_c = 0.57765 - 1.87080 * zc
    f = 0.46283 + 3.58230 * (omega * zc) + 8.19417 * (omega * zc) ** 2
    alpha = (1.0 + f * (1.0 - math.sqrt(T / tc))) ** 2
    a = om_a * R * R * tc * tc / Pc * alpha
    b = om_b * R * tc / Pc
    c = om_c * R * tc / Pc
    roots = np.roots([P,
                      P * c - R * T,
                      a - P * (b * b + 2 * b * c) - R * T * (b + c),
                      P * b * b * c + R * T * b * c - a * b])
    vs = sorted(r.real for r in roots if abs(r.imag) < 1e-9 and r.real > b)
    if len(vs) < 2:
        return None
    dz = P * (vs[-1] - vs[0]) / (R * T)
    return dz if dz > 0.05 else None


def virial_rackett_dz(T, p_kpa, tc, pc_kpa, omega):
    """Zv from Tsonopoulos B, Zl from Rackett; needs only Tc, Pc, omega."""
    tr = T / tc
    if tr >= 0.995:      # let it break visibly up to 0.99, guard the pole
        return None
    b0 = 0.1445 - 0.330/tr - 0.1385/tr**2 - 0.0121/tr**3 - 0.000607/tr**8
    b1 = 0.0637 + 0.331/tr**2 - 0.423/tr**3 - 0.008/tr**8
    B = R * tc / (pc_kpa * 1e3) * (b0 + omega * b1)
    zv = 1.0 + B * (p_kpa * 1e3) / (R * T)
    zra = 0.29056 - 0.08775 * omega
    vs = R * tc / (pc_kpa * 1e3) * zra ** (1.0 + (1.0 - tr) ** (2.0 / 7.0))
    dz = zv - (p_kpa * 1e3) * vs / (R * T)
    return dz if dz > 0.05 else None


perry = json.load(open(str(REPO_ROOT / "data/perry_properties.json")))["chemicals"]

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

ARMS = ["R1 ref slope, dZ=1",
        "R2 ref slope, dZ=PTV(all real)",
        "N1 NN slope,  dZ=1",
        "Z1 NN slope,  virial, no real criticals",
        "Z2 NN slope,  virial, real TcPc + LK w",
        "Z3 NN slope,  virial, real TcPc + real w",
        "Z4 NN slope,  PTV,    real TcPcVc + LK w",
        "Z5 NN slope,  PTV,    all four real"]
bins = {a: {} for a in ARMS}
crc_bins = {a: {} for a in ARMS}
n_comp = n_acid = 0

crc = {cas: row for cas, row in Hvap_data_CRC.iterrows()}

for cas, row in perry.items():
    if cas == "302-17-0":
        continue
    cc = row.get("critical_constants") or {}
    hv = (row.get("heat_of_vaporization") or [None])[0]
    vp = (row.get("vapor_pressure") or [None])[0]
    if not hv or not vp or not all(cc.get(k) for k in
            ("Tc_K", "Pc_MPa", "Vc_m3_per_kmol")) or cc.get("omega") is None:
        continue
    smi = smiles_for(cas)
    if not smi:
        continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    if mol.HasSubstructMatch(_COOH):
        n_acid += 1
        continue
    tc, pc = cc["Tc_K"], cc["Pc_MPa"] * 1e3
    vc, omega = cc["Vc_m3_per_kmol"], cc["omega"]

    hc = (list(hv["coefficients"]) + [0.0] * 4)[:4]
    hlo, hhi = hv["T_min_K"], hv["T_max_K"]

    def href(T):   # J/mol
        tr = T / tc
        return hc[0] * (1.0 - tr) ** (hc[1] + hc[2]*tr + hc[3]*tr*tr) / 1000.0

    c = (list(vp["coefficients"]) + [0.0] * 5)[:5]
    if c[4] == 0:
        c[4] = 1.0
    plo, phi = vp["T_min_K"], vp["T_max_K"]

    def pref_kpa(T):
        return math.exp(c[0] + c[1]/T + c[2]*math.log(T) + c[3]*T**c[4]) / 1e3

    tb = bisect_T(pref_kpa, 101.325, plo, phi)
    if tb is None:
        continue
    try:
        r_nn = nm.estimate_psat(smi, tb=tb)
        eB = nm.estimate(smi, tb=tb)
        if r_nn.db is None or eB.tc_K is None or eB.pc_kPa is None:
            continue
        w_est = LK_omega(tb, eB.tc_K, eB.pc_kPa * 1e3)   # no-real-crit tier
        w_lkA = LK_omega(tb, tc, pc * 1e3)               # real TcPc tier
    except Exception:
        continue

    def make_vals(T):
        h = 0.5
        dlnp = (math.log(pref_kpa(T + h)) - math.log(pref_kpa(T - h))) / (2*h)
        base_ref = R * T * T * dlnp
        base_nn = r_nn.dhvap_J_mol(T, dz_vap=1.0)
        p_ref, p_nn = pref_kpa(T), r_nn.psat_kPa(T)
        if base_nn is None or p_nn is None:
            return None
        dz_r2 = ptv_dz(T, p_ref, tc, pc, vc, omega)
        dz_z1 = virial_rackett_dz(T, p_nn, eB.tc_K, eB.pc_kPa, w_est)
        dz_z2 = virial_rackett_dz(T, p_nn, tc, pc, w_lkA)
        dz_z3 = virial_rackett_dz(T, p_nn, tc, pc, omega)
        dz_z4 = ptv_dz(T, p_nn, tc, pc, vc, w_lkA)
        dz_z5 = ptv_dz(T, p_nn, tc, pc, vc, omega)
        return {ARMS[0]: base_ref,
                ARMS[1]: base_ref * dz_r2 if dz_r2 else None,
                ARMS[2]: base_nn,
                ARMS[3]: base_nn * dz_z1 if dz_z1 else None,
                ARMS[4]: base_nn * dz_z2 if dz_z2 else None,
                ARMS[5]: base_nn * dz_z3 if dz_z3 else None,
                ARMS[6]: base_nn * dz_z4 if dz_z4 else None,
                ARMS[7]: base_nn * dz_z5 if dz_z5 else None}

    got = False
    for trt in TR_TARGETS:
        T = trt * tc
        if not (hlo + 1 < T < hhi - 1 and plo + 1 < T < phi - 1):
            continue
        ref = href(T)
        vals = make_vals(T)
        if ref <= 0 or vals is None:
            continue
        for arm, v in vals.items():
            if v is not None:
                bins[arm].setdefault(f"Tr={trt}", []).append(100*(v/ref - 1))
        got = True
    n_comp += got

    crow = crc.get(cas)
    if crow is not None:
        for label, tpt, hcol in (("at Tb", crow.get("Tb"), "HvapTb"),
                                 ("at 298 K", 298.15, "Hvap298")):
            hx = crow.get(hcol)
            if (tpt and hx and hx == hx and hlo < tpt < hhi
                    and plo + 1 < tpt < phi - 1):
                vals = make_vals(tpt)
                if vals is None:
                    continue
                for arm, v in vals.items():
                    if v is not None:
                        crc_bins[arm].setdefault(label, []).append(
                            100*(v/hx - 1))   # chemicals banks: J/mol

out = [f"Hvap benchmark: {n_comp} Perry compounds "
       f"({n_acid} carboxylic acids excluded: vapor dimerization)"]
out.append("\nvs Perry 2-150 correlations (MAPE % | bias %):")
out.append(f"{'arm':44s}" + "".join(f" {f'Tr={t}':>15s}" for t in TR_TARGETS))
for a in ARMS:
    cells = []
    for t in TR_TARGETS:
        v = bins[a].get(f"Tr={t}", [])
        cells.append(f"{sum(abs(x) for x in v)/len(v):7.1f}|{sum(v)/len(v):+7.1f}"
                     f"({len(v)})" if v else f"{'-':>13s}")
    out.append(f"{a:44s} " + " ".join(f"{cell:>20s}" for cell in cells))
out.append("\nvs CRC experimental points:")
for a in ARMS:
    cells = []
    for label in ("at Tb", "at 298 K"):
        v = crc_bins[a].get(label, [])
        cells.append(f"{label}: n={len(v)} MAPE {sum(abs(x) for x in v)/len(v):5.2f} "
                     f"bias {sum(v)/len(v):+5.2f}" if v else f"{label}: -")
    out.append(f"{a:44s} " + "   ".join(cells))
report = "\n".join(out)
print(report)
with open(str(REPO_ROOT / "outputs/nannoolal_hvap_benchmark.txt"), "w") as fh:
    fh.write(report + "\n")
