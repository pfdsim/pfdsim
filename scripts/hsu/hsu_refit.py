"""Refit experiments for Hsu group families with >=3 benchmark compounds.

Model residual: ln(eta_ref) - skeleton(T, Pc) = n_u*(a + b*T + c/T^2) [+ extras]
-> linear least squares, validated leave-one-compound-out (LOO).

Families:
  A. terminal alkyne unit HC#C-   (5 Perry compounds, d frozen 0)
  B. internal alkyne unit -C#C-   (4 Perry compounds, d frozen 0; vs gauge fix)
  C. polyhydric -OH (per OH) + vicinal-pair penalty (5 compounds, d frozen 0)
  D. secondary amine -NH- polish  (4 compounds, d frozen at PFDSim's 0.16467)
"""
from pathlib import Path
import json
import math
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import chemicals.viscosity as cv
from chemicals.identifiers import search_chemical
from chemicals.phase_change import Tm as Tm_chem

# skeleton coefficients (validated groups; printed values, Q7 flip irrelevant here)
G = {'ch3': (0.0570, -0.2383, 0.7556, -0.1765),
     'ch2': (-0.1497, 0.0060, 1.4157, 0.0751),
     'ch': (-2.2942, 0.4028, 4.5094, 0.6679)}

REPO_ROOT = Path(__file__).resolve().parents[2]
perry = json.load(open(str(REPO_ROOT / 'data/perry_properties.json')))['chemicals']

def perry_curve(cas):
    lv = (perry.get(cas) or {}).get('liquid_viscosity')
    if not lv:
        return None
    rec = lv[0]
    co = (list(rec['coefficients']) + [0.0] * 5)[:5]
    if co[4] == 0:
        co[4] = 1.0
    return (lambda T: math.exp(co[0] + co[1]/T + co[2]*math.log(T)
                               + (co[3]*T**co[4] if co[3] else 0.0)) * 1e3,
            rec['T_min_K'], rec['T_max_K'])

def vdi_curve(cas):
    if cas not in cv.mu_data_VDI_PPDS_7.index:
        return None
    r = cv.mu_data_VDI_PPDS_7.loc[cas]
    return (lambda T: cv.PPDS9(T, r['A'], r['B'], r['C'], r['D'], r['E']) * 1e3,
            None, None)

def crit(cas):
    cc = (perry.get(cas) or {}).get('critical_constants') or {}
    from chemicals.critical import Pc as Pc_c, Tc as Tc_c
    tc = cc.get('Tc_K') or Tc_c(cas)
    pc = cc.get('Pc_MPa') * 10.0 if cc.get('Pc_MPa') else Pc_c(cas) / 1e5
    return tc, pc

def rows_for(name, skeleton, n_units, extra=0):
    """-> list of (T, resid, n_units, extra) plus meta; grid Tr in-range."""
    cas = search_chemical(name).CASs
    curve = perry_curve(cas) or vdi_curve(cas)
    if curve is None:
        return None
    f, tlo, thi = curve
    tc, pc = crit(cas)
    tm = Tm_chem(cas)
    if tlo is None:
        tlo = (tm + 5) if tm else 0.40 * tc
        thi = 0.70 * tc
    lo = max(tlo, 0.32 * tc, (tm + 3) if tm else 0.0)
    hi = min(thi, 0.73 * tc)
    if hi - lo < 10:
        return None
    grid = [lo + (hi - lo) * k / 7 for k in range(8)]
    rows = []
    for T in grid:
        skel = sum(n * (a + b * 1e-2 * T + c * 1e4 / T**2 + d * math.log(pc))
                   for (a, b, c, d), n in skeleton)
        rows.append((T, math.log(f(T)) - skel, n_units, extra))
    return rows, pc, tc

def fit(dataset, use_extra=False):
    """dataset: {name: rows}; solve for (a, b, c[, delta]) weighted per compound."""
    A, y = [], []
    for name, (rows, pc, tc) in dataset.items():
        w = 1.0 / math.sqrt(len(rows))
        for T, resid, nu, ex in rows:
            base = [nu * w, nu * T * w, nu / T**2 * w]
            if use_extra:
                base.append(ex * w)
            A.append(base)
            y.append(resid * w)
    sol, *_ = np.linalg.lstsq(np.array(A), np.array(y), rcond=None)
    return sol

def mape_of(rows, sol, use_extra=False):
    errs = []
    for T, resid, nu, ex in rows:
        pred = nu * (sol[0] + sol[1] * T + sol[2] / T**2)
        if use_extra:
            pred += ex * sol[3]
        errs.append(abs(math.exp(pred - resid) - 1.0) * 100)
    return sum(errs) / len(errs)

def run_family(title, spec, use_extra=False):
    print(f'\n=== {title} ===')
    data = {}
    for name, skeleton, nu, ex in spec:
        got = rows_for(name, skeleton, nu, ex)
        if got:
            data[name] = got
        else:
            print(f'  {name}: no data, skipped')
    if len(data) < 3:
        print('  <3 compounds; refit not meaningful')
        return None
    full = fit(data, use_extra)
    lbl = ('a={:+.4f} b={:+.6f} c={:+.1f}'.format(*full[:3])
           + (f' delta_vic={full[3]:+.4f}' if use_extra else ''))
    print(f'  full fit: {lbl}')
    print(f'  {"compound":26s} {"in-fit":>8s} {"LOO":>8s}')
    loo_all = []
    for name in data:
        sub = {k: v for k, v in data.items() if k != name}
        if len(sub) < 2:
            continue
        sol_l = fit(sub, use_extra)
        m_in = mape_of(data[name][0], full, use_extra)
        m_loo = mape_of(data[name][0], sol_l, use_extra)
        loo_all.append(m_loo)
        print(f'  {name:26s} {m_in:7.1f}% {m_loo:7.1f}%')
    print(f'  LOO mean: {sum(loo_all)/len(loo_all):.1f}%')
    return full

CH3, CH2, CH = G['ch3'], G['ch2'], G['ch']

run_family('A. terminal alkyne unit HC#C- (d=0)', [
    ('1-pentyne', [(CH3, 1), (CH2, 2)], 1, 0),
    ('1-hexyne',  [(CH3, 1), (CH2, 3)], 1, 0),
    ('1-heptyne', [(CH3, 1), (CH2, 4)], 1, 0),
    ('1-octyne',  [(CH3, 1), (CH2, 5)], 1, 0),
    ('1-decyne',  [(CH3, 1), (CH2, 7)], 1, 0),
])

run_family('B. internal alkyne unit -C#C- (d=0)', [
    ('2-butyne',  [(CH3, 2)], 1, 0),
    ('2-pentyne', [(CH3, 2), (CH2, 1)], 1, 0),
    ('3-hexyne',  [(CH3, 2), (CH2, 2)], 1, 0),
    ('2-hexyne',  [(CH3, 2), (CH2, 2)], 1, 0),
])

run_family('C. polyhydric OH per-OH + vicinal pair penalty (d=0)', [
    ('ethylene glycol',      [(CH2, 2)], 2, 1),
    ('1,2-propylene glycol', [(CH3, 1), (CH, 1), (CH2, 1)], 2, 1),
    ('1,3-butanediol',       [(CH3, 1), (CH, 1), (CH2, 2)], 2, 0),
    ('1,3-propanediol',      [(CH2, 3)], 2, 0),
    ('glycerol',             [(CH2, 2), (CH, 1)], 3, 2),
], use_extra=True)

# D: secondary amine -NH-, d frozen at PFDSim's 0.16467 (fold into skeleton)
NH_D = (0.0, 0.0, 0.0, 0.16467)
run_family("D. secondary amine -NH- polish (d frozen at PFDSim's 0.16467)", [
    ('diethylamine',     [(CH3, 2), (CH2, 2), (NH_D, 1)], 1, 0),
    ('di-n-propylamine', [(CH3, 2), (CH2, 4), (NH_D, 1)], 1, 0),
    ('di-n-butylamine',  [(CH3, 2), (CH2, 6), (NH_D, 1)], 1, 0),
    ('diisopropylamine', [(CH3, 4), (CH, 2), (NH_D, 1)], 1, 0),
])
