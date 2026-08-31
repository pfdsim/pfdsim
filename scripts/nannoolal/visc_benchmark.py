"""Nannoolal Part-4 liquid viscosity vs Hsu: predictive and anchored modes.

Arms (saturated liquid, window max(Tmin, Tm+3, 0.3Tc) .. min(Tmax, 0.75Tc)):
  NN-Tb   Nannoolal dBv/Tv, experimental Tb (from the Perry Psat correlation
          at 101.325 kPa, or chemicals Tb for the VDI-only block)
  NN-est  fully predictive (internal Tb estimate)
  NN-anc  anchored: Tv back-calculated from ONE reference point near 298 K
          (that point excluded from scoring)
  Hsu     hsu_method native engine with the real Pc

Per-class table on the common NN+Hsu subset (Perry 2-313); coverage and
overall stats per arm on each arm's own subset.  VDI-PPDS-only compounds
(chemicals bank, unit convention validated against Perry 2026-07-12) are
appended as a second block.  Writes outputs/nannoolal_visc_benchmark.txt.
"""
from pathlib import Path
import json
import math
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import hsu_method
import nannoolal_method as nm
import chemicals.viscosity as cv
from chemicals.identifiers import search_chemical
from chemicals.phase_change import Tm as Tm_chem, Tb as Tb_chem
from chemicals.critical import Tc as Tc_chem, Pc as Pc_chem
from rdkit import Chem

perry = json.load(open(str(REPO_ROOT / "data/perry_properties.json")))["chemicals"]

_CLASS_PATTERNS = [
    ('acid', ['[CX3](=O)[OX2H1]']),
    ('phenol', ['[c][OX2H1]']),
    ('alcohol', ['[CX4][OX2H1]']),
    ('ester', ['[CX3](=O)[OX2][#6]']),
    ('ketone/aldehyde', ['[CX3]=[OX1]']),
    ('ether', ['[OX2]([#6])[#6]']),
    ('amine', ['[NX3;!$(N=O);!$(NC=O)]']),
    ('nitrile/nitro', ['[NX1]#[CX2]', '[$([NX3](=O)=O)]', '[NX3+](=O)[O-]']),
    ('sulfur', ['[#16]']),
    ('halogen', ['[F]', '[Cl]', '[Br]', '[I]']),
    ('aromatic HC', ['[c]']),
    ('cyclic HC', ['[C;R]']),
    ('unsat. HC', ['[$([CX3]=[CX3])]', '[$([CX2]#[CX2])]']),
]
_CLASS_PATTERNS = [(name, [Chem.MolFromSmarts(s) for s in smarts])
                   for name, smarts in _CLASS_PATTERNS]
_FUNCTIONAL_SMARTS = [Chem.MolFromSmarts(s) for s in (
    '[OX2H1]', '[CX3]=[OX1]', '[OX2]([#6])[#6]', '[NX3;!$(N=O)]',
    '[NX1]#[CX2]', '[$([NX3](=O)=O)]', '[NX3+](=O)[O-]', '[#16]')]

_COOH = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')

def classify(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 'other'
    n_func = sum(len(mol.GetSubstructMatches(p)) for p in _FUNCTIONAL_SMARTS)
    # a COOH matches both the OH and the C=O pattern: count it once
    n_func -= len(mol.GetSubstructMatches(_COOH))
    if n_func >= 2:
        return 'multifunctional'
    for name, patterns in _CLASS_PATTERNS:
        if any(mol.HasSubstructMatch(p) for p in patterns):
            return name
    return 'alkane'

_sc = {}
def smiles_for(cas):
    if cas not in _sc:
        try:
            _sc[cas] = search_chemical(cas).smiles
        except Exception:
            _sc[cas] = None
    return _sc[cas]

def bisect_T(f, target, lo, hi):
    try:
        if not (f(lo) <= target <= f(hi)):
            return None
    except (ValueError, OverflowError):
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def mape(vals):
    return sum(vals) / len(vals) if vals else None

def sane_reference(ref, lo, hi):
    """Reject reference curves extrapolated into nonsense: require monotone
    decreasing eta over the window and physical magnitudes."""
    try:
        etas = [ref(lo + (hi - lo) * k / 7) for k in range(8)]
    except (ValueError, OverflowError, ZeroDivisionError):
        return False
    return (all(a > b for a, b in zip(etas, etas[1:]))
            and etas[0] < 5.0e3 and etas[-1] > 1.0e-2)


def eval_compound(smi, ref, lo, hi, tb, tc, pc_bar):
    """ref: eta_mPa_s(T).  Returns dict arm -> MAPE% or None."""
    grid = [lo + (hi - lo) * k / 7 for k in range(8)]
    t_anchor = min(grid, key=lambda t: abs(t - 298.15))
    out = {}
    arms = {}
    try:
        arms['NN-Tb'] = nm.estimate_viscosity(smi, tb=tb) if tb else None
    except nm.NannoolalError:
        arms['NN-Tb'] = None
    try:
        arms['NN-est'] = nm.estimate_viscosity(smi)
    except nm.NannoolalError:
        arms['NN-est'] = None
    try:
        arms['NN-anc'] = nm.estimate_viscosity(
            smi, visc_point=(t_anchor, ref(t_anchor)))
    except (nm.NannoolalError, ValueError, OverflowError):
        arms['NN-anc'] = None
    hsu = None
    try:
        hsu = hsu_method.estimate_viscosity(smi, pc_kPa=pc_bar * 100.0, tc_K=tc)
    except (hsu_method.HsuFragmentationError, ValueError):
        pass

    points = {}
    for arm, res in arms.items():
        if res is None or res.dbv is None or res.tv_K is None:
            out[arm] = None
            continue
        errs, pts = [], []
        for T in grid:
            if arm == 'NN-anc' and abs(T - t_anchor) < 5.0:
                continue
            calc = res.viscosity_mPa_s(T)
            r = ref(T)
            if calc is not None and r > 0:
                e = 100.0 * abs(calc / r - 1.0)
                errs.append(e)
                pts.append((T / tc, e))
        out[arm] = mape(errs)
        points[arm] = pts
    if hsu is None:
        out['Hsu'] = None
    else:
        errs, pts = [], []
        for T in grid:
            if not (hsu.fragmentation.tr_min * tc <= T
                    <= hsu.fragmentation.tr_max * tc):
                continue
            calc = hsu.viscosity_mPa_s(T)
            r = ref(T)
            if calc is not None and r > 0:
                e = 100.0 * (abs(calc / r - 1.0))
                errs.append(e)
                pts.append((T / tc, e))
        out['Hsu'] = mape(errs)
        points['Hsu'] = pts
    return out, points

ARMS = ('NN-Tb', 'NN-est', 'NN-anc', 'Hsu')
rows = []          # (class, name, {arm: mape})
vdi_rows = []
seen_cas = set()

for cas, row in perry.items():
    lv = row.get('liquid_viscosity')
    vp = (row.get('vapor_pressure') or [None])[0]
    cc = row.get('critical_constants') or {}
    if not lv or not vp or not cc.get('Tc_K') or not cc.get('Pc_MPa'):
        continue
    smi = smiles_for(cas)
    if not smi:
        continue
    tc, pc_bar = cc['Tc_K'], cc['Pc_MPa'] * 10.0
    rec = lv[0]
    co = (list(rec['coefficients']) + [0.0] * 5)[:5]
    if co[4] == 0:
        co[4] = 1.0
    def ref(T, co=co):
        return math.exp(co[0] + co[1]/T + co[2]*math.log(T)
                        + (co[3]*T**co[4] if co[3] else 0.0)) * 1e3
    c = (list(vp['coefficients']) + [0.0] * 5)[:5]
    if c[4] == 0:
        c[4] = 1.0
    def pref(T, c=c):
        return math.exp(c[0] + c[1]/T + c[2]*math.log(T) + c[3]*T**c[4]) / 1e3
    tb = bisect_T(pref, 101.325, vp['T_min_K'], vp['T_max_K'])
    tm = Tm_chem(cas)
    lo = max(rec['T_min_K'], 0.30 * tc, (tm + 3.0) if tm else 0.0)
    hi = min(rec['T_max_K'], 0.75 * tc)
    if hi - lo < 10 or not sane_reference(ref, lo, hi):
        continue
    seen_cas.add(cas)
    mapes, points = eval_compound(smi, ref, lo, hi, tb, tc, pc_bar)
    rows.append((classify(smi), row.get('name', cas), mapes, points))

# VDI-only block (chemicals bank; Tb/Tm/criticals from chemicals)
for cas in cv.mu_data_VDI_PPDS_7.index:
    if cas in seen_cas or cas in perry:
        continue
    smi = smiles_for(cas)
    tc, pc, tb, tm = Tc_chem(cas), Pc_chem(cas), Tb_chem(cas), Tm_chem(cas)
    if not smi or not tc or not pc:
        continue
    r = cv.mu_data_VDI_PPDS_7.loc[cas]
    def ref(T, r=r):
        return cv.PPDS9(T, r['A'], r['B'], r['C'], r['D'], r['E']) * 1e3
    lo = max((tm + 5.0) if tm else 0.40 * tc, 0.30 * tc)
    hi = 0.72 * tc
    if hi - lo < 10 or not sane_reference(ref, lo, hi):
        continue
    try:
        mapes, points = eval_compound(smi, ref, lo, hi, tb, tc, pc / 1e5)
        vdi_rows.append((classify(smi), str(r.get('Chemical', cas)),
                         mapes, points))
    except Exception:
        continue

def block(title, data, per_class):
    out = [f'== {title}: {len(data)} compounds ==']
    hdr = f'{"":24s}' + ''.join(f'{a:>18s}' for a in ARMS)
    out.append(hdr)
    def stats(sub, arm):
        v = sorted(x[2][arm] for x in sub if x[2][arm] is not None)
        if not v:
            return f'{"-":>18s}'
        return f'{sum(v)/len(v):7.1f}/{v[len(v)//2]:6.1f} ({len(v):3d})'
    out.append(f'{"ALL (own subset)":24s}'
               + ''.join(stats(data, a) for a in ARMS))
    common = [x for x in data if all(x[2][a] is not None for a in ARMS)]
    out.append(f'{"COMMON subset":24s}'
               + ''.join(stats(common, a) for a in ARMS))
    if per_class:
        by_class = defaultdict(list)
        for x in common:
            by_class[x[0]].append(x)
        out.append('per class (common subset, mean/median (n)):')
        for cls in sorted(by_class, key=lambda c: -len(by_class[c])):
            sub = by_class[cls]
            if len(sub) < 3:
                continue
            out.append(f'  {cls:22s}'
                       + ''.join(stats(sub, a) for a in ARMS))
    return out

def tr_profile(data):
    """Per-point error binned by reduced temperature, common subset."""
    bins = [(0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70),
            (0.70, 0.76)]
    common = [x for x in data if all(x[2][a] is not None for a in ARMS)]
    out = ['per-point error vs Tr (common subset, mean/median (n points)):']
    out.append(f'{"Tr bin":24s}' + ''.join(f'{a:>18s}' for a in ARMS))
    for blo, bhi in bins:
        cells = []
        for arm in ARMS:
            es = sorted(e for x in common for tr, e in x[3].get(arm, [])
                        if blo <= tr < bhi)
            cells.append(f'{sum(es)/len(es):7.1f}/{es[len(es)//2]:6.1f}'
                         f' ({len(es):3d})' if es else f'{"-":>18s}')
        out.append(f'  Tr {blo:.2f}-{bhi:.2f}          ' + ''.join(cells))
    return out

out = ['Nannoolal Part-4 viscosity vs Hsu (mean/median MAPE %, n)', '']
out += block('Perry 2-313', rows, per_class=True)
out.append('')
out += tr_profile(rows)
out.append('')
out += block('VDI-PPDS only (chemicals)', vdi_rows, per_class=False)
report = '\n'.join(out)
print(report)
with open(str(REPO_ROOT / 'outputs/nannoolal_visc_benchmark.txt'), 'w') as fh:
    fh.write(report + '\n')
