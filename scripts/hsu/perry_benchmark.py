"""Acceptance benchmark for hsu_method: all Perry 2-313 liquid-viscosity
compounds, real Pc, native fragmentation engine.

Reports coverage (vs ~230/340 for the previous ugropy-mapping route),
MAPE distribution over each compound's valid T window (Tr <= tr_max),
rejection-reason histogram, and error-vs-method-factor calibration.

Writes outputs/hsu_perry_benchmark.txt.
"""
from pathlib import Path
import json
import math
import sys
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import hsu_method as hm
from chemicals.identifiers import search_chemical
from chemicals.phase_change import Tm as Tm_chem

perry = json.load(open(str(REPO_ROOT / "data/perry_properties.json")))["chemicals"]

_sc = {}
def smiles_for(cas):
    if cas not in _sc:
        try:
            _sc[cas] = search_chemical(cas).smiles
        except Exception:
            _sc[cas] = None
    return _sc[cas]

results, rejects = [], Counter()
no_input = 0
for cas, row in perry.items():
    lv = row.get("liquid_viscosity")
    cc = row.get("critical_constants") or {}
    if not lv or not cc.get("Tc_K") or not cc.get("Pc_MPa"):
        no_input += 1
        continue
    smi = smiles_for(cas)
    if not smi:
        no_input += 1
        continue
    tc, pc_kpa = cc["Tc_K"], cc["Pc_MPa"] * 1e3
    try:
        res = hm.estimate_viscosity(smi, pc_kPa=pc_kpa, tc_K=tc)
    except hm.HsuFragmentationError as exc:
        reason = str(exc)
        for key in ("polyhydric", "ring ether", "ring ketone", "amide",
                    "heterocycle", "diaryl", "mixed halogen", "polyene",
                    "fused aromatic", "ring amine", "unassigned",
                    "sulfone", "polycarboxylic", "hydroxy-acid",
                    "secondary", "ring quaternary", "methane", "element",
                    "disulfide", "lactone", "cyclic", "peroxide",
                    "unsupported"):
            if key in reason:
                rejects[key] += 1
                break
        else:
            rejects["other"] += 1
        continue

    rec = lv[0]
    co = (list(rec["coefficients"]) + [0.0] * 5)[:5]
    if co[4] == 0:
        co[4] = 1.0
    tm = Tm_chem(cas)
    lo = max(rec["T_min_K"], 0.30 * tc, res.fragmentation.tr_min * tc,
             (tm + 3.0) if tm else 0.0)
    hi = min(rec["T_max_K"], res.fragmentation.tr_max * tc)
    if hi - lo < 5:
        rejects["no T overlap"] += 1
        continue
    errs = []
    for k in range(8):
        T = lo + (hi - lo) * k / 7
        ref = math.exp(co[0] + co[1]/T + co[2]*math.log(T)
                       + (co[3]*T**co[4] if co[3] else 0.0)) * 1e3
        calc = res.viscosity_mPa_s(T)
        if calc is None or not (ref > 0 and math.isfinite(ref)):
            continue
        errs.append(100.0 * abs(calc / ref - 1.0))
    if not errs:
        rejects["no T overlap"] += 1
        continue
    mape = sum(errs) / len(errs)
    results.append((mape, row.get("name", cas),
                    res.fragmentation.method_factor,
                    res.fragmentation.d_sum))

results.sort(reverse=True)
mapes = sorted(m for m, *_ in results)
n = len(mapes)
out = ["Hsu native-engine benchmark vs Perry 2-313 correlations",
       f"eligible (correlation + criticals + SMILES): {len(perry) - no_input}",
       f"covered: {n}   rejected: {sum(rejects.values())}",
       "",
       f"MAPE over valid window:  mean {sum(mapes)/n:.1f}%   "
       f"median {mapes[n//2]:.1f}%   p90 {mapes[int(0.9*n)]:.1f}%",
       "",
       "error vs method factor (quality calibration):"]
buckets = defaultdict(list)
for mape, name, factor, dsum in results:
    buckets[round(factor, 2)].append(mape)
for f in sorted(buckets, reverse=True):
    v = sorted(buckets[f])
    out.append(f"  factor {f:.2f}: n={len(v):3d}  mean {sum(v)/len(v):6.1f}%  "
               f"median {v[len(v)//2]:6.1f}%")
out.append("")
out.append("worst 15:")
for mape, name, factor, dsum in results[:15]:
    out.append(f"  {name:34s} {mape:7.1f}%  factor {factor:.2f}  Sd {dsum:+6.2f}")
out.append("")
out.append("rejection reasons:")
for key, cnt in rejects.most_common():
    out.append(f"  {key:22s} {cnt}")
report = "\n".join(out)
print(report)
with open(str(REPO_ROOT / "outputs/hsu_perry_benchmark.txt"), "w") as fh:
    fh.write(report + "\n")
