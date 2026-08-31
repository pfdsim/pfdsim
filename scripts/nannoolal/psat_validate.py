"""Validate the Psat layer against the 8 worked examples of [3] (Tables 15a-h)."""
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from nannoolal_method import estimate_psat

# (name, smiles, expected groups, expected corrections, dB, Tb, T, Ps_calc[kPa])
CASES = [
    ("15a alpha-pinene", "CC1=CCC2CC1C2(C)C",
     {1: 3, 9: 2, 10: 2, 11: 1, 62: 1}, {125: 1, 132: 2},
     0.1078596, 429.00, 388.15, 31.03),
    ("15b 1,2-ethanediol", "OCCO",
     {7: 2, 36: 2}, {},
     1.1310491, 470.50, 410.65, 13.05),
    ("15c perfluoro-2-propanone", "O=C(C(F)(F)F)C(F)(F)F",
     {7: 2, 21: 6, 51: 1}, {120: 1, 121: 2, 123: 1},
     0.1179392, 245.90, 210.16, 14.63),
    ("15d acrylic acid", "C=CC(=O)O",
     {44: 1, 61: 1}, {134: 1},
     0.9163297, 413.60, 344.15, 6.52),
    ("15e glycol monoacetate", "CC(=O)OCCO",
     {1: 1, 7: 2, 36: 1, 45: 1}, {},
     0.5111422, 458.65, 352.65, 2.24),
    ("15f dipropyl succinate", "CCCOC(=O)CCC(=O)OCCC",
     {1: 2, 4: 4, 7: 2, 45: 2}, {},
     0.9878298, None, None, None),   # anchor from psat point, see below
    # 15g diethanolamine: paper prints dB 1.7493490, but its own eq. (8)
    # arithmetic gives 1.6131919 (sign slip in the example; see module notes)
    ("15g diethanolamine", "OCCNCCO",
     {7: 4, 36: 2, 42: 1}, {},
     1.6131919, 541.15, 401.13, 0.4039),
    ("15h R122", "ClC(Cl)C(F)(F)Cl",
     {7: 2, 21: 2, 26: 2, 27: 1}, {121: 1, 124: 1},
     0.0447374, 344.25, 297.46, 17.5093),
]

fails = 0
for name, smi, exp_groups, exp_corr, exp_db, tb, T, ps in CASES:
    r = estimate_psat(smi, tb=tb)
    ok = True
    if r.groups != exp_groups:
        print(f"FAIL {name}: groups {r.groups} != {exp_groups}")
        ok = False
    if r.corrections != exp_corr:
        print(f"FAIL {name}: corrections {r.corrections} != {exp_corr}")
        ok = False
    if r.db is None or abs(r.db - exp_db) > 5e-7:
        print(f"FAIL {name}: dB {r.db} != {exp_db}")
        ok = False
    if T is not None and r.db is not None:
        p = r.psat_kPa(T)
        if abs(p - ps) > 0.01 * max(1.0, ps):
            print(f"FAIL {name}: Psat({T}) = {p:.4f} != {ps}")
            ok = False
        # round trip
        t_back = r.temperature_K(p)
        if abs(t_back - T) > 1e-6:
            print(f"FAIL {name}: round-trip T {t_back} != {T}")
            ok = False
    if ok:
        extra = ""
        if T is not None and r.db is not None:
            extra = f"  Psat({T} K) = {r.psat_kPa(T):.4f} kPa (paper-style {ps})"
        print(f"ok   {name}: dB = {r.db:.7f}{extra}")
    fails += not ok

# 15f second half: Tb back-calculated from (374.65 K, 0.4 kPa) -> 520.85 K
r = estimate_psat("CCCOC(=O)CCC(=O)OCCC", psat_point=(374.65, 0.4))
print(f"15f Tb from point: {r.tb_K:.2f} K (paper 520.85, exp 523.95)"
      f"  [{r.tb_source}]")
if abs(r.tb_K - 520.85) > 0.02:
    print("FAIL 15f Tb-from-point")
    fails += 1

# dHvap sanity: n-heptane at Tb (exp 31.77 kJ/mol; dz~0.93 there)
r = estimate_psat("CCCCCCC", tb=371.57)
dh = r.dhvap_J_mol(371.57, dz_vap=0.93)
print(f"n-heptane dHvap(Tb) = {dh/1000:.2f} kJ/mol (exp 31.77)")

# mono-functional alcohol warning
r = estimate_psat("CCO", tb=351.44)
print("ethanol warnings:", [w for w in r.warnings if "alcohol" in w])
print("ethanol Psat(298.15) =", round(r.psat_kPa(298.15), 3), "kPa (exp ~7.87)")

# group without dB -> None + warning (methyl thioacetate, group 109)
r = estimate_psat("CC(=O)SC", tb=369.8)
print("thioacetate: db =", r.db, "| warn:",
      [w for w in r.warnings if "psat" in w])

sys.exit(1 if fails else 0)
