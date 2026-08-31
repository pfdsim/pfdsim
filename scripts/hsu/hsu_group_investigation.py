"""Hsu group investigation: excluded / caution-flagged groups vs benchmarks.

For each probe compound, evaluate ln(eta/mPa.s) = sum_i N_i(a_i + b_i T
+ c_i/T^2 + d_i ln Pc[bar]) under competing interpretations (printed
coefficients, decimal/sign repairs, alternative fragmentations, pair
gating), against reference curves: Perry 2-313 (eq 101), VDI PPDS9,
Viswanath-Natarajan 3/2, Dutt-Prasad.  Validity: Tr < 0.75, 1 atm.

Reports MAPE / bias over the overlap grid + Sum(N_i d_i) (Pc sensitivity:
error inflation ~ |sum d| * sigma_lnPc).
"""
from pathlib import Path
import json
import math
import warnings

warnings.filterwarnings("ignore")
import chemicals.viscosity as cv
from chemicals.critical import Pc as Pc_chem, Tc as Tc_chem
from chemicals.identifiers import search_chemical
from chemicals.phase_change import Tm as Tm_chem
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
import nannoolal_method as nnm

# ---- printed Hsu Tables 2-4: Q -> (a, b*1e2, c*1e-4, d) --------------------
Q = {
 1: (-1.7296, -1.0563, 0.8928, -0.0019), 2: (0.0570, -0.2383, 0.7556, -0.1765),
 3: (-0.1497, 0.0060, 1.4157, 0.0751), 4: (-2.2942, 0.4028, 4.5094, 0.6679),
 5: (1.0031, -0.3677, -6.0316, 1.1972), 6: (0.9256, -0.2656, 0.9860, -0.4417),
 7: (1.3365, 0.1612, 1.9408, 0.2507), 8: (-3.5020, 0.4305, 3.1287, 1.0465),
 9: (87.6040, -0.1106, 4.4245, -24.1836), 10: (-91.6154, -0.0111, 0.3265, 25.0542),
 11: (6.0416, -0.1778, 0.8437, -1.5184), 12: (-33.8745, 0.7637, 7.2433, 8.5951),
 13: (1.2028, -0.0120, 2.0143, -0.3677), 14: (-56.2158, 1.7694, 19.0452, 13.3885),
 15: (-0.8570, -0.0098, 2.4376, 0.1311), 16: (0.7896, -0.0231, -0.9222, 0.1928),
 17: (2.0973, 0.0444, 8.1690, -0.4351), 18: (0.4392, 0.0683, 8.8426, -0.1685),
 19: (27.3350, 1.2165, 34.2857, -11.6500), 20: (14.2586, -0.8665, -14.7474, -2.7574),
 21: (5.7852, -0.5310, 9.5499, -1.0300), 22: (1.4351, -1.0010, 13.8366, 0.3418),
 23: (-2.6895, -0.3645, 29.8404, 0.4246), 24: (-18.5630, 2.4275, 78.5417, 0.9650),
 25: (16.7808, 0.8509, 77.1759, -6.9285), 26: (-0.0125, -0.3634, 23.2329, -0.0172),
 27: (-2.0856, 0.6362, 50.0840, -1.0539), 28: (-2.6991, -0.4377, 17.2243, 0.7139),
 29: (-0.7185, 0.0985, 2.9405, 0.1149), 30: (-29.8045, -0.2847, -4.3145, 8.3131),
 31: (-2.3454, 0.0872, 6.4296, 0.5389), 32: (-0.8288, -0.2612, 3.7241, 0.2386),
 33: (-2.6622, 0.1142, 6.7008, 0.7348), 34: (45.9143, -0.2405, 3.8828, -12.4994),
 35: (-2.7291, 0.0413, 27.4079, 0.0002), 36: (-4.0451, -0.1841, 12.6878, 1.1139),
 37: (-0.6721, -0.1693, 20.0309, 0.0279), 38: (-3.3731, -0.0113, 9.4694, 0.6071),
 39: (-0.0635, -0.2162, 1.9325, 0.4686), 40: (-2.5390, 0.0006, 5.4231, 0.8717),
 41: (-5.4872, 1.5834, 34.5474, -0.4244), 42: (-11.8236, 0.0111, 7.2831, 3.6587),
 43: (-8.0314, 0.2848, 9.3746, 2.1486), 44: (-16.9531, 1.0614, 49.1049, 2.8583),
 45: (-13.0333, 0.1801, 12.9392, 2.8987), 46: (-1.9653, 0.1322, 15.8672, -0.0701),
 47: (-1.2954, 0.0427, 12.1837, -0.0948), 48: (-3.2767, 0.0779, 4.4123, 0.9549),
 49: (-2.1030, -0.0965, 6.0066, 0.3464), 50: (-0.2481, -0.3285, 1.9387, 0.1148),
 51: (-12.3498, 1.2621, 23.1473, 1.3950), 52: (-15.2678, 0.5248, 14.2694, 3.7646),
 53: (3.7475, -1.2592, -23.9353, 0.8329), 54: (-32.8607, 0.6232, 27.5184, 7.7525),
 55: (-1.1345, -0.2126, 7.0544, 0.1336), 56: (-6.9489, -0.1723, 5.7804, 1.6467),
 57: (-2.1403, 0.4842, 6.1893, 0.4718), 58: (-6.3646, -0.0180, 23.2752, 1.0653),
 59: (-1.7592, 0.2208, 14.9707, 0.1171), 60: (-1.2982, 0.5975, 14.0415, -0.0031),
 61: (-1.5435, -0.2774, 31.8007, 0.0001), 62: (-8.1097, 0.0432, 20.9135, 1.8795),
 63: (-122.3280, 26.4615, 394.1670, 0.3530), 64: (-6.7363, 0.1316, 45.5193, 1.2172),
 65: (8.9977, 1.5664, 60.8742, -4.6399), 66: (17.8400, -4.5188, -62.0987, 1.2353),
 67: (-10.1316, 0.6712, 37.9465, 1.9199), 68: (-0.1589, 0.1910, 12.0578, -0.0276),
 69: (-4.7601, 0.1120, 6.98437, 0.9719), 70: (-2.7194, -0.1324, 7.7955, 0.6293),
 71: (0.9435, -0.0086, 8.6310, -0.6443), 72: (-1.7997, -0.3851, 3.0118, 0.5524),
 73: (1.5851, -0.1934, 3.7798, -0.4748), 74: (-3.0561, -1.0770, 0.1882, 1.2223),
 75: (-1.3357, -0.3220, 8.8683, 0.1702), 76: (4.2070, -0.4130, 13.3194, -1.1972),
 77: (-0.3083, -0.0623, 4.1382, -0.2644), 78: (-9.4982, 0.2607, 11.3406, 1.8461),
 79: (-10.3980, -1.1189, 1.3134, 2.6681), 80: (1.5394, 0.8465, 17.8121, -2.9915),
 81: (0.4079, -0.2352, -0.1505, -0.2893), 82: (-0.8565, -0.3682, 4.6451, -0.0751),
 83: (-3.4552, -0.5629, 3.6831, 0.3613), 84: (54.2824, 0.0109, 5.9474, -14.5771),
 85: (-2.1710, 0.1403, 10.3743, -1.1972), 86: (-0.7586, -0.6623, -2.4228, 0.7385),
 87: (-279.0030, -0.3420, 1.4253, 73.6293), 88: (-8.1919, -0.1635, 3.0150, 0.0621),
 89: (-1.4672, -0.2787, 4.3362, 0.5635), 90: (70.9918, -0.0245, 7.2061, -18.9106),
 91: (-2.3300, -0.0470, 8.2815, 0.4485),
}

def ln_eta(groups, T, pc_bar, over=None):
    a = b = c = d = 0.0
    for q, n in groups.items():
        qa, qb, qc, qd = (over or {}).get(q, Q[q])
        a += n * qa; b += n * qb * 1e-2; c += n * qc * 1e4; d += n * qd
    return a + b * T + c / T**2 + d * math.log(pc_bar), d

# ---- reference curves ------------------------------------------------------
perry = json.load(open(str(REPO_ROOT / 'data/perry_properties.json')))['chemicals']

def perry_ref(cas):
    row = perry.get(cas)
    lv = (row or {}).get('liquid_viscosity')
    if not lv:
        return None
    rec = lv[0]
    co = (list(rec['coefficients']) + [0.0] * 5)[:5]
    if co[4] == 0:
        co[4] = 1.0
    def f(T, co=co):
        return math.exp(co[0] + co[1]/T + co[2]*math.log(T)
                        + (co[3]*T**co[4] if co[3] else 0.0)) * 1e3   # mPa.s
    return f, rec['T_min_K'], rec['T_max_K'], 'Perry'

def bank_ref(cas):
    # unit conventions verified against Perry overlap in the sanity block:
    # VDI PPDS9 -> Pa.s (*1e3); VN3 bank -> cP as-is; VN2 -> *10.
    if cas in cv.mu_data_VDI_PPDS_7.index:
        r = cv.mu_data_VDI_PPDS_7.loc[cas]
        return (lambda T, r=r: cv.PPDS9(T, r['A'], r['B'], r['C'], r['D'], r['E']) * 1e3,
                None, None, 'VDI')
    if cas in cv.mu_data_VN3.index:
        r = cv.mu_data_VN3.loc[cas]
        return (lambda T, r=r: cv.Viswanath_Natarajan_3(T, r['A'], r['B'], r['C']),
                r['Tmin'], r['Tmax'], 'VN3')
    if cas in cv.mu_data_VN2.index:
        r = cv.mu_data_VN2.loc[cas]
        return (lambda T, r=r: cv.Viswanath_Natarajan_2(T, r['A'], r['B']) * 10.0,
                r['Tmin'], r['Tmax'], 'VN2')
    if cas in cv.mu_data_Dutt_Prasad.index and DUTT_FORM is not None:
        r = cv.mu_data_Dutt_Prasad.loc[cas]
        return (lambda T, r=r: DUTT_FORM(T, r['A'], r['B'], r['C']),
                r['Tmin'], r['Tmax'], 'Dutt')
    return None

# resolve the Dutt-Prasad functional form against Perry overlap compounds
DUTT_FORM = None
def _resolve_dutt():
    global DUTT_FORM
    candidates = {
        '10^(A+B/(T+C)) cP': lambda T, A, B, C: 10.0**(A + B / (T + C)),
        '10^(A+B/(T-C)) cP': lambda T, A, B, C: 10.0**(A + B / (T - C)),
        'exp(A+B/(T+C)) cP': lambda T, A, B, C: math.exp(A + B / (T + C)),
        '10^(A+B/(T+C)) Pa.s*1e3': lambda T, A, B, C: 10.0**(A + B / (T + C)) * 1e3,
    }
    tests = []
    for nm in ('phenol', 'tetrahydrofuran', '1,4-dioxane', 'diethylamine',
               'triethylamine', 'toluene'):
        try:
            cas = search_chemical(nm).CASs
        except Exception:
            continue
        pref, drow = perry_ref(cas), None
        if cas in cv.mu_data_Dutt_Prasad.index:
            drow = cv.mu_data_Dutt_Prasad.loc[cas]
        if pref and drow is not None:
            f, tlo, thi, _ = pref
            T = 0.5 * (max(tlo, drow['Tmin']) + min(thi, drow['Tmax']))
            tests.append((f(T), drow, T))
    best, best_err = None, 0.35
    for lbl, fn in candidates.items():
        errs = []
        for truth, r, T in tests:
            try:
                v = fn(T, r['A'], r['B'], r['C'])
                errs.append(abs(math.log(v / truth)))
            except (ValueError, OverflowError, ZeroDivisionError):
                errs.append(99.0)
        if tests and sum(errs) / len(errs) < best_err:
            best, best_err = (lbl, fn), sum(errs) / len(errs)
    if best:
        print(f'  [Dutt-Prasad form resolved: {best[0]}, '
              f'mean |ln err| {best_err:.3f} on {len(tests)} compounds]')
        DUTT_FORM = best[1]
    else:
        print('  [Dutt-Prasad form unresolved -> bank dropped]')

def crit(cas):
    row = perry.get(cas)
    cc = (row or {}).get('critical_constants') or {}
    tc = cc.get('Tc_K') or Tc_chem(cas)
    pc = cc.get('Pc_MPa') * 10.0 if cc.get('Pc_MPa') else (
        Pc_chem(cas) / 1e5 if Pc_chem(cas) else None)
    return tc, pc

# ---- probe definitions -----------------------------------------------------
# (family, name, {variant_label: groups} [, coeff_overrides {label: {q: tuple}}])
PAIR = {9, 10}
CASES = []
def case(fam, name, variants, overrides=None):
    CASES.append((fam, name, variants, overrides or {}))

# controls (kept groups; calibrate what "good" looks like)
case('control', 'n-hexane', {'std': {2: 2, 3: 4}})
case('control', 'toluene', {'std': {15: 5, 16: 1, 2: 1}})
case('control', '1-butanol', {'std': {2: 1, 3: 3, 22: 1}})
case('control', 'cyclohexane', {'std': {11: 6}})
case('control', 'methylcyclohexane', {'std': {11: 5, 12: 1, 2: 1}})
case('control', 'phenol', {'std': {15: 5, 16: 1, 27: 1}})
case('control', 'o-cresol', {'std': {15: 4, 16: 2, 2: 1, 27: 1}})

for n_ch2, nm in ((2, '1-pentyne'), (3, '1-hexyne'), (4, '1-heptyne'),
                  (5, '1-octyne'), (7, '1-decyne')):
    case('alkyne-terminal', nm, {'pair(9+10)': {9: 1, 10: 1, 3: n_ch2, 2: 1}})
# gauge-fixed Q10: d absorbed at the implied anchor Pc*=39.4 bar
# (a' = a + d*ln(39.4); consistent across all 4 internal alkynes in run 2)
Q10FIX = {'Q10 gauge@39.4': {10: (-91.6154 + 25.0542 * math.log(39.4),
                                  -0.0111, 0.3265, 0.0)}}
case('alkyne-internal', '2-butyne', {'2xQ10': {10: 2, 2: 2},
                                     'unit(9+10)': {9: 1, 10: 1, 2: 2}}, Q10FIX)
case('alkyne-internal', '2-pentyne', {'2xQ10': {10: 2, 2: 2, 3: 1},
                                      'unit(9+10)': {9: 1, 10: 1, 2: 2, 3: 1}}, Q10FIX)
case('alkyne-internal', '3-hexyne', {'2xQ10': {10: 2, 2: 2, 3: 2},
                                     'unit(9+10)': {9: 1, 10: 1, 2: 2, 3: 2}}, Q10FIX)
case('alkyne-internal', '2-hexyne', {'2xQ10': {10: 2, 2: 2, 3: 2},
                                     'unit(9+10)': {9: 1, 10: 1, 2: 2, 3: 2}}, Q10FIX)

case('diol', 'ethylene glycol', {'2xQ26': {3: 2, 26: 2}, '1xQ26': {3: 2, 26: 1},
                                 'as-prim(Q21)': {3: 2, 21: 2}})
case('diol', '1,2-propylene glycol', {'2xQ26': {2: 1, 4: 1, 3: 1, 26: 2}})
case('diol', '1,3-butanediol', {'2xQ26': {2: 1, 4: 1, 3: 2, 26: 2}})
case('diol', '1,3-propanediol', {'2xQ26': {3: 3, 26: 2}})
case('diol', 'glycerol', {'3xQ26': {3: 2, 4: 1, 26: 3}})
case('diol', 'diethylene glycol', {'2xQ26+etherO': {3: 4, 29: 1, 26: 2},
                                   '2xQ28+etherO': {3: 4, 29: 1, 28: 2}})

case('alkoxyalcohol', '2-methoxyethanol', {'Q28': {2: 1, 3: 2, 29: 1, 28: 1},
                                           'as-prim(Q22)': {2: 1, 3: 2, 29: 1, 22: 1}})
case('alkoxyalcohol', '2-ethoxyethanol', {'Q28': {2: 1, 3: 3, 29: 1, 28: 1},
                                          'as-prim(Q22)': {2: 1, 3: 3, 29: 1, 22: 1}})
case('alkoxyalcohol', '2-butoxyethanol', {'Q28': {2: 1, 3: 5, 29: 1, 28: 1},
                                          'as-prim(Q22)': {2: 1, 3: 5, 29: 1, 22: 1}})

case('aromatic-ether', 'anisole', {'Q31': {15: 5, 16: 1, 31: 1, 2: 1},
                                   'as-Q29': {15: 5, 16: 1, 29: 1, 2: 1}})
case('aromatic-ether', 'phenetole', {'Q31': {15: 5, 16: 1, 31: 1, 3: 1, 2: 1},
                                     'as-Q29': {15: 5, 16: 1, 29: 1, 3: 1, 2: 1}})
case('aromatic-ether', 'diphenyl ether', {'Q31': {15: 10, 16: 2, 31: 1},
                                          'as-Q29': {15: 10, 16: 2, 29: 1},
                                          '2xQ31': {15: 10, 16: 2, 31: 2}})

case('ring-ketone', 'cyclohexanone', {'Q34': {11: 5, 34: 1}, 'as-Q33': {11: 5, 33: 1}})
case('ring-ketone', 'cyclopentanone', {'Q34': {11: 4, 34: 1}, 'as-Q33': {11: 4, 33: 1}})

case('carbonate', 'dimethyl carbonate', {'Q43': {2: 2, 43: 1}})
case('carbonate', 'diethyl carbonate', {'Q43': {2: 2, 3: 2, 43: 1}})

case('sulfoxide', 'dimethyl sulfoxide', {'Q54': {2: 2, 54: 1}})

case('aromatic-amine', 'N-methylaniline', {'Q59': {15: 5, 16: 1, 59: 1, 2: 1}})
case('aromatic-amine', 'diphenylamine', {'Q59': {15: 10, 16: 2, 59: 1}})
case('aromatic-amine', 'N,N-dimethylaniline', {'Q60': {15: 5, 16: 1, 60: 1, 2: 2}})

case('amide', 'formamide', {'Q61': {61: 1}})
case('amide', 'N,N-dimethylformamide', {'Q63': {2: 2, 63: 1}},
     {'Q63 b/100': {63: (-122.3280, 0.264615, 394.1670, 0.3530)},
      'Q63 a/10': {63: (-12.23280, 26.4615, 394.1670, 0.3530)},
      'Q63 a/10,b/10,c/10': {63: (-12.23280, 2.64615, 39.41670, 0.3530)}})

case('freon', '1,1,2-trichlorotrifluoroethane',
     {'Q83+Q84': {83: 1, 84: 1}, '+2 carbons': {5: 2, 83: 1, 84: 1}})
case('freon', 'dichlorodifluoromethane',
     {'Q85': {85: 1}, '+carbon': {5: 1, 85: 1}})
case('freon', '1,2-dichlorotetrafluoroethane',
     {'2xQ84': {84: 2}, '+2 carbons': {5: 2, 84: 2}})

case('halogen-carbon', 'dichloromethane', {'Q74 only': {74: 1}, 'Q74+CH2': {3: 1, 74: 1}})
case('halogen-carbon', 'chloroform', {'Q75 only': {75: 1}, 'Q75+CH': {4: 1, 75: 1}})
case('halogen-carbon', '1,1-dichloroethane', {'Q74 only': {2: 1, 74: 1},
                                              'Q74+CH': {2: 1, 4: 1, 74: 1}})

case('br-sec', '2-bromopropane', {'Q87': {2: 2, 4: 1, 87: 1},
                                  'as-Q86': {2: 2, 4: 1, 86: 1}})
case('iodo-ar', 'iodobenzene', {'Q90': {15: 5, 16: 1, 90: 1}},
     {'Q90 gauge@45.3': {90: (70.9918 - 18.9106 * math.log(45.3),
                              -0.0245, 7.2061, 0.0)}})
case('amide', 'acetamide', {'Q64': {2: 1, 64: 1}})
case('acid-chloride', 'acetyl chloride', {'Q91': {2: 1, 91: 1}})
case('acid-chloride', 'propionyl chloride', {'Q91': {2: 1, 3: 1, 91: 1}})

case('ring-ether', 'tetrahydrofuran', {'Q30': {11: 4, 30: 1}})
case('ring-ether', 'tetrahydropyran', {'Q30': {11: 5, 30: 1}})
case('ring-ether', '1,4-dioxane', {'2xQ30': {11: 4, 30: 2}})

case('cl-sec', '2-chloropropane', {'as-Q72(hers)': {2: 2, 4: 1, 72: 1}})

case('amine-3', 'triethylamine', {'Q57': {2: 3, 3: 3, 57: 1}})

D10 = {'Q56 hers(abd/10)': {56: (-0.69489, -0.01723, 5.7804, 0.16467)},
       'Q56 a/10': {56: (-0.69489, -0.1723, 5.7804, 1.6467)},
       'Q56 a,d/10': {56: (-0.69489, -0.1723, 5.7804, 0.16467)}}
case('amine-2', 'diethylamine', {'printed': {2: 2, 3: 2, 56: 1}}, D10)
case('amine-2', 'di-n-propylamine', {'printed': {2: 2, 3: 4, 56: 1}}, D10)
case('amine-2', 'di-n-butylamine', {'printed': {2: 2, 3: 6, 56: 1}}, D10)
case('amine-2', 'diisopropylamine', {'printed': {2: 4, 4: 2, 56: 1}}, D10)

case('fused', 'tetralin', {'Q20': {15: 4, 20: 2, 11: 4},
                           'as-Q16': {15: 4, 16: 2, 11: 4},
                           'as-Q18': {15: 4, 18: 2, 11: 4}})
case('fused', 'decalin', {'8xQ11+2xQ12': {11: 8, 12: 2}})

case('ar-br', 'bromobenzene', {'printed': {15: 5, 16: 1, 88: 1}},
     {'hers(a/10)': {88: (-0.81919, -0.1635, 3.0150, 0.0621)}})

# ---- run -------------------------------------------------------------------
def nn_pc(name):
    try:
        smi = search_chemical(name).smiles
        est = nnm.estimate(smi)
        return est.pc_kPa / 100.0 if est.pc_kPa else None   # bar
    except Exception:
        return None

def eval_case(fam, name, variants, overrides):
    try:
        cas = search_chemical(name).CASs
    except Exception:
        return f'{fam:16s} {name:28s} unidentified'
    ref = perry_ref(cas) or bank_ref(cas)
    if ref is None:
        return f'{fam:16s} {name:28s} no reference data'
    f, tlo, thi, src = ref
    tc, pc = crit(cas)
    if not tc or not pc:
        return f'{fam:16s} {name:28s} no criticals'
    tm = Tm_chem(cas)
    if tlo is None:                              # VDI: no tabulated range
        tlo = (tm + 5.0) if tm else 0.40 * tc
        thi = 0.70 * tc
    lo = max(tlo, 0.30 * tc, (tm + 3.0) if tm else 0.0)
    hi = min(thi, 0.75 * tc)
    if hi - lo < 5:
        return f'{fam:16s} {name:28s} no valid T overlap ({src})'
    grid = [lo + (hi - lo) * k / 7 for k in range(8)]
    out = []
    all_variants = {**{lbl: (g, None) for lbl, g in variants.items()},
                    **{lbl: (next(iter(variants.values())), ov)
                       for lbl, ov in overrides.items()}}
    for lbl, (groups, over) in all_variants.items():
        errs, core, dsum = [], [], 0.0
        for T in grid:
            try:
                le, dsum = ln_eta(groups, T, pc, over)
                calc, refv = math.exp(le), f(T)
            except (OverflowError, ValueError):
                errs = None; break
            if not (refv > 0 and math.isfinite(refv) and math.isfinite(calc)):
                errs = None; break
            e = 100.0 * (calc / refv - 1.0)
            errs.append(e)
            if 0.45 <= T / tc <= 0.65:
                core.append(e)
        if errs is None:
            out.append(f'    {lbl:22s} OVERFLOW/invalid')
            continue
        mape = sum(abs(e) for e in errs) / len(errs)
        bias = sum(errs) / len(errs)
        # shape error: MAPE after removing the geometric-mean offset
        # (small shape + big bias == mis-anchored but salvageable)
        lns = [math.log(1.0 + e / 100.0) for e in errs if e > -100.0]
        if len(lns) == len(errs):
            g = sum(lns) / len(lns)
            shape = sum(abs(math.exp(l - g) - 1.0) for l in lns) / len(lns) * 100
            shp = f'{shape:6.1f}%'
        else:
            shp = '     -'
        cmape = (f'{sum(abs(e) for e in core)/len(core):7.1f}%'
                 if core else '      -')
        diag = ''
        if abs(dsum) >= 3.0 and abs(bias) > 5.0 and 1.0 + bias / 100.0 > 0:
            pc_star = pc * math.exp(-math.log(1.0 + bias / 100.0) / dsum)
            diag = f'  Pc*={pc_star:.1f}'
        big = ' ***' if mape > 25 else ''
        out.append(f'    {lbl:22s} MAPE {mape:8.1f}%  core {cmape}  shp {shp}  '
                   f'bias {bias:+8.1f}%  Sd {dsum:+6.2f}{diag}{big}')
    pcn = nn_pc(name)
    hdr = (f'{fam:16s} {name:28s} ref={src} Pc={pc:.1f}bar'
           f'{f" NNPc={pcn:.1f}" if pcn else ""} '
           f'T {lo:.0f}-{hi:.0f}K (Tr {lo/tc:.2f}-{hi/tc:.2f})')
    return hdr + '\n' + '\n'.join(out)

# sanity: cross-check each bank against Perry on shared compounds
print('=== bank sanity (vs Perry at mid-range, corrected units) ===')
_resolve_dutt()
for nm in ('phenol', 'diethylamine', 'tetrahydrofuran', '1,4-dioxane'):
    cas = search_chemical(nm).CASs
    pref = perry_ref(cas)
    if not pref:
        continue
    f, tlo, thi, _ = pref
    T = 0.5 * (max(tlo, 290) + min(thi, 400))
    row = [f'{nm:18s} T={T:.0f}K Perry={f(T):.3f} mPa.s']
    for bank, df, ev in (
        ('VDI', cv.mu_data_VDI_PPDS_7, lambda r, T: cv.PPDS9(T, r['A'], r['B'], r['C'], r['D'], r['E']) * 1e3),
        ('VN3', cv.mu_data_VN3, lambda r, T: cv.Viswanath_Natarajan_3(T, r['A'], r['B'], r['C'])),
        ('VN2', cv.mu_data_VN2, lambda r, T: cv.Viswanath_Natarajan_2(T, r['A'], r['B']) * 10.0),
        ('Dutt', cv.mu_data_Dutt_Prasad,
         lambda r, T: DUTT_FORM(T, r['A'], r['B'], r['C']) if DUTT_FORM else float('nan')),
    ):
        if cas in df.index:
            try:
                row.append(f'{bank}={ev(df.loc[cas], T):.3f}')
            except Exception:
                row.append(f'{bank}=ERR')
    print('  ' + '  '.join(row))

print('\n=== investigation ===')
last_fam = None
for fam, name, variants, overrides in CASES:
    if fam != last_fam:
        print(f'\n--- {fam} ---')
        last_fam = fam
    print(eval_case(fam, name, variants, overrides))
