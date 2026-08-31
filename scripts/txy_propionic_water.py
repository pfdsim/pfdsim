"""T-xy for water-propionic acid — UNIQUAC (fixed Antoine + VDM comparison)"""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
from thermodynamics import UNIQUACThermodynamics
from unifac import UNIFACModel
from vapor_dimerization import VaporDimerizationModel, get_dimerization_params
from chemical_properties import get_database

# --- Fix the bad Antoine BEFORE creating thermo model ---
db = get_database()
props = db.get('C2H5COOH')
props.antoine_A, props.antoine_B, props.antoine_C = 4.74558, 1679.869, 213.318
props.antoine_Tmin, props.antoine_Tmax = 345.54, 401.49

# --- Custom UNIQUAC with r/q from UNIFAC ---
r_pa, q_pa = 2.8768, 2.6120  # {CH3:1, CH2:1, COOH:1}
class CustomUNIQUAC(UNIQUACThermodynamics):
    def _uniquac_rq(self, comp):
        if comp == 'C2H5COOH': return r_pa, q_pa
        return super()._uniquac_rq(comp)

thermo = CustomUNIQUAC(['C2H5COOH', 'H2O'])
P_atm = 1.01325

# --- Verify Psat ---
for Tc in [57, 100, 141]:
    psat = thermo.Psat('C2H5COOH', Tc + 273.15)
    print(f'Psat at {Tc:4d}°C: {psat:.6f} bar')

# --- UNIQUAC-only T-xy ---
data_uni = thermo.generate_Txy_data('C2H5COOH', 'H2O', P_atm, n_points=80)
azo_uni = data_uni.get('azeotrope')

# --- UNIQUAC + VDM ---
params = get_dimerization_params('C2H5COOH')
vdm = VaporDimerizationModel('C2H5COOH', '(C2H5COOH)2',
                             delta_S=params['delta_S_J_per_mol_K'],
                             delta_H=params['delta_H_J_per_mol'])

def phi_sat(vdm, comp, T):
    if comp != 'C2H5COOH': return 1.0
    Psat = thermo.Psat(comp, T)
    phi = vdm.fugacity_coefficients(T, Psat, {'C2H5COOH': 1.0, 'H2O': 0.0}, rk_model=None)
    return phi.get(comp, 1.0)

def bubble_T_vdm(vdm, x, P, T_guess=380):
    T = T_guess; comps = list(x.keys())
    for _ in range(60):
        gamma = thermo.activity_coefficients(T, x)
        Psat = {c: thermo.Psat(c, T) for c in comps}
        phi_s = {c: phi_sat(vdm, c, T) for c in comps}
        K = {c: gamma[c] * phi_s[c] * Psat[c] / P for c in comps}
        y = {c: x[c] * K[c] for c in comps}
        s = sum(y.values()); y = {c: yi / s for c, yi in y.items()}
        for _ in range(15):
            phi_v = vdm.fugacity_coefficients(T, P, y, rk_model=None)
            K = {c: gamma[c] * phi_s[c] * Psat[c] / max(phi_v.get(c, 1.0) * P, 1e-12)
                 for c in comps}
            y_new = {c: x[c] * K[c] for c in comps}
            s = sum(y_new.values()); y_new = {c: yi / s for c, yi in y_new.items()}
            if max(abs(y_new[c] - y[c]) for c in comps) < 1e-10:
                y = y_new; break
            y = y_new
        sum_xK = sum(x[c] * K[c] for c in comps)
        if abs(sum_xK - 1.0) < 1e-7: return T, y
        dT = -5.0 * (sum_xK - 1.0) if sum_xK > 1 else 5.0 * (1.0 - sum_xK)
        T += dT; T = max(200, min(700, T))
    return T, y

# --- Print ---
print()
print(f'{"x(PA)":>8s}  {"y(UNI)":>8s}  {"y(VDM)":>8s}  {"Tb(UNI)":>8s}  {"Tb(VDM)":>8s}  {"phiV":>8s}')
print('-' * 60)
for i in range(0, len(data_uni['x']), 8):
    x1 = data_uni['x'][i]
    x = {'C2H5COOH': x1, 'H2O': 1 - x1}
    Tb_v, y_v = bubble_T_vdm(vdm, x, P_atm)
    phi_v = vdm.fugacity_coefficients(Tb_v, P_atm, y_v, rk_model=None)
    print(f'{x1:8.4f}  {data_uni["y"][i]:8.4f}  {y_v["C2H5COOH"]:8.4f}  '
          f'{data_uni["T_bubble"][i]:8.2f}  {Tb_v - 273.15:8.2f}  '
          f'{phi_v.get("C2H5COOH", 1):8.4f}')

# --- Azeotrope ---
print()
if azo_uni:
    print(f'UNIQUAC-only: azeotrope at x={azo_uni["x"]:.4f}, T={azo_uni["T"]:.2f} °C')
else:
    x = {'C2H5COOH': 0.001, 'H2O': 0.999}
    Tb, y = bubble_T_vdm(vdm, x, P_atm)
    d1 = 0.001 - y['C2H5COOH']
    x = {'C2H5COOH': 0.999, 'H2O': 0.001}
    Tb, y = bubble_T_vdm(vdm, x, P_atm)
    d2 = 0.999 - y['C2H5COOH']
    sign_change = d1 * d2 < 0
    print(f'UNIQUAC-only: {"AZEOTROPE" if sign_change else "NO azeotrope"}')

x_fine = np.linspace(0.001, 0.999, 200)
prev_diff = None; azeo_vdm = None
for x1 in x_fine:
    x = {'C2H5COOH': float(x1), 'H2O': float(1 - x1)}
    _, y = bubble_T_vdm(vdm, x, P_atm)
    diff = x1 - y['C2H5COOH']
    if prev_diff is not None and prev_diff * diff < 0:
        azeo_vdm = x1; break
    prev_diff = diff

if azeo_vdm:
    print(f'UNIQUAC+VDM: azeotrope at x≈{azeo_vdm:.4f}')
else:
    print(f'UNIQUAC+VDM: NO azeotrope')

Tb_h2o = thermo.bubble_point_T({'C2H5COOH': 0.001, 'H2O': 0.999}, P_atm) - 273.15
Tb_pa  = thermo.bubble_point_T({'C2H5COOH': 0.999, 'H2O': 0.001}, P_atm) - 273.15
print(f'Pure H2O: {Tb_h2o:.1f} °C')
print(f'Pure propionic acid (UNIQUAC): {Tb_pa:.1f} °C  (experimental: 141 °C)')
