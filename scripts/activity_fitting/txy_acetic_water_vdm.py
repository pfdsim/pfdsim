"""T-xy for acetic acid-water: UNIQUAC + VDM vapor correction vs UNIQUAC-only.

Corrected formulation:
    K_i = gamma_i * phi_i^sat * P_i^sat(T) / (phi_i^V * P)

where phi_i^sat is the VDM fugacity coefficient of pure i at its saturation
pressure, and phi_i^V is the VDM fugacity coefficient in the mixture vapour.
The ratio phi_i^sat / phi_i^V captures how dimerization in the mixture differs
from dimerization in the pure-component vapour — this is what drives the
VLE correction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from thermodynamics import UNIQUACThermodynamics
from vapor_dimerization import VaporDimerizationModel

P_atm = 1.01325  # bar

thermo = UNIQUACThermodynamics(['CH3COOH', 'H2O'])

vdm = VaporDimerizationModel(
    monomer='CH3COOH',
    dimer='(CH3COOH)2',
    K_ref=1415.0,
    T_ref=298.15,
    delta_H=-65500.0,
)


def phi_sat(comp: str, T: float) -> float:
    """VDM fugacity coefficient of pure *comp* at its saturation pressure.

    For a non-associating component this is ~1.  For the monomer it
    captures dimerization in the pure-component vapour at P_sat(T).
    """
    if comp != 'CH3COOH':
        return 1.0
    Psat = thermo.Psat(comp, T)
    y_pure = {'CH3COOH': 1.0, 'H2O': 0.0}
    phi = vdm.fugacity_coefficients(T, Psat, y_pure, rk_model=None)
    return phi.get(comp, 1.0)


def bubble_T_vdm(x: dict, P: float, T_guess: float = 370.0,
                 max_outer: int = 60, tol: float = 1e-7):
    """Bubble-point T with VDM-corrected vapour fugacity.

        K_i = gamma_i * phi_i^sat * P_i^sat / (phi_i^V * P)
    """
    T = T_guess
    comps = list(x.keys())

    for _ in range(max_outer):
        gamma = thermo.activity_coefficients(T, x)
        Psat = {c: thermo.Psat(c, T) for c in comps}
        phi_s = {c: phi_sat(c, T) for c in comps}

        # initial K without vapour correction (phi^V = 1)
        K = {c: gamma[c] * phi_s[c] * Psat[c] / P for c in comps}
        y = {c: x[c] * K[c] for c in comps}
        s = sum(y.values())
        y = {c: yi / s for c, yi in y.items()}

        # inner fixed-point:  K ↔ y ↔ phi^V
        for _ in range(15):
            phi_v = vdm.fugacity_coefficients(T, P, y, rk_model=None)
            K = {c: gamma[c] * phi_s[c] * Psat[c] / max(phi_v.get(c, 1.0) * P, 1e-12)
                 for c in comps}
            y_new = {c: x[c] * K[c] for c in comps}
            s = sum(y_new.values())
            y_new = {c: yi / s for c, yi in y_new.items()}

            if max(abs(y_new[c] - y[c]) for c in comps) < 1e-10:
                y = y_new
                break
            y = y_new

        sum_xK = sum(x[c] * K[c] for c in comps)
        if abs(sum_xK - 1.0) < tol:
            return T, y

        dT = -5.0 * (sum_xK - 1.0) if sum_xK > 1 else 5.0 * (1.0 - sum_xK)
        T += dT
        T = max(200.0, min(700.0, T))

    return T, y


def dew_T_vdm(y_target: dict, P: float, T_guess: float = 380.0,
              max_iter: int = 100, tol: float = 1e-7):
    """Dew-point T with VDM correction.

    At the dew point:  sum(y_i / K_i) = 1
    where liquid composition x is unknown and K depends on both x and y.
    We iterate:  x_i = y_i / K_i  (normalised), then recompute K.
    """
    T = T_guess
    comps = list(y_target.keys())

    for _ in range(max_iter):
        gamma = thermo.activity_coefficients(T, y_target)  # initial guess x ≈ y
        Psat = {c: thermo.Psat(c, T) for c in comps}
        phi_s = {c: phi_sat(c, T) for c in comps}
        phi_v = vdm.fugacity_coefficients(T, P, y_target, rk_model=None)

        K = {c: gamma[c] * phi_s[c] * Psat[c] / max(phi_v.get(c, 1.0) * P, 1e-12)
             for c in comps}
        x = {c: y_target[c] / max(K[c], 1e-12) for c in comps}
        s = sum(x.values())
        x = {c: xi / s for c, xi in x.items()}

        # iterate x ↔ gamma
        for _ in range(15):
            gamma = thermo.activity_coefficients(T, x)
            K = {c: gamma[c] * phi_s[c] * Psat[c] / max(phi_v.get(c, 1.0) * P, 1e-12)
                 for c in comps}
            x_new = {c: y_target[c] / max(K[c], 1e-12) for c in comps}
            s = sum(x_new.values())
            x_new = {c: xi / s for c, xi in x_new.items()}

            if max(abs(x_new[c] - x[c]) for c in comps) < 1e-10:
                x = x_new
                break
            x = x_new

        sum_yKinv = sum(y_target[c] / max(K[c], 1e-12) for c in comps)

        if abs(sum_yKinv - 1.0) < tol:
            return T

        if sum_yKinv > 1:
            T += 3.0 * (sum_yKinv - 1.0)
        else:
            T -= 3.0 * (1.0 - sum_yKinv)
        T = max(200.0, min(700.0, T))

    return T


# --- Compute ---
n = 80
x_vals = np.linspace(0.001, 0.999, n)
data_uni = thermo.generate_Txy_data('CH3COOH', 'H2O', P_atm, n_points=n)

T_bub_vdm, T_dew_vdm = [], []
y_bub_vdm = []

for x1 in x_vals:
    x = {'CH3COOH': float(x1), 'H2O': float(1 - x1)}

    T_bub, y = bubble_T_vdm(x, P_atm)
    T_bub_vdm.append(T_bub - 273.15)
    y_bub_vdm.append(y['CH3COOH'])

    y_comp = {'CH3COOH': y['CH3COOH'], 'H2O': 1.0 - y['CH3COOH']}
    T_dew = dew_T_vdm(y_comp, P_atm, T_guess=T_bub + 5)
    T_dew_vdm.append(T_dew - 273.15)

# --- Print ---
print(f"{'x(AcOH)':>8s}  {'y(Uni)':>8s}  {'y(VDM)':>8s}  "
      f"{'Tb(Uni)':>8s}  {'Tb(VDM)':>8s}  {'Td(VDM)':>8s}  "
      f"{'phiV':>8s}  {'phiSat':>8s}")
print("-" * 80)
for i in range(0, n, 8):
    x1 = x_vals[i]
    y_uni = data_uni['y'][i]
    y_vdm = y_bub_vdm[i]
    T_uni = data_uni['T_bubble'][i]
    Tb = T_bub_vdm[i]
    Td = T_dew_vdm[i]

    y_comp = {'CH3COOH': float(y_vdm), 'H2O': float(1 - y_vdm)}
    Tk = Tb + 273.15
    phi_v = vdm.fugacity_coefficients(Tk, P_atm, y_comp, rk_model=None)
    phiV = phi_v.get('CH3COOH', 1.0)
    phiS = phi_sat('CH3COOH', Tk)

    print(f"{x1:8.4f}  {y_uni:8.4f}  {y_vdm:8.4f}  "
          f"{T_uni:8.2f}  {Tb:8.2f}  {Td:8.2f}  "
          f"{phiV:8.4f}  {phiS:8.4f}")

for idx in (0, n - 1):
    x1 = x_vals[idx]
    y_uni = data_uni['y'][idx]
    y_vdm = y_bub_vdm[idx]
    T_uni = data_uni['T_bubble'][idx]
    Tb = T_bub_vdm[idx]
    Td = T_dew_vdm[idx]
    Tk = Tb + 273.15
    y_comp = {'CH3COOH': float(y_vdm), 'H2O': float(1 - y_vdm)}
    phi_v = vdm.fugacity_coefficients(Tk, P_atm, y_comp, rk_model=None)
    phiV = phi_v.get('CH3COOH', 1.0)
    phiS = phi_sat('CH3COOH', Tk)
    print(f"{x1:8.4f}  {y_uni:8.4f}  {y_vdm:8.4f}  "
          f"{T_uni:8.2f}  {Tb:8.2f}  {Td:8.2f}  "
          f"{phiV:8.4f}  {phiS:8.4f}")

# --- Azeotrope check ---
azeo_vdm = None
for i in range(1, len(x_vals)):
    prev_diff = x_vals[i - 1] - y_bub_vdm[i - 1]
    curr_diff = x_vals[i] - y_bub_vdm[i]
    if prev_diff * curr_diff < 0:
        frac = abs(prev_diff) / (abs(prev_diff) + abs(curr_diff))
        x_az = x_vals[i - 1] + frac * (x_vals[i] - x_vals[i - 1])
        T_az = T_bub_vdm[i - 1] + frac * (T_bub_vdm[i] - T_bub_vdm[i - 1])
        azeo_vdm = (x_az, T_az)
        break

print()
if azeo_vdm:
    print(f"VDM azeotrope: x = {azeo_vdm[0]:.4f}, T = {azeo_vdm[1]:.2f} C")
else:
    print("VDM: NO azeotrope detected")
    print(f"  (x-y) at x≈0.001: {x_vals[0] - y_bub_vdm[0]:+.6f}")
    print(f"  (x-y) at x≈0.999: {x_vals[-1] - y_bub_vdm[-1]:+.6f}")

azo_uni = data_uni.get('azeotrope')
if azo_uni:
    print(f"UNIQUAC-only azeotrope: x = {azo_uni['x']:.4f}, T = {azo_uni['T']:.2f} C")

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: VDM-corrected T-xy
ax1.plot(x_vals, T_bub_vdm, 'b-', lw=1.5, label='Bubble (VDM)')
ax1.plot(y_bub_vdm, T_dew_vdm, 'r--', lw=1.5, label='Dew (VDM)')
ax1.set_xlabel('Mole fraction acetic acid')
ax1.set_ylabel('Temperature [°C]')
ax1.set_title('UNIQUAC + VDM')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.02, 1.02)

# annotate pure-component boiling points
ax1.annotate(f'{T_bub_vdm[0]:.1f} °C', xy=(0, T_bub_vdm[0]),
             xytext=(0.05, T_bub_vdm[0] + 5), fontsize=9,
             arrowprops=dict(arrowstyle='->', lw=0.8))
ax1.annotate(f'{T_bub_vdm[-1]:.1f} °C', xy=(1, T_bub_vdm[-1]),
             xytext=(0.80, T_bub_vdm[-1] + 5), fontsize=9,
             arrowprops=dict(arrowstyle='->', lw=0.8))

if azeo_vdm:
    ax1.plot(azeo_vdm[0], azeo_vdm[1], 'ko', markersize=5)

# Right: y-x comparison  (trim to matching lengths)
n_plot = min(len(x_vals), len(data_uni['x']))
ax2.plot([0, 1], [0, 1], 'k-', lw=0.5, label='y = x')
ax2.plot(data_uni['x'][:n_plot], data_uni['y'][:n_plot],
         'orange', lw=1.5, alpha=0.7, label='UNIQUAC only')
ax2.plot(x_vals[:n_plot], y_bub_vdm[:n_plot],
         'b-', lw=1.5, label='UNIQUAC + VDM')
ax2.set_xlabel('x (liquid mole fraction AcOH)')
ax2.set_ylabel('y (vapour mole fraction AcOH)')
ax2.set_title('Equilibrium curve')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)

if azo_uni:
    ax2.plot(azo_uni['x'], azo_uni['x'], 'o', color='gray', markersize=5)
    ax2.annotate('UNIQUAC\noazeotrope', xy=(azo_uni['x'], azo_uni['x']),
                 xytext=(azo_uni['x'] + 0.15, azo_uni['x'] - 0.08),
                 fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8))

plt.tight_layout()
out = Path(__file__).resolve().parents[2] / 'outputs' / 'txy_acetic_water_vdm.png'
plt.savefig(out, dpi=150)
print(f"\nPlot saved to {out}")
