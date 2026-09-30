"""Generate T-xy diagram for acetic acid-water using UNIQUAC at 1 atm."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from thermodynamics import UNIQUACThermodynamics

P_atm = 1.01325  # bar

thermo = UNIQUACThermodynamics(['CH3COOH', 'H2O'])
data = thermo.generate_Txy_data('CH3COOH', 'H2O', P_atm, n_points=80)

# Print representative values
print(f"{'x(AcOH)':>10s} {'y(AcOH)':>10s} {'T_bub(C)':>10s} {'T_dew(C)':>10s}")
print("-" * 44)
for i in range(0, len(data['x']), 8):
    print(f"{data['x'][i]:10.4f} {data['y'][i]:10.4f} "
          f"{data['T_bubble'][i]:10.2f} {data['T_dew'][i]:10.2f}")

# Always include endpoint rows
for idx in (0, len(data['x']) - 1):
    print(f"{data['x'][idx]:10.4f} {data['y'][idx]:10.4f} "
          f"{data['T_bubble'][idx]:10.2f} {data['T_dew'][idx]:10.2f}")

azeo = data.get('azeotrope')
if azeo:
    print(f"\nAzeotrope: x = {azeo['x']:.4f}, T = {azeo['T']:.2f} C "
          f"({azeo.get('type', '?')})")

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(data['x'], data['T_bubble'], 'b-', linewidth=1.5, label='Bubble point')
ax.plot(data['y'], data['T_dew'], 'r--', linewidth=1.5, label='Dew point')

Tb_h2o = data['T_bubble'][0]
Tb_acoh = data['T_bubble'][-1]

ax.set_xlabel('Mole fraction acetic acid', fontsize=12)
ax.set_ylabel('Temperature [°C]', fontsize=12)
ax.set_title('T-xy: Acetic Acid - Water at 1 atm (UNIQUAC)', fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim(-0.02, 1.02)
ax.grid(True, alpha=0.3)

ax.annotate(f'Water\n{Tb_h2o:.1f} °C', xy=(0, Tb_h2o), xytext=(0.08, Tb_h2o + 4),
            fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.8))
ax.annotate(f'AcOH\n{Tb_acoh:.1f} °C', xy=(1, Tb_acoh), xytext=(0.82, Tb_acoh + 4),
            fontsize=9, arrowprops=dict(arrowstyle='->', lw=0.8))

if azeo:
    ax.plot(azeo['x'], azeo['T'], 'ko', markersize=6)
    ax.annotate(f"Azeotrope\nx={azeo['x']:.3f}\n{azeo['T']:.1f} °C",
                xy=(azeo['x'], azeo['T']),
                xytext=(azeo['x'] + 0.12, azeo['T'] - 8),
                fontsize=9,
                arrowprops=dict(arrowstyle='->', lw=0.8))

plt.tight_layout()
out_path = Path(__file__).resolve().parents[2] / 'outputs' / 'txy_acetic_acid_water_uniquac.png'
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")
