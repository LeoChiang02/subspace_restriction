import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

# 1. Load dataset and extract unique molecules in order (including H2)
df = pd.read_csv('subspace_results.csv')
molecules = [m for m in df['Molecule'].unique()]

# 2. Extract reference energy and spin expectation values
ref_df = df[['Molecule', 'Reference', 'Ref_SCF_Energy', 'Spin_Sq_SCF']].drop_duplicates()

piv_e = ref_df.pivot(index='Molecule', columns='Reference', values='Ref_SCF_Energy').reindex(molecules)
piv_s2 = ref_df.pivot(index='Molecule', columns='Reference', values='Spin_Sq_SCF').reindex(molecules)

# Compute signed differences (UHF - RHF)
delta_e = (piv_e['UHF'] - piv_e['RHF']).values
delta_s2 = (piv_s2['UHF'] - piv_s2['RHF']).values
x = np.arange(len(molecules))

# 3. Setup 2 stacked subplots sharing x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), dpi=300, sharex=True)

# Color differentiation (gray for closed-shell ~0, blue/orange for active differences)
colors_e = ['#b0bec5' if abs(v) < 1e-6 else '#1f77b4' for v in delta_e]
colors_s2 = ['#b0bec5' if abs(v) < 1e-6 else '#d95f02' for v in delta_s2]

# --- Subplot 1: Energy difference ---
ax1.bar(x, delta_e, color=colors_e, edgecolor='black', linewidth=0.8, width=0.55, zorder=3)
ax1.axhline(0, color='black', linestyle='-', linewidth=0.9, zorder=2)
ax1.set_ylabel(r'$\Delta E_{\mathrm{ref}} = E_{\mathrm{UHF}} - E_{\mathrm{RHF}}$ (Hartree)', fontsize=11)
ax1.set_title(r'(a) Reference SCF Energy Difference ($\Delta E_{\mathrm{ref}}$)', fontsize=12, pad=10)
ax1.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)
ax1.set_ylim(min(delta_e) * 1.25, 0.003)

for i, v in enumerate(delta_e):
    if abs(v) > 1e-6:
        ax1.text(i, v - 0.0008, f'{v:.4f}', ha='center', va='top', fontsize=8.5, fontweight='bold', color='#08306b')

# --- Subplot 2: Spin contamination difference ---
ax2.bar(x, delta_s2, color=colors_s2, edgecolor='black', linewidth=0.8, width=0.55, zorder=3)
ax2.axhline(0, color='black', linestyle='-', linewidth=0.9, zorder=2)
ax2.set_ylabel(r'$\Delta \langle S^2 \rangle_{\mathrm{ref}} = \langle S^2 \rangle_{\mathrm{UHF}} - \langle S^2 \rangle_{\mathrm{RHF}}$', fontsize=11)
ax2.set_title(r'(b) Reference Spin Contamination Difference ($\Delta \langle S^2 \rangle_{\mathrm{ref}}$)', fontsize=12, pad=10)
ax2.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6, zorder=0)
ax2.set_ylim(-0.03, max(delta_s2) * 1.18)

for i, v in enumerate(delta_s2):
    if abs(v) > 1e-6:
        ax2.text(i, v + 0.015, f'{v:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#7f2704')

# X-axis ticks and labels
ax2.set_xticks(x)
ax2.set_xticklabels(molecules, rotation=45, ha='right', fontsize=11)
ax2.set_xlim(-0.6, len(molecules) - 0.4)
ax2.set_xlabel('Molecule', fontsize=12, labelpad=8)

# Legends
leg1 = [
    Patch(facecolor='#1f77b4', edgecolor='k', label=r'Spin-polarized ($\Delta E < 0$)'),
    Patch(facecolor='#b0bec5', edgecolor='k', label=r'Closed-shell singlet ($\Delta E \approx 0$)')
]
ax1.legend(handles=leg1, loc='lower left', frameon=True, fontsize=9.5)

leg2 = [
    Patch(facecolor='#d95f02', edgecolor='k', label=r'Spin-contaminated ($\Delta \langle S^2 \rangle > 0$)'),
    Patch(facecolor='#b0bec5', edgecolor='k', label=r'Pure spin state ($\Delta \langle S^2 \rangle \approx 0$)')
]
ax2.legend(handles=leg2, loc='upper left', frameon=True, fontsize=9.5)

fig.suptitle(r'UHF vs. RHF Reference Comparison: Energy & Spin Contamination ($\mathrm{UHF} - \mathrm{RHF}$)', fontsize=13, y=0.99)
plt.tight_layout()
plt.savefig('UHFvsRHF.png', bbox_inches='tight')
plt.show()
