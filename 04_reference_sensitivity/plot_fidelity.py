import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# Load dataset and extract MHS blocks (all molecules, including H2)
df = pd.read_csv('subspace_results.csv')
mhs = df[df['Subspace'].str.startswith('MHS')].copy()

# Preserve molecular ordering
molecules = [m for m in df['Molecule'].unique()]
mol_to_x = {m: i for i, m in enumerate(molecules)}

# Parity palettes
even_colors = {
    0: '#084594',  # Deep Navy
    2: '#2171b5',  # Royal Blue
    4: '#4292c6',  # Medium Blue
    6: '#6baed6',  # Soft Blue
    8: '#9ecae1'   # Pale Blue
}

odd_colors = {
    1: '#7f0000',  # Deep Maroon
    3: '#b30000',  # Crimson
    5: '#e34a33',  # Vermilion
    7: '#fdbb84'   # Light Orange
}

omega_colors = {**even_colors, **odd_colors}
offsets = {'RHF': -0.12, 'UHF': 0.12}
marker_map = {'RHF': 'o', 'UHF': 'X'}

# Setup figure
fig, ax = plt.subplots(figsize=(15, 6.5), dpi=300)

for mol in molecules:
    x_center = mol_to_x[mol]
    ax.axvline(x_center, color='gray', linestyle=':', linewidth=0.5, alpha=0.3, zorder=0)

    for ref in ['RHF', 'UHF']:
        sub = mhs[(mhs['Molecule'] == mol) & (mhs['Reference'] == ref)].sort_values('Omega')
        if sub.empty:
            continue

        x_pos = x_center + offsets[ref]
        y_vals = sub['Fidelity_to_PCS'].values

        # Vertical ladder connection
        ax.plot([x_pos] * len(y_vals), y_vals, color='gray', linestyle='-', linewidth=0.8, alpha=0.5, zorder=1)

        # Seniority markers
        for _, row in sub.iterrows():
            w = int(row['Omega'])
            ax.scatter(
                x_pos, row['Fidelity_to_PCS'],
                color=omega_colors[w],
                marker=marker_map[ref], s=80, edgecolor='black', linewidth=0.7, zorder=2
            )

# Axis formatting
ax.set_ylim(-0.05, 1.05)
ax.set_yticks(np.linspace(0, 1, 6))
ax.set_xticks(range(len(molecules)))
ax.set_xticklabels(molecules, rotation=45, ha='right', fontsize=11)
ax.set_xlim(-0.6, len(molecules) - 0.4)
ax.set_xlabel('Molecule', fontsize=12, labelpad=8)
ax.set_ylabel(r'Subspace Fidelity to PCS', fontsize=11)
ax.set_title(r'Seniority-Resolved Subspace Fidelity across Molecules', fontsize=13, pad=12)
ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)

# Construct unified parallel 3-column legend handles
empty = Line2D([], [], color='none')

col1_handles = [
    Line2D([], [], color='none'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=even_colors[0], markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=even_colors[2], markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=even_colors[4], markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=even_colors[6], markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=even_colors[8], markeredgecolor='k', markersize=8),
]
col1_labels = [r'$\bf{Seniority\ (\omega\ even)}$', r'$\omega = 0$', r'$\omega = 2$', r'$\omega = 4$', r'$\omega = 6$', r'$\omega = 8$']

col2_handles = [
    Line2D([], [], color='none'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=odd_colors[1], markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=odd_colors[3], markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=odd_colors[5], markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=odd_colors[7], markeredgecolor='k', markersize=8),
    empty,
]
col2_labels = [r'$\bf{Seniority\ (\omega\ odd)}$', r'$\omega = 1$', r'$\omega = 3$', r'$\omega = 5$', r'$\omega = 7$', '']

col3_handles = [
    Line2D([], [], color='none'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8),
    Line2D([0], [0], marker='X', color='w', markerfacecolor='gray', markeredgecolor='k', markersize=8),
    empty, empty, empty
]
col3_labels = [r'$\bf{Reference}$', 'RHF', 'UHF', '', '', '']

ordered_handles = col1_handles + col2_handles + col3_handles
ordered_labels = col1_labels + col2_labels + col3_labels

# Lowered legend position: stays clear of y=0 baseline and y>=0.92 top clusters
ax.legend(
    ordered_handles, ordered_labels,
    loc='lower right', bbox_to_anchor=(0.99, 0.10),
    ncols=3, fontsize=9.5, columnspacing=2.2, handletextpad=0.6,
    frameon=True, borderaxespad=0.8
)

plt.tight_layout()
plt.savefig('subspace_fidelity_linear.png', bbox_inches='tight')
plt.show()
