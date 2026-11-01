import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import numpy as np

def plot_pes_from_csv(csv_file, molecule='LiH', save_path=None, num_sectors=None, energy_ylim=None):
    df = pd.read_csv(csv_file)
    if 'Molecule' in df.columns:
        df = df[df['Molecule'] == molecule]

    # Clean machine epsilon noise
    if 'S2' in df.columns:
        df['S2'] = df['S2'].apply(lambda x: 0.0 if abs(x) < 1e-10 else x)
    if 'S2_SCF' in df.columns:
        df['S2_SCF'] = df['S2_SCF'].apply(lambda x: 0.0 if abs(x) < 1e-10 else x)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True, sharey='row')

    m_every = 2
    hollow_kwargs = {'markerfacecolor': 'none', 'markeredgewidth': 1.2, 'markevery': m_every}

    fci_style = {'color': 'black', 'ls': '-', 'lw': 1.8, 'label': 'FCI'}
    ref_style = {'color': 'purple', 'ls': '--', 'marker': 'd', 'ms': 5, 'lw': 1.2, **hollow_kwargs}
    hs_style  = {'color': '#2a9d8f', 'ls': '-.', 'marker': 'x', 'ms': 5, 'lw': 1.2, 'label': 'HS', 'markevery': m_every}

    def get_omega(s):
        try:
            return int(s.split('w=')[1].rstrip(')'))
        except:
            return 999

    available_sectors = sorted([s for s in df['Subspace'].unique() if s.startswith('MHS')], key=get_omega)
    if num_sectors is not None:
        mhs_sectors = available_sectors[:num_sectors]
    else:
        mhs_sectors = available_sectors

    # Generate a smooth sequential color gradient (from light/yellow-green to dark teal/purple or viridis)
    # Using matplotlib colormaps (e.g., 'YlGnBu' or 'viridis' or 'plasma')
    n_sec = len(mhs_sectors)
    # We want a colormap ranging from ~0.2 to ~0.85 so colors aren't too pale or too dark
    cmap = cm.get_cmap('viridis', max(n_sec + 1, 3))
    colors = [cmap(v) for v in np.linspace(0.2, 0.85, n_sec)]
    marker_palette = ['o', 'v', '^', 's', '<', 'p', '*']

    mhs_styles = {}
    for idx, s in enumerate(mhs_sectors):
        mhs_styles[s] = {
            'color': colors[idx],
            'ls': '--' if idx == 0 else ':',
            'marker': marker_palette[idx % len(marker_palette)],
            'ms': 6,
            'lw': 1.2,
            'label': s.replace('MHS(w=', r'MHS ($\omega=').replace(')', r'$)'),
            **hollow_kwargs
        }

    spin_step = 2.0 if molecule == 'N2' else 0.2

    for col_idx, ref in enumerate(['RHF', 'UHF']):
        ref_df = df[df['Reference'] == ref]
        if ref_df.empty:
            continue

        base = ref_df[['Distance', 'E_FCI', 'E_SCF', 'S2_SCF']].drop_duplicates().sort_values('Distance')

        # Row 0: Energy
        axes[0, col_idx].plot(base['Distance'], base['E_FCI'], **fci_style)
        axes[0, col_idx].plot(base['Distance'], base['E_SCF'], **{**ref_style, 'label': f'{ref} (Ref)'})

        # HS
        hs_data = ref_df[ref_df['Subspace'] == 'HS'].sort_values('Distance')
        if not hs_data.empty:
            axes[0, col_idx].plot(hs_data['Distance'], hs_data['Energy'], **hs_style)

        # MHS sectors
        for s in mhs_sectors:
            s_data = ref_df[ref_df['Subspace'] == s].sort_values('Distance')
            if not s_data.empty:
                st = mhs_styles[s]
                axes[0, col_idx].plot(s_data['Distance'], s_data['Energy'], **st)
                if 'S2' in s_data.columns and s_data['S2'].notna().any():
                    axes[1, col_idx].plot(s_data['Distance'], s_data['S2'], **st)
                if 'Fidelity' in s_data.columns and s_data['Fidelity'].notna().any():
                    axes[2, col_idx].plot(s_data['Distance'], s_data['Fidelity'], **st)

        # Row 1: Spin <S^2>
        axes[1, col_idx].plot(base['Distance'], base['S2_SCF'], **{**ref_style, 'label': f'{ref} $\\langle S^2\\rangle$'})
        axes[1, col_idx].axhline(0.0, color='gray', lw=0.8, alpha=0.6, ls='--')

        # Formatting
        axes[0, col_idx].set_title(f'{molecule} ({ref} Reference)', fontsize=13, fontweight='bold')
        axes[0, col_idx].legend(loc='best', frameon=True, fontsize=8.5)
        axes[0, col_idx].grid(True, alpha=0.3)

        axes[1, col_idx].legend(loc='best', frameon=True, fontsize=8.5)
        axes[1, col_idx].grid(True, alpha=0.3)

        axes[2, col_idx].set_xlabel(r'Distance ($\AA$)', fontsize=11)
        axes[2, col_idx].set_ylim(-0.05, 1.05)
        axes[2, col_idx].legend(loc='best', frameon=True, fontsize=8.5)
        axes[2, col_idx].grid(True, alpha=0.3)

    # Shared axis labels
    axes[0, 0].set_ylabel('Energy (Ha)', fontsize=11)
    axes[1, 0].set_ylabel(r'Spin $\langle S^2 \rangle$', fontsize=11)
    axes[2, 0].set_ylabel('Fidelity', fontsize=11)

    axes[1, 0].yaxis.set_major_locator(ticker.MultipleLocator(spin_step))
    axes[1, 1].yaxis.set_major_locator(ticker.MultipleLocator(spin_step))

    if energy_ylim is not None:
        axes[0, 0].set_ylim(energy_ylim)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()

if __name__ == '__main__':
    # Test LiH with upper bound -7.60
    plot_pes_from_csv('data/lih_scan.csv', molecule='LiH', energy_ylim=(-7.92, -7.60), save_path='lih_pes_gradient.png')
    # Test N2 with upper bound -106.6
    plot_pes_from_csv('data/n2_scan.csv', molecule='N2', energy_ylim=(-107.75, -106.6), save_path='n2_pes_gradient.png')
