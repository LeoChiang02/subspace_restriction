"""Figure 12: UCCSD under Jordan-Wigner, error versus FCI per molecule (L-BFGS and Adam, 15 seeds).

Run from this folder:  python plot_uccsd.py   ->  UCCSD_box_scatter.png
Input: ../uccsd/SRS_VQE_UCCSD.csv (all trials). Errors are total energies minus the total FCI energy.
Title matched to published figure.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Total FCI energies (STO-3G). HF, LiH, CH, OH: the values used in plot_vqe_blocks.py; CH is the quartet.
# H2: FCI_Energy of H2 in ../../04_reference_sensitivity/subspace_results.csv.
fci_tot = {
    "H2": -1.137270174660903,
    "HF": -98.5965865806049,
    "LiH": -7.8824034103355,
    "CH": -37.8113493455759,
    "OH": -74.3871341272098,
}

df_uccsd = pd.read_csv("../uccsd/SRS_VQE_UCCSD.csv")


def generate_uccsd_plot(df, fci_dict, output_filename, mol_order=['H2', 'LiH', 'HF', 'OH', 'CH']):
    sns.set_theme(
        style="ticks",
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "grid.color": "#e8e8e8",
            "grid.linestyle": "--",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "font.family": "sans-serif",
        },
    )

    df_plot = df.copy()
    df_plot = df_plot[df_plot['converged'].astype(str).str.upper() == 'TRUE'].copy()
    df_plot['fci'] = df_plot['molecule'].map(fci_dict)
    df_plot['error_mHa'] = (df_plot['final_energy'] - df_plot['fci']) * 1000.0
    df_plot['optimizer_label'] = df_plot['optimizer'].replace({'lbfgs': 'L-BFGS', 'adam': 'Adam'})

    palette = {"L-BFGS": "#1f77b4", "Adam": "#d62728"}

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # Boxplot
    sns.boxplot(
        data=df_plot,
        x="molecule",
        y="error_mHa",
        hue="optimizer_label",
        order=mol_order,
        ax=ax,
        palette=palette,
        fliersize=0,
        width=0.5,
        boxprops=dict(alpha=0.35, edgecolor="black", linewidth=1.0),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="#333333", linewidth=1.0),
        capprops=dict(color="#333333", linewidth=1.0),
    )

    # Strip plot
    sns.stripplot(
        data=df_plot,
        x="molecule",
        y="error_mHa",
        hue="optimizer_label",
        order=mol_order,
        ax=ax,
        palette=palette,
        dodge=True,
        jitter=0.2,
        size=4.5,
        alpha=0.75,
    )

    # Chemical accuracy threshold (1.6 mHa)
    ax.axhline(1.6, color="#7f7f7f", linestyle=":", linewidth=1.3, zorder=1, label="1.6 mHa")
    ax.text(
        0.98,
        1.6,
        "1.6 mHa",
        transform=ax.get_yaxis_transform(),
        color="#7f7f7f",
        va="bottom",
        ha="right",
        fontsize=9,
        fontweight="bold",
    )

    ax.set_yscale("log")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.xaxis.grid(False)

    ax.set_xlabel("molecule", fontsize=11, labelpad=8)
    ax.set_ylabel(r"$E - E(\mathrm{FCI})$ [mHa], log scale", fontsize=11)
    ax.set_title("UCCSD VQE Energy Error", loc="left", fontsize=12, fontweight="bold", pad=12)

    # Clean legend
    handles = [
        plt.Line2D([0], [0], color="#1f77b4", lw=2.5, label="L-BFGS"),
        plt.Line2D([0], [0], color="#d62728", lw=2.5, label="Adam"),
        plt.Line2D([0], [0], color="#7f7f7f", lw=1.5, ls=":", label="1.6 mHa"),
    ]
    ax.get_legend().remove()
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        frameon=False,
        fontsize=10.5,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved", output_filename)


generate_uccsd_plot(df_uccsd, fci_tot, "UCCSD_box_scatter.png")
