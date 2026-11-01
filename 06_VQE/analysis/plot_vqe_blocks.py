import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ----------------------------------------------------------------------
# 1. Molecular Constants & Subspace Floor Setup
# ----------------------------------------------------------------------
fci_tot = {
    "HF": -98.5965865806049,
    "LiH": -7.8824034103355,
    "CH": -37.8113493455759,
    "OH": -74.3871341272098,
}

nuc_repulsion = {
    "HF": 5.194802463219896,
    "OH": 4.365698347282665,
    "LiH": 0.995380044366418,
    "CH": 2.83513105234396,
}

sector_e0 = {
    ("HF", 0): -103.78651942347501,
    ("HF", 2): -103.35696033951213,
    ("LiH", 0): -8.873388605959928,
    ("LiH", 2): -8.761504365993739,
    ("LiH", 4): -6.698872338461917,
    ("OH", 1): -78.7449247887357,
    ("OH", 3): -78.35908641062225,
    ("CH", 1): -40.62895820401409,
    ("CH", 3): -40.63675349533744,
    ("CH", 5): -40.003312975179895,
}

# Electronic FCI: E_elec(FCI) = E_tot(FCI) - E_nuc
fci_elec = {mol: fci_tot[mol] - nuc_repulsion[mol] for mol in fci_tot}

# ----------------------------------------------------------------------
# 2. Data Filtering & Preprocessing
# ----------------------------------------------------------------------
df_raw = pd.read_csv("../mhs_hea/SRS_VQE_5cases.csv")
df = df_raw[df_raw["molecule"] != "molecule"].copy()

# Drop runs where converged column indicates failure
df = df[df["converged"].astype(str).str.upper() == "TRUE"].copy()

df["final_energy"] = pd.to_numeric(df["final_energy"])
df["layers"] = pd.to_numeric(df["layers"])
df["mol"] = df["molecule"].apply(lambda x: x.split("_")[0])
df["omega"] = df["molecule"].apply(lambda x: int(x.split("_")[1][1:]))
df["optimizer_label"] = df["optimizer"].replace(
    {"lbfgs": "L-BFGS", "adam": "Adam"}
)


# ----------------------------------------------------------------------
# 3. Figure Generator with Enlarged Label Offset
# ----------------------------------------------------------------------
def generate_log_figure(
    df,
    layout,
    sector_e0,
    fci_elec,
    output_filename,
    sharey="row",
    y_offset_pts=-9,
):
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

    n_rows = len(layout)
    n_cols = max(len(r) for r in layout)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 3.8 * n_rows),
        sharex=True,
        sharey=sharey,
        squeeze=False,
    )

    palette = {"L-BFGS": "#1f77b4", "Adam": "#d62728"}

    for r_idx, row in enumerate(layout):
        mol_name = row[0][0]
        min_e0 = min(sector_e0[(mol_name, om)] for _, om in row)

        for c_idx in range(n_cols):
            ax = axes[r_idx, c_idx]
            if c_idx >= len(row):
                ax.set_visible(False)
                continue

            mol, w = row[c_idx]
            sub_df = df[(df["mol"] == mol) & (df["omega"] == w)].copy()
            fci_ref = fci_elec[mol]
            sub_df["error_mHa"] = (sub_df["final_energy"] - fci_ref) * 1000.0

            # Boxplot
            sns.boxplot(
                data=sub_df,
                x="layers",
                y="error_mHa",
                hue="optimizer_label",
                ax=ax,
                palette=palette,
                fliersize=0,
                width=0.55,
                boxprops=dict(alpha=0.35, edgecolor="black", linewidth=1.0),
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(color="#333333", linewidth=1.0),
                capprops=dict(color="#333333", linewidth=1.0),
            )

            # Strip plot
            sns.stripplot(
                data=sub_df,
                x="layers",
                y="error_mHa",
                hue="optimizer_label",
                ax=ax,
                palette=palette,
                dodge=True,
                jitter=0.2,
                size=3.5,
                alpha=0.75,
            )

            # Subspace floor dashed line
            floor_mHa = (sector_e0[(mol, w)] - fci_ref) * 1000.0
            ax.axhline(
                floor_mHa,
                color="#2ca02c",
                linestyle="--",
                linewidth=1.3,
                zorder=2,
            )

            # Subspace floor label placed with an enlarged point-offset below the line
            ax.annotate(
                f"{floor_mHa:.2f} mHa",
                xy=(0.96, floor_mHa),
                xycoords=ax.get_yaxis_transform(),
                xytext=(0, y_offset_pts),
                textcoords="offset points",
                color="#2ca02c",
                va="top",
                ha="right",
                fontsize=8.5,
                fontweight="bold",
            )

            # Chemical accuracy (1.6 mHa)
            ax.axhline(
                1.6, color="#7f7f7f", linestyle=":", linewidth=1.2, zorder=1
            )

            # Sector Title: MHS(w) [ground block]
            is_ground = abs(sector_e0[(mol, w)] - min_e0) < 1e-6
            ground_tag = " (ground block)" if is_ground else ""
            ax.set_title(
                f"{mol}, MHS({w}){ground_tag}",
                loc="left",
                fontsize=11,
                fontweight="bold",
                pad=8,
            )

            ax.set_yscale("log")
            ax.yaxis.grid(True, linestyle="--", alpha=0.6)
            ax.xaxis.grid(False)

            ax.set_xlabel(
                "ansatz layers" if r_idx == n_rows - 1 else "", fontsize=10
            )
            if c_idx == 0 or not sharey:
                ax.set_ylabel(
                    r"$E - E(\mathrm{FCI})$ [mHa], log scale", fontsize=10
                )
            else:
                ax.set_ylabel("")

            if ax.get_legend():
                ax.get_legend().remove()

    # Legend positioned below the plot
    handles = [
        plt.Line2D([0], [0], color="#1f77b4", lw=2.5, label="L-BFGS"),
        plt.Line2D([0], [0], color="#d62728", lw=2.5, label="Adam"),
        plt.Line2D(
            [0], [0], color="#2ca02c", lw=1.5, ls="--", label="subspace floor"
        ),
        plt.Line2D([0], [0], color="#7f7f7f", lw=1.5, ls=":", label="1.6 mHa"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
        fontsize=10.5,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------
# 4. Generate Figures
# ----------------------------------------------------------------------
# Fig 1 (2x2): row HF (w0, w2), row OH (w1, w3)
fig1_layout = [[("HF", 0), ("HF", 2)], [("OH", 1), ("OH", 3)]]

# Fig 2 (2x3): row LiH (w0, w2, w4), row CH (w1, w3, w5)
fig2_layout = [
    [("LiH", 0), ("LiH", 2), ("LiH", 4)],
    [("CH", 1), ("CH", 3), ("CH", 5)],
]

# y_offset_pts=-9 produces a distinct, readable gap below the dashed green line
generate_log_figure(
    df,
    fig1_layout,
    sector_e0,
    fci_elec,
    "HF_OH_VQE.png",
    sharey="row",
    y_offset_pts=-9,
)
generate_log_figure(
    df,
    fig2_layout,
    sector_e0,
    fci_elec,
    "LiH_CH_VQE.png",
    sharey="row",
    y_offset_pts=-9,
)