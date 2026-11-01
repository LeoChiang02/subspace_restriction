"""VQE statistics quoted in the text and in Table 4.

A run is successful if it converged and |E_r - E_MHS(Omega)| <= 1.6e-3 Ha, with E_MHS(Omega) the exact block
ground energy (E0 of the block in data/records); non-converged runs count as failures.

1. Success rates (Sec. VQE): fraction of successful runs, pooled over the HF, LiH, OH and CH blocks
   (H2 excluded), per depth L and optimizer. Denominator: converged runs, and for Adam also all runs.
2. Depth L* of a block (Sec. 5.2, Table 4): the smallest L at which at least 8 of the 15 L-BFGS runs at that
   depth are successful ("the depth at which most L-BFGS runs reach 1.6 mHa").
3. Table 4: L-BFGS iterations, mean +- sample standard deviation over the 15 initializations.
   UCCSD: SRS_VQE_UCCSD.csv. MHS: lowest-energy block at L*, with CNOT = L*(q-1) for the linear-entangler HEA.
Run from this folder:  python vqe_stats.py   ->  success_rates.csv, depth_per_block.csv, table4_iterations.csv
"""
import json
import os
import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))
MHS_CSV = os.path.join(here, "..", "mhs_hea", "SRS_VQE_5cases.csv")
UCC_CSV = os.path.join(here, "..", "uccsd", "SRS_VQE_UCCSD.csv")
HAM_DIR = os.path.join(here, "..", "mhs_hea", "hamiltonians")
REC_DIR = os.path.join(here, "..", "..", "data", "records")
TABLE4_BLOCKS = {"HF": "HF_w0", "LiH": "LiH_w0", "OH": "OH_w1", "CH": "CH_w3"}   # lowest-energy blocks
CHEM_ACC = 1.6e-3


def block_energies():
    """E_MHS(Omega) (electronic) for every block, keyed like the 'molecule' column, e.g. 'CH_w3'."""
    E = {}
    for mol in ["H2", "HF", "LiH", "OH", "CH"]:
        for r in json.load(open(os.path.join(REC_DIR, f"{mol}_records.json"))):
            if r["shift"] == "zero":
                E[f"{mol}_w{r['omega']}"] = r["E0"]
    return E


def load(path):
    df = pd.read_csv(path).dropna(how="all")
    df = df[df["molecule"] != "molecule"].copy()            # drop the embedded repeated header
    df["layers"] = df["layers"].astype(int)
    df["iterations"] = df["iterations"].astype(int)
    df["converged"] = df["converged"].astype(str).str.upper() == "TRUE"
    return df


mhs = load(MHS_CSV)
ucc = load(UCC_CSV)
E_MHS = block_energies()
mhs["success"] = mhs["converged"] & ((mhs["final_energy"].astype(float) - mhs["molecule"].map(E_MHS)).abs() <= CHEM_ACC)

# ---- 1. success rates ---------------------------------------------------------
pool = mhs[mhs["molecule"].str.split("_").str[0].isin(["HF", "LiH", "OH", "CH"])]
rows = []
for opt in ["lbfgs", "adam"]:
    for L in sorted(pool["layers"].unique()):
        g = pool[(pool["optimizer"] == opt) & (pool["layers"] == L)]
        conv = g[g["converged"]]
        rows.append({"optimizer": opt, "L": L, "runs": len(g), "converged": len(conv), "success": int(g["success"].sum()),
                     "rate_converged_pct": round(100 * conv["success"].mean(), 1),
                     "rate_all_pct": round(100 * g["success"].mean(), 1)})
rates = pd.DataFrame(rows)
rates.to_csv(os.path.join(here, "success_rates.csv"), index=False)
print(rates.to_string(index=False))


# ---- 2. depth L* per block ----------------------------------------------------
def depth(g):
    """Smallest L with at least 8 of the 15 L-BFGS runs at that L successful (None if never)."""
    ok = g.groupby("layers")["success"].sum()
    return next((int(L) for L, n in ok.items() if n >= 8), None), ok


rows = []
for blk in sorted(mhs["molecule"].unique()):
    L_star, ok = depth(mhs[(mhs["molecule"] == blk) & (mhs["optimizer"] == "lbfgs")])
    rows.append({"block": blk, **{f"success_L{L}": int(n) for L, n in ok.items()}, "L_star": L_star})
depths = pd.DataFrame(rows)
depths.to_csv(os.path.join(here, "depth_per_block.csv"), index=False)
print(depths.to_string(index=False))


# ---- 3. Table 4 iterations ----------------------------------------------------
def mean_sd(x):
    return f"{np.mean(x):.1f} +- {np.std(x, ddof=1):.1f}"


t4 = []
for mol in TABLE4_BLOCKS:
    g = ucc[(ucc["molecule"] == mol) & (ucc["optimizer"] == "lbfgs")]
    t4.append({"Molecule": mol, "Encoding": "JW/UCCSD", "L": "", "q": "", "CNOT": "", "n": len(g), "Iter": mean_sd(g["iterations"])})
for mol, blk in TABLE4_BLOCKS.items():
    g = mhs[(mhs["molecule"] == blk) & (mhs["optimizer"] == "lbfgs")]
    L_star, _ = depth(g)
    at = g[g["layers"] == L_star]
    q = len(open(os.path.join(HAM_DIR, blk.replace("_w", "_MHS_omega") + ".txt")).readline().split(":")[0].strip())
    t4.append({"Molecule": mol, "Encoding": f"MHS({blk.split('_w')[1]})", "L": L_star, "q": q,
               "CNOT": L_star * (q - 1), "n": len(at), "Iter": mean_sd(at["iterations"])})
t4 = pd.DataFrame(t4)
t4.to_csv(os.path.join(here, "table4_iterations.csv"), index=False)
print(t4.to_string(index=False))
