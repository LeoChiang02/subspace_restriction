# 01 — Subspace sizes and qubit counts

**Paper items:** Table 1, Table 5, Figure 2.

| Command (run in this folder) | Output | Paper item |
|---|---|---|
| `python table_sizes.py` | `table1.csv`, `table5.csv` | Tables 1 and 5 |
| `python plot_qubit_usage.py` | `qubit_usage_comparison.png`, `.pdf` | Figure 2 |

## Closed forms used by `table_sizes.py`
For M spatial orbitals, N electrons and spin 2S, all in STO-3G and taken from `../data/mol_info.py`:

| Subspace | Dimension |
|---|---|
| PCS | C(2M, N) |
| MS (fixed M_s = S) | C(M, N_α) · C(M, N_β), with N_α − N_β = 2S |
| HS (Prop. SizeEstimation) | Σ_{k=0}^{⌊N/2⌋} C(M, N−k) · C(N−k, k) |
| MHS(2S), the block listed in the tables | C(M, (N+2S)/2) · C((N+2S)/2, (N−2S)/2) |

- Qubits are q = ⌈log₂ D⌉, computed in exact integer arithmetic. JW uses 2M qubits.
- H22 is the 1 Å linear chain used in `../03_H22`.
- The H22 entries for PCS, MS and HS are marked "*" in Table 5 because those subspaces were not tested. The script prints their exact sizes: 2.10×10¹², 4.98×10¹¹ and 3.24×10⁹.

`table_sizes.py` reproduces every cell of Tables 1 and 5.

`plot_qubit_usage.py` needs no input data. It keeps only those M for which N = cM is an even integer.

**Fonts.** The committed `qubit_usage_comparison.png`/`.pdf` were rendered with the Inter font. A fresh environment from `requirements.txt` falls back to DejaVu Sans, so a rerun differs in the text glyphs; the curves are identical.
