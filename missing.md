# Open items

Status of every paper item that this repository cannot yet fully reproduce. Figure and table numbers follow the manuscript.

The repository provides the data that are hard to reproduce and the scripts for every figure. Tables can be assembled by readers from the committed data.

## Still to be supplied
| # | Paper item | Missing | Where it will go |
|---|---|---|---|
| — | LICENSE | LICENSE (pending co-author agreement). | repository root |

## Mostly closed
- **G8, Table 6 run log.** The output of `03_H22/H22_test.py` exists only as a block pasted at the end of the script. The build cell of `03_H22/h22_encoding.ipynb` saved no output. No separate log file exists.

## Closed
- **G1/G2**, Tables 1 and 5 and the HS dimensions: `01_subspace_sizes/table_sizes.py`, using the closed forms. All cells match.
- **G3**, N_XBM: `srs/encoding.py` and `02_resources/encode_blocks.ipynb`.
- **G4**, HCB generator: `srs/hcb.py`, an independent re-implementation covering H2, LiH, HF, BeH2 and H2O. It agrees with the committed files to ≤ 4.3×10⁻⁷, and its q, T, N_XBM and Λ match Tables 2–3. The original generator is not available.
- **G5, G6** (assembly of Tables 2–3 and the JW rows): dropped from scope. JW is standard, and table assembly is left to readers.
- **G9**, H22 stream: `03_H22/h22_encoding.ipynb`.
- **G10**, H22 DMRG reference (Table 6): computed with block2 v0.5.4 (https://github.com/block-hczhai/block2-preview/tree/v0.5.4), as stated in `03_H22/README.md`. The DMRG input and output are not included.
- **G13**, CH configuration of the UCCSD runs: added to `06_VQE/uccsd/run_benchmark_uccsd.py` from the authors' configuration (C 0 0 0; H 0 0 1.1199 Å; charge 0; spin 1, i.e. `mult` 2). With seed 0 it reproduces the committed CH rows exactly for both L-BFGS and Adam (`06_VQE/uccsd/README.md`).
- **G14**, success rates, §5.2 depths and Table 4 (L, MHS CNOT, Iter.): `06_VQE/analysis/vqe_stats.py`. All values match. L\* is the smallest depth at which at least 8 of the 15 L-BFGS runs reach 1.6 mHa.
- **G15**, Table 4 UCCSD CNOT and qubit-excitation counts: explained in `06_VQE/README.md` ("Table 4: UCCSD CNOT counts"). All eight numbers were checked in PennyLane 0.45.1 (locked environment of `06_VQE/uccsd/`) and against Yordanov et al., Phys. Rev. A 102, 062612 (2020).
- **G16**, Table 7: `04_reference_sensitivity/table_spin_purity.py`. All 71 rows match.
- **G11, Figures 4–6:** `04_reference_sensitivity/plot_energy.py`, `plot_uhf_vs_rhf.py` and `plot_fidelity.py`. Figures 4 and 6 are pixel-identical to the published figures. For Figure 5 the labels and legend differ from the published figure, and the plotted values are identical.
- **Figures 7, 8, 10, 11, 12:** plot scripts included and verified against the manuscript figures. Figure 12 is `06_VQE/analysis/plot_uccsd.py`, with its title matched to the published figure.
- **PES Phase-1 records:** `05_PES/data/{lih,n2}_phase1.pkl` are committed. They match the committed CSVs, and Phase 2 can be run from them (`05_PES/README.md`).
