# 02 — Encoding resources (T, N_XBM, Λ) and MHS vs HCB

**Paper items:** Table 2 (MHS and HCB rows) and Table 3.

## MHS blocks: `encode_blocks.ipynb`
- **What it does:** builds every MHS(Ω) block of H2, LiH, HF, BeH2, H2O, OH and CH with `srs/SRS_ray.py` (RHF orbitals; ROHF for OH and CH). It then encodes each block with `srs/encoding.py` (`stream_encode`), with padding λ = 0 and λ = min-diag.
- **What it prints:** D, q, T, Λ (`L_noI`), N_XBM and E₀ for every block.
- **How to run:** open the notebook in this folder and run cells 0–2.
  - Cell 0 installs `../requirements.txt`.
  - Cell 2 writes `hamiltonians/*_MHS_omega*.txt` and `hamiltonians/*_records.json`. These outputs are regenerable and not committed.
- **Cell 3** is not code. It is a pasted copy of the output of cell 2 from the original run, kept as the record of that run.
- **Committed data:** `../data/records/*_records.json`. These were produced by an earlier qiskit-based pipeline, which is not included.
  - This notebook reproduces their T exactly and their Λ to 2×10⁻⁶.
  - It also adds N_XBM, which the records do not contain.
- **Pauli files:** the Pauli files the notebook writes equal the files used for the VQE (`../06_VQE/mhs_hea/hamiltonians/`) to within the 7th printed decimal.

The notebook is cells 0–3 of the original `H22_encoding.ipynb`. The only changes are import paths (`srs/`, `data/`) and the requirements path.

## HCB (Ω = 0): `hcb_hamiltonians/*_HCB_omega0.txt`
- These files are the data behind the HCB columns of Tables 2–3: q, T, N_XBM, Λ_HCB, occupancy and the shot-bound ratios.
- Each file's lowest eigenvalue in the N/2-pair sector equals E_MHS(0) to 3×10⁻⁷ Ha.
- The identity coefficient is electronic (E_nuc not included).

**`../srs/hcb.py` is an independent re-implementation; the original generator is not available.**
- It builds the HCB Hamiltonian with Tequila (`frozen_core=False`).
- It agrees with the committed `*_HCB_omega0.txt` files to 4×10⁻⁷ elementwise.
- It covers H2, LiH, HF, BeH2 and H2O. For all five, q, T, N_XBM and Λ_HCB round to the values in Tables 2–3:
  - H2: 2 / 6 / 2 / 1.542
  - LiH: 6 / 52 / 16 / 9.302
  - HF: 6 / 52 / 16 / 64.707
  - BeH2: 7 / 71 / 22 / 16.112
  - H2O: 7 / 71 / 22 / 54.790
- It agrees with the committed files to ≤ 4.3×10⁻⁷ elementwise. The H2, BeH2 and H2O entries were added for this check; their geometries come from `data/mol_info.py`.
- `srs_hamiltonian()` loads its MHS(0) file from `06_VQE/mhs_hea/hamiltonians/`, which holds only H2, HF and LiH. For BeH2 and H2O, generate the file with `encode_blocks.ipynb`.
- Extra requirements: `tequila-basic==1.9.12`, `pennylane==0.42.3`, `openfermion==1.8.1`.

## Tables 2–3
The HCB and JW columns of Tables 2–3 and the shot-bound ratios are not assembled by a script here; JW is standard, and the tables can be assembled from the committed data (`hcb_hamiltonians/`, `../data/records/`).
