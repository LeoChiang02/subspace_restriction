# 05 — Potential-energy scans of N2 and LiH

**Paper items:** Figure 7 (N2) and Figure 8 (LiH): energy, ⟨S²⟩ and fidelity versus bond length, RHF and UHF references.

| Command (run in this folder) | Output | Paper item |
|---|---|---|
| `python plot_pes_gradient.py` | `n2_pes_gradient.png`, `lih_pes_gradient.png` | Figs 7, 8 |
| `Phase1.ipynb` (with `scan.py`) | `<mol>_phase1.pkl`, `<mol>_scan.csv` | regenerates `data/` |

## Data (`data/`)
- **`lih_scan.csv`:** R = 0.80–4.00 Å in steps of 0.05 Å, 65 geometries.
- **`n2_scan.csv`:** R = 0.80–2.08 Å in steps of 0.02 Å, 65 geometries. This is the range shown in Fig. 7.
  - The scan itself covers 0.80–2.80 Å.
  - Beyond 2.08 Å the RHF orbital alignment fails at 2.44 Å (`Gauge_Deficit` = 0.267, against about 0.005 elsewhere), which makes the RHF MHS(0/2/4) energies jump by 28–56 mHa. Rows beyond 2.08 Å are therefore not included.
  - The committed file is the first 911 lines (the header plus every row with R ≤ 2.08 Å) of the full output, unchanged.
- **The kink near R = 1.42–1.46 Å** in the RHF MHS(0) curve is present in the data and is not an artefact of plotting.

## Phase-1 records (`data/lih_phase1.pkl`, `data/n2_phase1.pkl`)
These are the per-geometry SCF records from the AWS run that produced the committed CSVs. Each record holds the converged, orbital-aligned RHF and UHF objects (orbital coefficients, energies, ⟨S²⟩, gauge deficit) and the FCI energy. LiH has 65 geometries (0.80–4.00 Å) and N2 has 101 (0.80–2.80 Å).

- **Verified against the committed CSVs** at every geometry of `data/*_scan.csv`, both references: E_SCF, ⟨S²⟩_SCF and E_FCI agree to ≤ 6×10⁻¹⁴ Ha. Re-running Phase 2 from the stored orbitals at LiH 1.60 Å and N2 1.10 Å reproduces every row, including the MHS/HS energies, fidelities and ⟨S²⟩, to ≤ 2×10⁻¹² Ha.
- **Why they matter (orbital gauge).** The committed MHS/HS values were computed from the SCF orbitals stored in `rerun.pkl` (Release asset) and in `05_PES/data/*_phase1.pkl`. For molecules with degenerate orbital sets (e.g. CO, CNH, CH4, N2), the SCF solution fixes the orbitals only up to a rotation within each degenerate set; fresh runs can choose a different rotation, and MHS/HS values may then differ by up to tens of mHa. Gauge-invariant quantities (E_FCI, E_SCF, PCS, MS) reproduce to ~1e-11 Ha. Phase 2 run from the stored orbitals reproduces every committed row.
- **Skipping Phase 1.** Copy the two files into this folder (`05_PES/`) and run only cell 3 of `Phase1.ipynb` (Phase 2). It reads `<mol>_phase1.pkl` from the working folder, so nothing in `data/` is touched, and it writes `<mol>_scan.csv` here. Run it from `05_PES/`, not from `data/`: Phase 2 resumes from any `<mol>_scan.csv` it finds in the working folder and would append to the committed file.
- **N2 range.** The N2 pickle also holds the geometries from 2.10 to 2.80 Å. The committed `n2_scan.csv` is cut at 2.08 Å (see above); the RHF orbital alignment fails at 2.44 Å (gauge deficit 0.267), so those geometries are not validated.
- The pickles contain `pyscf` and `numpy` objects only. Load them in an environment with the pinned `pyscf` (`requirements.txt`).

## How the scan runs
- **Phase 1** (cells 1–2 of `Phase1.ipynb`) scans the reference SCF curves.
- **Phase 2** (cell 3) builds every subspace block at each geometry and appends rows to `<mol>_scan.csv` in this folder.
- Both phases use `scan.py` and the `srs/` library.

`plot_pes_gradient.py` is the plotting script from the original Gemini wrapper, extracted as a plain script. Only its input paths changed.

## Verification
The full scan was not rerun because of its runtime. The committed data were verified in three ways:

- **A. Consistency of the committed CSVs** (no recomputation). At every R and for both references:
  - |E_PCS − E_FCI| ≤ 10⁻⁸;
  - E_FCI ≤ every subspace energy + 10⁻⁸;
  - E_HS ≤ min_Ω E_MHS(Ω) + 10⁻⁸;
  - under RHF, ⟨S²⟩ of every MHS(Ω) equals (Ω/2)(Ω/2+1) to 10⁻⁸;
  - every fidelity lies in [0, 1] to 10⁻¹⁰.

  The worst violation is 3×10⁻¹².
- **B. One geometry recomputed** per molecule with `scan.py`: N2 at 1.10 Å and LiH at 1.60 Å, both references. E_FCI, E_SCF, ⟨S²⟩_SCF and the PCS and MS energies agree with the CSV to 2×10⁻¹¹. MHS and HS energies were not compared because they depend on the orbital gauge.
- **Figures 7–8** regenerated from these CSVs are pixel-identical to the committed figures and match the manuscript figures.
