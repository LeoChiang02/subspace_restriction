# 04 — Seniority blocks: energy cost, reference dependence, fidelity, spin purity

**Paper items:** Figures 4–6, Table 7 (spin purity).

| Command (run in this folder) | Output | Paper item |
|---|---|---|
| `python runner.py` | `subspace_results.csv` (+ `rerun.pkl`, not in git) | data for Figs 4–6 and Table 7 |
| `python plot_energy.py` | `subspace_energy_linear.png` | Fig 4 |
| `python plot_uhf_vs_rhf.py` | `UHFvsRHF.png` | Fig 5 |
| `python plot_fidelity.py` | `subspace_fidelity_linear.png` | Fig 6 |
| `python table_spin_purity.py` | `table7_spin_purity.csv` | Table 7 |

## `subspace_results.csv`
This is the committed data. It holds 20 molecules × {RHF, UHF} references. For PCS, MS and every MHS(Ω) block (the 71-block test set), it lists:
- the block energy and the FCI energy;
- the SCF energy and ⟨S²⟩ of the reference;
- ⟨S²⟩ of the block ground state;
- the fidelity to the (possibly degenerate) PCS ground space.

HS rows are included for H2, LiH and OH.

**Orbital gauge.** The committed MHS/HS values were computed from the SCF orbitals stored in `rerun.pkl` (Release asset) and in `05_PES/data/*_phase1.pkl`. For molecules with degenerate orbital sets (e.g. CO, CNH, CH4, N2), the SCF solution fixes the orbitals only up to a rotation within each degenerate set; fresh runs can choose a different rotation, and MHS/HS values may then differ by up to tens of mHa. Gauge-invariant quantities (E_FCI, E_SCF, PCS, MS) reproduce to ~1e-11 Ha. Phase 2 run from the stored orbitals reproduces every committed row.

## `runner.py`
- It calls `srs/driver.py`, which uses `srs/SRS_ray.py`, `srs/reference.py` and `srs/analysis.py`, for each molecule in `data/mol_info.py`.
- It was run on a large-memory instance. It sets Ray's spill directory to `/var/tmp` and uses a 40 GB object store.
- Its output path was changed from the original instance path to this folder; nothing else changed.
- `NAMES` lists 22 molecules, but the committed CSV contains 20 of them: C2H2 and NH2OH are absent.
- `rerun.pkl` (89.6 MB) is its pickled output, including the SCF objects. It is not in git; it is attached to the GitHub Release, alongside `03_H22/H22_MHS0.npz`.

## Plot scripts
The three scripts are the original plotting code with three changes:
- the input path is now `subspace_results.csv`;
- a `savefig` call was added before `plt.show()`;
- the filter that excluded H2 was removed, because the published figures include H2 (comments updated to match).

`plot_energy.py` (Fig 4) and `plot_fidelity.py` (Fig 6) reproduce the published figures pixel for pixel. For `plot_uhf_vs_rhf.py` (Fig 5), the labels and legend differ from the published figure; the plotted values are identical.

`table_spin_purity.py` reproduces Table 7 exactly: 71 blocks, with ⟨S²⟩ to 4 decimals for both references.
