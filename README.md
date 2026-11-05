# Subspace restriction and Hund's rule

Code and data for **"Molecular Ground State Simulation by Subspace Restriction and Hund's Rule"** (submitted to *J. Chem. Theory Comput.*). The full citation will be added on publication.

The repository provides the data that are hard to reproduce and the scripts that draw every figure. Tables can be assembled from the committed data; convenience scripts are included for Tables 1, 4, 5 and 7. Figure and table numbers follow the manuscript. `missing.md` lists the open items and how each closed item was resolved.

## Layout
| Folder | Content |
|---|---|
| `srs/` | Shared library. `SRS_ray.py` does basis restriction, Hamiltonian build and eigensolver. `encoding.py` holds `stream_encode`: labelling with centre shift s = ⌊(2^q−D)/2⌋, padding λ = 0 or min-diag, Walsh–Hadamard Pauli decomposition, T, Λ and N_XBM. `reference.py` and `analysis.py` provide RHF/UHF references, ⟨S²⟩ and fidelity. `driver.py` runs the per-molecule block solves. `hcb.py` builds the HCB Hamiltonian. |
| `data/` | `mol_info.py` (geometries from CCCBDB; H22 is a 1 Å chain defined in `03_H22`) and `records/*_records.json` (per-block D, q, T, Λ and spectrum). |
| `01_subspace_sizes/` | Tables 1 and 5, Figure 2 |
| `02_resources/` | Tables 2 and 3 |
| `03_H22/` | Table 6 and the H22 Pauli statistics |
| `04_reference_sensitivity/` | Figures 4–6, Table 7 |
| `05_PES/` | Figures 7 and 8 |
| `06_VQE/` | Figures 10–12, Table 4, success rates |

## Paper item → command → output
Run each command from its folder.

| Paper item | Folder | Command | Output |
|---|---|---|---|
| Table 1, Table 5 | `01_subspace_sizes` | `python table_sizes.py` | `table1.csv`, `table5.csv` |
| Figure 2 | `01_subspace_sizes` | `python plot_qubit_usage.py` | `qubit_usage_comparison.png/.pdf` |
| Table 2 (MHS rows), Table 3 (MHS columns) | `02_resources` | `encode_blocks.ipynb` | printed T, Λ, N_XBM; `02_resources/hamiltonians/` (not in git) |
| Table 2 (HCB rows), Table 3 (HCB columns) | `02_resources` | data: `hcb_hamiltonians/*.txt` (generator: `srs/hcb.py`) | — |
| Table 6 (E_MHS(0)) | `03_H22` | `python H22_test.py` (about 36 min) | output pasted in the script |
| H22 T, Λ, N_XBM | `03_H22` | `h22_encoding.ipynb` | `H22_MHS0_stream.json` |
| H22 DMRG reference | — | computed with block2 v0.5.4; code not included (see `03_H22/README.md`) | — |
| Figures 4–6 data | `04_reference_sensitivity` | `python runner.py` | `subspace_results.csv` |
| Figure 4 | `04_reference_sensitivity` | `python plot_energy.py` | `subspace_energy_linear.png` |
| Figure 5 | `04_reference_sensitivity` | `python plot_uhf_vs_rhf.py` (labels/legend differ from the published figure; plotted values identical) | `UHFvsRHF.png` |
| Figure 6 | `04_reference_sensitivity` | `python plot_fidelity.py` | `subspace_fidelity_linear.png` |
| Table 7 | `04_reference_sensitivity` | `python table_spin_purity.py` | `table7_spin_purity.csv` |
| Figures 7–8 | `05_PES` | `python plot_pes_gradient.py` (data from `Phase1.ipynb`) | `n2_pes_gradient.png`, `lih_pes_gradient.png` |
| Figures 10–11 | `06_VQE/analysis` | `python plot_vqe_blocks.py` (data from `mhs_hea/run_benchmark.py`) | `HF_OH_VQE.png`, `LiH_CH_VQE.png` |
| Figure 12 | `06_VQE/analysis` | `python plot_uccsd.py` (data from `uccsd/run_benchmark_uccsd.py`) | `UCCSD_box_scatter.png` |
| Success rates, Table 4 (Iter., MHS CNOT) | `06_VQE/analysis` | `python vqe_stats.py` | `success_rates.csv`, `table4_iterations.csv` |
| Table 4 (UCCSD CNOT, qubit exc.) | `06_VQE` | how the counts were obtained: `06_VQE/README.md`, "Table 4: UCCSD CNOT counts" | — |

## Environment
Python 3.10.

```
pip install -r requirements.txt
```

This installs everything except the VQE code. For the VQE code:
- **MHS-HEA runs:** `pip install -r 06_VQE/mhs_hea/requirements.txt` (PennyLane 0.42.3). The CUDA packages pinned there (`custatevec-cu12`, `jax-cuda12-*`, `nvidia-*-cu12`, `pennylane_lightning_gpu`) are optional; the published runs used the CPU (`BACKEND='cpu'` in `tool.py`).
- **UCCSD runs:** the uv environment in `06_VQE/uccsd/` (Python 3.13).
- **`srs/hcb.py`:** additionally needs `tequila-basic==1.9.12` and `pennylane==0.42.3`.

**VQE energies** are analytic (noiseless) PennyLane `lightning.qubit` expectation values.

## Files not in git and how to regenerate them
| File | Size | Regenerate with |
|---|---|---|
| `03_H22/H22_MHS0.npz` | 216 MB | GitHub Release asset, or `03_H22/h22_encoding.ipynb` cell [3] (cell 5 of the original notebook; about 36 min) |
| `04_reference_sensitivity/rerun.pkl` | 89.6 MB | GitHub Release asset. It holds the SCF orbitals behind the committed `subspace_results.csv`; a fresh `runner.py` run may choose different orbitals (see `04_reference_sensitivity/README.md`). |
| Encoded Pauli files and records for every block | — | `02_resources/encode_blocks.ipynb`, cell 2 (writes `02_resources/hamiltonians/`) |

## Credit
The JW/UCCSD VQE code in `06_VQE/uccsd/` is by Calvin Ku; original repository: https://github.com/calvinku96/subspace_restriction.
