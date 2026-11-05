# 06 — VQE benchmarks

**Paper items:**
- Figures 10–11: MHS-HEA VQE per seniority block.
- Figure 12: JW/UCCSD.
- Table 4: circuit resources and iterations.
- The success rates quoted in the text.
- The CH footnote: the FCI quartet ground state.

All VQE energies are analytic (noiseless) PennyLane `lightning.qubit` expectation values.

## `mhs_hea/`: hardware-efficient ansatz on the MHS blocks
| File | Content |
|---|---|
| `run_benchmark.py`, `tool.py` | Driver and ansatz (`VQE_HEA_RY`: RY layers with a linear CNOT chain). Depths L = 1–7, L-BFGS and Adam (lr 0.1), at most 1000 iterations, convergence when ΔE < 10⁻⁶ Ha, 15 seeds (seed = trial index). |
| `hamiltonians/*_MHS_omega*.txt` | The encoded MHS(Ω) blocks used for these runs (padding λ = 0). |
| `SRS_VQE_5cases.csv` | All 2310 trials: H2, HF, LiH, OH and CH blocks. The file is a concatenation, so it contains one repeated header row and some blank lines. |
| `requirements.txt` | Environment of these runs: pennylane 0.42.3, lightning 0.42.0, jax 0.6.2, optax 0.2.5. The CUDA packages it pins are optional; the published runs used the CPU (`BACKEND='cpu'` in `tool.py`). |

**How to rerun.** `run_benchmark.py` currently has the three CH blocks enabled. To run the other blocks, uncomment their entries in `MOLECULES`; each entry's `exact_energy` is that block's ground energy E_MHS(Ω). Run it from `mhs_hea/`.

**Output file names.** The drivers write `benchmark_raw_trials.csv` (`mhs_hea/`) and `benchmark_raw_trials_uccsd.csv` (`uccsd/`). The committed `mhs_hea/SRS_VQE_5cases.csv` and `uccsd/SRS_VQE_UCCSD.csv` are the raw-trial files of the published runs, with the same columns (the 5-cases file concatenates the per-block runs). The `benchmark_*.csv` outputs of a rerun are not tracked by git.

## `uccsd/`: JW/UCCSD (code by Calvin Ku)
See `uccsd/README.md`.

## `analysis/`
| Command (run in `analysis/`) | Output | Paper item |
|---|---|---|
| `python plot_vqe_blocks.py` | `HF_OH_VQE.png`, `LiH_CH_VQE.png` | Figs 10, 11 |
| `python plot_uccsd.py` | `UCCSD_box_scatter.png` | Fig 12 |
| `python vqe_stats.py` | `success_rates.csv`, `depth_per_block.csv`, `table4_iterations.csv` | text, §5.2 depths, Table 4 (L, MHS CNOT, Iter.) |
| `python ch_ground_state_fci.py` | printed FCI roots | CH footnote |

**`plot_vqe_blocks.py`** draws only converged runs. Its points are strip-plot jitter, which is unseeded, so the dots move slightly between runs; the boxes do not.

**`plot_uccsd.py`** draws the converged UCCSD runs of `../uccsd/SRS_VQE_UCCSD.csv` (L-BFGS and Adam, 15 seeds each) as boxes and points against the total FCI energy. All 150 runs converged and lie within 1.6 mHa.
- The plotting function is as supplied. The script header adds the imports, the CSV load and the `fci_tot` dictionary. Its entries for HF, LiH, CH and OH are the ones used in `plot_vqe_blocks.py`, and H2 is the FCI energy in `04_reference_sensitivity/subspace_results.csv`.
- The title string was changed to "UCCSD VQE Energy Error" to match the published figure.

**`vqe_stats.py`:**
- **Success:** a run succeeds if it converged and |E_r − E_MHS(Ω)| ≤ 1.6 mHa, where E_MHS(Ω) is the block's exact ground energy from `data/records`. Non-converged runs count as failures. This gives the same counts as the CSV's "Chemically Accurate" status.
- **Success rates** are pooled over the HF, LiH, OH and CH blocks; H2 is excluded.
- **Depth L\*:** the smallest L at which at least 8 of the 15 L-BFGS runs at that depth succeed. This is "the depth at which most L-BFGS runs reach 1.6 mHa" in §5.2. `depth_per_block.csv` lists it for every block.
- **Table 4 Iter.:** the L-BFGS mean ± sample standard deviation over the 15 initializations at L\*. MHS CNOT = L\*(q − 1).
- **Results:** the script reproduces Table 4 (L = 2, 2, 4, 6; CNOT 4, 6, 16, 30; Iter. 38.7 ± 11.2, 31.4 ± 15.9, 157.9 ± 44.7, 230.4 ± 99.0), the §5.2 depths (HF MHS(2) L = 1, CH MHS(5) L = 1, LiH MHS(4) L = 2) and the quoted success rates.

**`ch_ground_state_fci.py`** prints the CH FCI roots in the (4α, 3β) sector:
- the quartet ground state at E = −37.811349 Ha;
- the ²Π doublet 0.1607 mHa above it.

Its `spin_square` call was fixed for pyscf 2.14.

## Reproducibility of reruns
**Recorded data.** The committed CSVs reproduce every number quoted in the paper.

**Fresh reruns** of the MHS-HEA and UCCSD drivers on other hardware: for the blocks tested here they reproduce the trials exactly (energies and iteration counts), except OH MHS(1).
- The L-BFGS trajectories for this block are hardware-sensitive.
- A rerun at L = 4 on a different CPU (the original runs used an Intel i9-13900H) gives 8/15 successes instead of 9/15. L\* = 4 is unchanged.
- In that rerun, Iter. is 145.7 ± 39.6 versus the published 157.9 ± 44.7.

## Table 4: UCCSD CNOT counts
The JW/UCCSD CNOT counts in Table 4 (HF 2400, LiH 6976, OH 4120, CH 7688) are those of the `qml.UCCSD` circuit built as in `uccsd/tool.py`, in PennyLane 0.45.1 (the version locked in `uccsd/uv.lock`), decomposed into CNOTs and single-qubit gates. Each single excitation spanning n qubits uses 4(n−1) CNOTs, and each double excitation spanning n qubits uses 16(n−1). This is the standard CNOT-staircase construction described in Appendix A of Yordanov, Arvidsson-Shukur and Barnes, Phys. Rev. A 102, 062612 (2020).

The qubit-excitation counts (HF 345, LiH 1020, OH 624, CH 1165) are 2N_s + 13N_d. They use the qubit-excitation circuits of the same paper: 2 CNOTs per single excitation (footnote 1) and 13 per double excitation (Fig. 7). N_s and N_d are the numbers of single and double excitations returned by `qml.qchem.excitations` for 12 qubits: HF 10/25, LiH 16/76, OH 13/46 and CH 17/87. Both counts are summed over the excitations, with no cancellation between neighbouring excitation circuits.
