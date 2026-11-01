# JW/UCCSD VQE

Code by Calvin Ku; original repository: https://github.com/calvinku96/subspace_restriction (commit 8f4143af24c433cf7ea0b4d837fe05ac463e1644)

| File | Content |
|---|---|
| `run_benchmark_uccsd.py` | Driver. The JW Hamiltonian is built on the fly with `qml.qchem.molecular_hamiltonian`. Runs L-BFGS and Adam, 15 seeds, at most 1000 iterations, convergence when ΔE < 10⁻⁶ Ha. |
| `tool.py` | Calvin's version of `tool.py`, which adds `VQE_UCCSD`. It differs from `../mhs_hea/tool.py`. |
| `fci.py`, `ufci.py` | RHF/FCI and UHF/FCI reference energies. |
| `SRS_VQE_UCCSD.csv` | All 150 trials (H2, HF, LiH, OH, CH × L-BFGS, Adam × 15 seeds). These data underlie Fig. 12 and the JW rows of Table 4. |
| `pyproject.toml`, `uv.lock`, `.python-version` | Calvin's uv environment: Python 3.13, jax 0.10.0, pennylane ≥ 0.45.1. |

## Changes from the original
- A one-line credit header was added to each `.py` file and to `pyproject.toml`.
- In `run_benchmark_uccsd.py`, the package-relative `from .tool import ...` became `from tool import ...`, so the script runs from this folder: `python run_benchmark_uccsd.py`.
- There are no other changes.

## Known gap
The committed driver lists H2, HF, LiH and OH. The CH configuration that produced the CH rows of `SRS_VQE_UCCSD.csv` is not in commit 8f4143a (G13 in `../../missing.md`).
