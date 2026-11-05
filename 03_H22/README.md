# 03 — H22 (linear chain, 1 Å spacing, STO-3G)

**Paper items:**
- Table 6: E_MHS(0) and E_RHF.
- The H22 Pauli statistics quoted in the text: T, Λ and N_XBM.
- The DMRG reference (computed with block2; see below).

| File | What it is |
|---|---|
| `H22_test.py` | Builds the MHS(0) block (D = 705 432, q = 20) in float64 and solves it with SciPy `eigsh`. Result: E_MHS(0) = −11.4893921019 Ha. The build takes about 36 min. |
| `h22_encoding.ipynb` | Builds the same block and streams its Pauli decomposition (`srs/encoding.py`). |
| `H22_MHS0_meta.npy` | [E_nuc, E_RHF], saved by the build cell. |
| `H22_MHS0_stream.json` | Output of the stream cell: T = 74 803 927 415, Λ (`L_noI`) = 27 042.51, N_XBM = 316 619, E₀ and the other fields. |

**Run log.** The output of `H22_test.py` is pasted at the end of the script. That pasted block is the only record of the run; no separate log was kept. The script's docstring refers to it as `h22_mhs0_float64.py`.

## `h22_encoding.ipynb`
It contains cells 0, 1, 4, 5 and 6 of the original `H22_encoding.ipynb`. Cell 1 holds the function definitions that the build cell needs. Only import paths changed.

| Cell here | Cell in the original | Content |
|---|---|---|
| [0] | 0 | `pip install -r ../requirements.txt` |
| [1] | 1 | `get_ham`, `build_block` and related helpers |
| [2] | 4 | H22 geometry |
| [3] | 5 | Builds the block (about 36 min) and writes `H22_MHS0.npz` and `H22_MHS0_meta.npy` |
| [4] | 6 | `stream_encode(H, pad='mindiag', n_workers=30)`, which writes `H22_MHS0_stream.json` |

## `H22_MHS0.npz` (216 MB) is not in git
To get it, either:
- regenerate it with cell [3] (cell 5 of the original notebook, about 36 min), or
- download it from the GitHub Release asset of this repository.

## DMRG reference
The DMRG reference (−11.817206 Ha, bond dimension 2000) was computed with block2 v0.5.4 (https://github.com/block-hczhai/block2-preview/tree/v0.5.4). The DMRG input and output are not included in this repository.
