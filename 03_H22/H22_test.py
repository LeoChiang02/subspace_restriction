"""
H_n chain, MHS(0) block, float64 -- uses SRS_ray.py from LeoChiang02/subspace_restriction.
Run from the repository root:   python h22_mhs0_float64.py [n_atoms]   (default 22)
"""
import sys, time
from math import comb
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import jax, ray
from pyscf import gto, scf

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # repo root (srs/)
from srs.SRS_ray import SymmetryBasis, HamiltonianLoader, build_hamiltonian

n = int(sys.argv[1]) if len(sys.argv) > 1 else 22
spacing = 1.0  # Angstrom, as in tab:H22

# ---- 0. precision guard ------------------------------------------------------
assert jax.config.jax_enable_x64, "JAX x64 must be enabled (SRS_ray sets it at import)"

# ---- 1. molecule and RHF reference (same construction as main.ipynb cell 10) -
atoms = "; ".join(f"H 0 0 {i * spacing:.2f}" for i in range(n))
mol = gto.M(atom=atoms, basis="sto-3g", spin=0, verbose=0)
mf = scf.RHF(mol).run(conv_tol=1e-12)
M, N = mol.nao, sum(mol.nelec)

# ---- 2. MHS(0) basis ---------------------------------------------------------
basis = SymmetryBasis(2 * M, N).build(mode="MHS", multiplicity=0)
D = len(basis)
assert D == comb(M, N // 2), (D, comb(M, N // 2))  # DOCI dimension

# ---- 3. restricted Hamiltonian (electronic part) -----------------------------
h1, h2, e_nuc = HamiltonianLoader(n_qubits=M).from_pyscf(mol, mf)
t0 = time.perf_counter()
H_bcsr = build_hamiltonian(h1, h2, basis, show=False)
ray.shutdown()
H = sp.csr_matrix((np.asarray(H_bcsr.data), np.asarray(H_bcsr.indices),
                   np.asarray(H_bcsr.indptr)), shape=H_bcsr.shape)
t_build = time.perf_counter() - t0
assert H.dtype == np.float64, H.dtype

# ---- 4. structural checks ----------------------------------------------------
asym = abs(H - H.T).max()                               # must be 0 (real symmetric)
hf_state = (1 << (2 * (N // 2))) - 1                     # orbitals 0..N/2-1 doubly occupied
idx = int(np.searchsorted(np.asarray(basis), hf_state))
assert int(basis[idx]) == hf_state, "HF determinant not found: check bit convention"
e_hf_diag = H[idx, idx] + e_nuc                          # must equal E_RHF

# ---- 5. lowest eigenvalue, SciPy Lanczos, machine-precision tolerance --------
t0 = time.perf_counter()
vals, vecs = spla.eigsh(H, k=2, which="SA", tol=0)
t_eig = time.perf_counter() - t0
v = vecs[:, 0]
resid = np.linalg.norm(H @ v - vals[0] * v)

print(f"H{n} | M={M} N={N} | D={D} | nnz={H.nnz} | dtype={H.dtype}")
print(f"E_nuc            = {e_nuc:.10f}")
print(f"E_RHF (PySCF)    = {mf.e_tot:.10f}")
print(f"<HF|H|HF> + Enuc = {e_hf_diag:.10f}   diff = {e_hf_diag - mf.e_tot:.2e}")
print(f"max|H - H^T|     = {asym:.2e}")
print(f"E_MHS(0)         = {vals[0] + e_nuc:.10f}  (float32 run: -11.48939 for H22)")
print(f"gap to 2nd       = {vals[1] - vals[0]:.3e} Ha   residual = {resid:.2e}")
print(f"variational      : E_MHS(0) <= E_RHF  ->  {vals[0] + e_nuc <= mf.e_tot + 1e-10}")
print(f"time: build {t_build:.1f} s, eigsh {t_eig:.1f} s")

'''
output:
H22 | M=22 N=22 | D=705432 | nnz=86062704 | dtype=float64
E_nuc            = 31.3261751188
E_RHF (PySCF)    = -11.4572903256
<HF|H|HF> + Enuc = -11.4572903256   diff = -8.53e-14
max|H - H^T|     = 0.00e+00
E_MHS(0)         = -11.4893921019  (float32 run: -11.48939 for H22)
gap to 2nd       = 2.635e-01 Ha   residual = 2.48e-13
variational      : E_MHS(0) <= E_RHF  ->  True
time: build 2130.7 s, eigsh 19.3 s
'''