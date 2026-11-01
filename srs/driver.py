"""
driver.py -- subspace-restriction rerun.

Per molecule and per reference mode:
  * one SCF reference (reference.build_reference); mol is never mutated
  * one integral transform and one FCI, shared by every subspace
  * PCS (k roots, for the degeneracy), MS, and the full omega sweep of MHS,
    all from the same mo_coeff
  * E_HS = min over MHS blocks (Lemma B); the assembled HS matrix is built
    only for the molecules in check_hs, as a standing test of build_hamiltonian
"""

import gc, math, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg
from pyscf import gto, fci

import jax
jax.config.update('jax_enable_x64', True)

from .SRS_ray import SymmetryBasis, HamiltonianLoader, build_hamiltonian
from .reference import build_reference
from .analysis import spin_square, fidelity_to_eigenspace

TOL = 1e-9
DEG_TOL = 1e-8            # eigenvalue spacing counted as degenerate
PCS_ROOTS = 12             # roots requested from PCS, to measure the degeneracy
DENSE_MAX = 4096          # at or below this, diagonalise densely


def feasible_omega(M, N):
    """Seniorities admitted by an M-orbital, N-electron Hund space.

    omega = Na - Nb with Na + Nb = N, so Nb = (N - omega)/2 and the number of
    doubly occupied orbitals is d = Nb.  Three constraints:
        omega = N (mod 2)   integrality of Nb
        Nb >= 0             omega <= N
        Na <= M             omega <= 2M - N
    """
    hi = min(N, 2*M - N)
    return list(range(N % 2, hi + 1, 2)) if hi >= 0 else []


def dim_mhs(M, N, omega):
    """Choose d doubly occupied orbitals, then omega singly from the rest."""
    d = (N - omega) // 2
    if d < 0 or omega < 0 or d + omega > M:
        return 0
    return math.comb(M, d) * math.comb(M - d, omega)


def block_counts(M, N, omega):
    """(|D|, |E|) for a fixed-omega Hund block: doubly occupied, empty."""
    d = (N - omega) // 2
    return d, M - d - omega


def seniority_of(basis, M):
    """Seniority (Na - Nb) of each determinant in a bit-encoded basis."""
    b = [int(x) for x in np.asarray(basis)]
    a = np.array([[(s >> (2*p)) & 1 for p in range(M)] for s in b])
    c = np.array([[(s >> (2*p + 1)) & 1 for p in range(M)] for s in b])
    return (a.sum(1) - c.sum(1)).astype(int)


def to_dense(H, D):
    return np.asarray(sp.csr_matrix((np.asarray(H.data),
                                     np.asarray(H.indices),
                                     np.asarray(H.indptr)),
                                    shape=(D, D)).todense())


def lowest_states(H, D, e_nuc, k=1):
    """The k lowest eigenpairs of an assembled subspace Hamiltonian.

    which='SA' selects smallest *algebraic*, which is what a ground state is;
    'SM' (smallest magnitude) would return whatever sits nearest zero.  tol=0
    means machine precision -- ARPACK's tol is relative to ||H||, so a loose
    value licenses an absolute error far above the 1e-9 used by the assertions
    below.  ncv is enlarged because these spectra are clustered: a bigger
    Krylov space cuts the restarts needed to separate near-degenerate roots.
    """
    if D <= DENSE_MAX or k >= D - 1:
        w, V = scipy.linalg.eigh(to_dense(H, D))
        return w[:k] + e_nuc, V[:, :k]
    A = sp.csr_matrix((np.asarray(H.data), np.asarray(H.indices),
                       np.asarray(H.indptr)), shape=(D, D))
    w, V = spla.eigsh(A, k=k, which='SA', ncv=min(D, 50), tol=0)
    order = np.argsort(w)
    return w[order] + e_nuc, V[:, order]


def solve_subspace(h1, h2, e_nuc, n_qubits, N, mode, omega=None, k=1,
                   show=False):
    """Assemble and diagonalise one subspace.

    Returns (record, evals, evecs).  Only k=1 is used for MS/MHS/HS; PCS asks
    for more so that the ground-state degeneracy can be measured, not assumed.
    """
    basis = np.asarray(SymmetryBasis(n_qubits, N).build(mode=mode,
                                                        multiplicity=omega))
    D = int(basis.size)
    if D == 0:
        return {'mode': mode, 'omega': omega, 'D': 0, 'E': None}, None, None

    t0 = time.time()
    H = build_hamiltonian(h1, h2, basis, show=show)
    dtype = str(np.asarray(H.data).dtype)
    assert dtype == 'float64', f'{mode} omega={omega}: matrix narrowed to {dtype}'

    w, V = lowest_states(H, D, e_nuc, k=min(k, D))
    rec = {'mode': mode, 'omega': omega, 'D': D,
           'nnz': int(np.asarray(H.data).size), 'E': float(w[0]),
           'dtype': dtype, 'secs': round(time.time() - t0, 2),
           'vec': np.asarray(V[:, 0], dtype=float), 'basis': basis}
    del H
    gc.collect()
    return rec, w, V


def hs_offblock_max(h1, h2, n_qubits, N, M):
    """Largest cross-seniority matrix element of the assembled HS matrix.

    Lemma B follows from [S_z, H] = 0 alone: inside HS at fixed N the seniority
    equals Na - Nb, and H conserves Na and Nb separately, so every
    cross-seniority block vanishes identically.  A nonzero value localises a
    defect to build_hamiltonian, before any eigensolve is run.
    """
    basis = np.asarray(SymmetryBasis(n_qubits, N).build(mode='HS'))
    H = build_hamiltonian(h1, h2, basis, show=False)
    Hd = to_dense(H, int(basis.size))
    w = seniority_of(basis, M)
    worst, pairs, uw = 0.0, {}, np.unique(w)
    for i, w1 in enumerate(uw):
        for w2 in uw[i+1:]:
            blk = Hd[np.ix_(w == w1, w == w2)]
            nrm = float(np.abs(blk).max()) if blk.size else 0.0
            pairs[f'{w1}-{w2}'] = nrm
            worst = max(worst, nrm)
    del H, Hd
    gc.collect()
    return worst, pairs


def run_molecule(name, spec, basis_set='sto-3g', modes=('rhf', 'uhf'),
                 check_hs=False, show=False):
    mol = gto.M(atom=spec['atom'], basis=basis_set, spin=spec['spin'],
                charge=spec.get('charge', 0), verbose=0)
    M, N = int(mol.nao), int(mol.nelectron)
    nq = 2 * M
    omegas = feasible_omega(M, N)
    assert mol.spin in omegas, f'{name}: target omega={mol.spin} not feasible'
    ovlp = mol.intor('int1e_ovlp')
    e_fci = None                     # computed once, shared by both modes

    out = {'name': name, 'M': M, 'N': N, 'charge': int(mol.charge),
           'basis_set': basis_set, 'omega_target': int(mol.spin),
           'omegas': omegas, 'modes': {}}

    for mode in modes:
        mf, info = build_reference(mol, mode)
        mo = tuple(mf.mo_coeff) if np.asarray(mf.mo_coeff).ndim == 3 else None
        ov = ovlp if mo is not None else 1
        h1, h2, e_nuc = HamiltonianLoader(nq).from_pyscf(mol, mf)
        if e_fci is None:            # FCI is orbital-basis independent
            cis = fci.FCI(mol, mf.mo_coeff)
            cis.conv_tol = 1e-12
            e_fci = float(cis.kernel()[0])

        rec = {'ref': info, 'mf': mf, 'e_fci': e_fci,
               'subspaces': {}, 'mhs': {}}

        # --- PCS: supplies the reference eigenspace for every fidelity ------
        k = PCS_ROOTS
        while True:
            r, w, V = solve_subspace(h1, h2, e_nuc, nq, N, 'PCS', None,
                                     k=k, show=show)
            g = int(np.sum(np.abs(w - w[0]) < DEG_TOL))
            if g < len(w) or k >= 4 * PCS_ROOTS:
                break
            k *= 2          # degeneracy saturated the roots -- widen and retry
        assert g < len(w), (
            f'{name} {mode}: PCS degeneracy {g} still saturates {len(w)} roots '
            f'at k={k}')
        Vdeg = V[:, :g]
        rec['pcs_degeneracy'] = g
        rec['pcs_deg_vecs'] = Vdeg
        rec['pcs_gap_to_next'] = float(w[g] - w[0])
        rec['subspaces']['PCS'] = r

        def annotate(r, omega):
            """Attach spin, fidelity and block counts to a subspace record."""
            if r.get('D'):
                r['spin_sq'] = spin_square(r['vec'], r['basis'], M, mo, ov)[0]
                r['fidelity'] = fidelity_to_eigenspace(
                    r['vec'], r['basis'], Vdeg,
                    rec['subspaces']['PCS']['basis'])
                r['n_D'], r['n_E'] = (block_counts(M, N, omega)
                                      if omega is not None else (None, None))
            return r

        annotate(rec['subspaces']['PCS'], None)

        r, _, _ = solve_subspace(h1, h2, e_nuc, nq, N, 'MS', int(mol.spin),
                                 show=show)
        rec['subspaces']['MS'] = annotate(r, None)

        for wom in omegas:
            r, _, _ = solve_subspace(h1, h2, e_nuc, nq, N, 'MHS', wom, show=show)
            assert r['D'] == dim_mhs(M, N, wom), (name, wom, r['D'])
            r['D_closed_form'] = dim_mhs(M, N, wom)
            rec['mhs'][wom] = annotate(r, wom)

        E_blocks = {k_: v['E'] for k_, v in rec['mhs'].items()}
        rec['E_HS'] = min(E_blocks.values())               # Lemma B
        rec['E_HS_argmin'] = min(E_blocks, key=E_blocks.get)
        rec['E_MHS_target'] = E_blocks[int(mol.spin)]

        if check_hs:
            r, _, _ = solve_subspace(h1, h2, e_nuc, nq, N, 'HS', None, show=show)
            rec['subspaces']['HS'] = annotate(r, None)
            assert r['D'] == sum(v['D'] for v in rec['mhs'].values()), name
            assert abs(r['E'] - rec['E_HS']) < TOL, (
                f"{name} {mode}: assembled HS {r['E']} != min over blocks "
                f"{rec['E_HS']}")
            worst, pairs = hs_offblock_max(h1, h2, nq, N, M)
            rec['hs_offblock_max'], rec['hs_offblock_pairs'] = worst, pairs
            assert worst < 1e-12, (
                f'{name} {mode}: cross-seniority element {worst:.3e} -- '
                f'build_hamiltonian violates Lemma B')

        E_pcs = rec['subspaces']['PCS']['E']
        E_ms = rec['subspaces']['MS']['E']

        # PCS spans every Sz sector at fixed N and therefore *contains* the FCI
        # space of the target sector, so E_PCS <= E_FCI.  Equality holds iff
        # the ground multiplet reaches the target Sz; the gap is recorded
        # rather than asserted, since a strict inequality is physics, not a bug.
        assert E_pcs <= e_fci + TOL, (name, mode, 'PCS above FCI -- impossible')
        rec['e_fci_minus_pcs'] = e_fci - E_pcs
        assert rec['E_HS'] >= E_pcs - TOL, (name, mode, 'HS below PCS')
        assert E_ms >= E_pcs - TOL, (name, mode, 'MS below PCS')
        # MS is not a subspace of the target-sector FCI space -- it IS that
        # space, so this is an equality, and the only check that compares
        # build_hamiltonian against a fully independent implementation.
        assert abs(E_ms - e_fci) < 1e-10, (
            name, mode, f'MS != FCI by {E_ms - e_fci:.3e} -- '
            f'build_hamiltonian disagrees with PySCF FCI')
        assert rec['E_MHS_target'] >= rec['E_HS'] - TOL, (
            name, mode, 'MHS(S) below HS')

        # mol.spin is 2*Sz, not 2*S: the FCI ground state of the target sector
        # may carry any S >= Sz.  Recorded, never asserted.
        S_t = mol.spin / 2.0
        rec['fci_is_target_spin'] = bool(
            abs(rec['subspaces']['MS']['spin_sq'] - S_t*(S_t + 1)) < 1e-6)

        rec['err_PCS'] = E_pcs - e_fci                     # signed, never abs
        rec['err_MS'] = E_ms - e_fci
        rec['err_HS'] = rec['E_HS'] - e_fci
        rec['err_MHS'] = rec['E_MHS_target'] - e_fci
        out['modes'][mode] = rec
        gc.collect()

    return out


CSV_COLS = ['Molecule', 'Reference', 'Subspace', 'Omega', 'Target_Spin',
            'Status', 'Subspace_Dim', 'Fidelity_to_PCS', 'PCS_Degeneracy',
            'Subspace_Energy', 'Ref_SCF_Energy', 'FCI_Energy',
            'Energy_Error_vs_FCI', 'Spin_Sq_SCF', 'Spin_Sq_Subspace',
            'n_D', 'n_E', 'PCS_Gap_To_Next', 'FCI_Is_Target_Spin', 'M', 'N']


def to_rows(results):
    """Flatten the pickle structure into CSV rows, one per subspace block."""
    rows = []
    for name, out in results.items():
        for mode, m in out['modes'].items():
            base = {'Molecule': name, 'Reference': mode.upper(),
                    'Target_Spin': out['omega_target'],
                    'Ref_SCF_Energy': m['ref']['e_tot'],
                    'FCI_Energy': m['e_fci'],
                    'Spin_Sq_SCF': m['ref']['spin_sq'],
                    'PCS_Degeneracy': m['pcs_degeneracy'],
                    'PCS_Gap_To_Next': m['pcs_gap_to_next'],
                    'FCI_Is_Target_Spin': m['fci_is_target_spin'],
                    'M': out['M'], 'N': out['N']}
            items = list(m['subspaces'].items())
            items += [(f'MHS(w={w})', v) for w, v in m['mhs'].items()]
            for label, r in items:
                row = dict(base, Subspace=label)
                if r.get('skipped'):
                    rows.append(dict(row, Status='SKIPPED'))
                    continue
                rows.append(dict(
                    row, Status='SUCCESS', Omega=r.get('omega'),
                    Subspace_Dim=r['D'], Fidelity_to_PCS=r.get('fidelity'),
                    Subspace_Energy=r['E'],
                    Energy_Error_vs_FCI=r['E'] - m['e_fci'],   # signed
                    Spin_Sq_Subspace=r.get('spin_sq'),
                    n_D=r.get('n_D'), n_E=r.get('n_E')))
    return rows