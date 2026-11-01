"""
scan.py -- PES scan in two phases.

  Phase 1 (cheap): SCF references + FCI at every geometry, with convergence,
    stability and gauge diagnostics.  Plot it, look at it, and only then
  Phase 2 (expensive): subspace energies reusing the *stored* mean-field
    objects, so Phase 2 cannot silently use a different reference from the
    one you approved.

SRS_ray is used only for SymmetryBasis / HamiltonianLoader / build_hamiltonian.
"""

import os, pickle, time
import numpy as np
from pyscf import gto, fci

import jax
jax.config.update('jax_enable_x64', True)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # repo root (srs/)
from srs.SRS_ray import SymmetryBasis, HamiltonianLoader, build_hamiltonian
from srs.reference import build_reference
from srs.analysis import spin_square, fidelity_to_eigenspace
from srs.driver import feasible_omega, dim_mhs, block_counts, lowest_states

TOL = 1e-9
DEG_TOL = 1e-8
PCS_ROOTS = 12

# PySCF's FCI defaults (max_space=12, max_cycle=50) do not converge N2/STO-3G
# anywhere on the dissociation tail.  Measured 2026-09-07: cis.converged was
# False at every geometry from 2.40 to 2.80 A, and kernel() returns silently.
FCI_MAX_SPACE = 40
FCI_MAX_CYCLE = 200
# Backstop for the failure cis.converged cannot see (see scan_reference).
MONOTONE_TAIL = True


def _mo_spins(mf):
    """MO coefficient blocks, one per spin channel."""
    C = np.asarray(mf.mo_coeff)
    return [C[0], C[1]] if C.ndim == 3 else [C]


def gauge_deficit(mfA, molA, mfB, molB):
    """1 - min_p |<a_p | b_pi(p)>| using the true cross-geometry AO overlap.

    Orbitals drift smoothly with geometry, so this is never machine zero for a
    real step -- it scales as (dtheta/dR * h)^2 / 2, of order 1e-4 for a
    0.02 A step.  What matters is a *spike* against the local trend: a gauge
    rotation inside a degenerate manifold is O(1) and discontinuous, and moves
    E_MHS without moving the SCF energy.  Callers must therefore threshold on
    the trend, not on an absolute floor (see _spikes).

    Both spin channels are checked, so a beta-only rotation is visible.
    """
    S = gto.intor_cross('int1e_ovlp', molA, molB)
    worst = 0.0
    for Ca, Cb in zip(_mo_spins(mfA), _mo_spins(mfB)):
        G = np.abs(Ca.T @ S @ Cb)
        pi = G.argmax(axis=0)
        worst = max(worst, float(np.max(1.0 - G[pi, np.arange(G.shape[1])])))
    return worst

def align_gauge(mf, mol, mf_ref, mol_ref, deg_tol=1e-6):
    """Rotate mf's orbitals into maximum alignment with mf_ref's.

    Degenerate orbitals have no canonical basis: an SCF returns an arbitrary
    rotation within each degenerate manifold, chosen independently at every
    geometry.  E_MHS depends on that choice -- HS and MHS are defined by the
    occupancy of *individual* spatial orbitals -- so an unaligned scan samples
    a different subspace at every point and the MHS curve is noise on top of
    physics.

    Within each block of orbitals sharing an eigenvalue (to deg_tol) the
    rotation is free: it leaves the SCF energy, the density and the total
    space exactly invariant.  We choose the rotation that best matches the
    previous geometry, by orthogonal Procrustes on the cross-geometry overlap

        Omega = C_ref^T S_cross C_block  ->  U = V W^T from the SVD of Omega,

    which maximises tr(U^T Omega), i.e. the summed overlap with the reference.
    Orbitals outside any degenerate block are left alone apart from a sign
    flip, which is also free and which keeps the coefficients continuous.

    Mutates and returns mf.  No-op if mf_ref is None.
    """
    if mf_ref is None:
        return mf
    S = gto.intor_cross('int1e_ovlp', mol_ref, mol)
    C = np.array(mf.mo_coeff, copy=True) 
    E = np.asarray(mf.mo_energy)
    spins = range(C.shape[0]) if C.ndim == 3 else [None]
    Cr = np.asarray(mf_ref.mo_coeff)

    for sp in spins:
        Ci = C[sp] if sp is not None else C
        Ei = E[sp] if sp is not None else E
        Cri = Cr[sp] if (sp is not None and Cr.ndim == 3) else (
            Cr[0] if Cr.ndim == 3 else Cr)
        Omega = Cri.T @ S @ Ci                      # (nmo_ref, nmo)

        start = 0
        for k in range(1, len(Ei) + 1):
            if k == len(Ei) or abs(Ei[k] - Ei[start]) > deg_tol:
                blk = slice(start, k)
                if k - start > 1:                   # degenerate manifold
                    Om = Omega[blk, blk]
                    U, _, Wt = np.linalg.svd(Om)
                    Ci[:, blk] = Ci[:, blk] @ (U @ Wt).T
                else:                               # sign only
                    if Omega[start, start] < 0:
                        Ci[:, start] *= -1.0
                start = k
        if sp is not None:
            C[sp] = Ci
        else:
            C = Ci
    # Rotation within a degenerate block is orthogonal, so orthonormality is
    # preserved exactly.  If it is not, blocks were mis-identified or an
    # already-aligned object was aligned again, and the orbitals are no longer
    # SCF eigenvectors -- fail here rather than let it reach the eigensolver.
    Sm = mol.intor('int1e_ovlp')
    for Ci in ([C[0], C[1]] if C.ndim == 3 else [C]):
        dev = float(np.abs(Ci.T @ Sm @ Ci - np.eye(Ci.shape[1])).max())
        assert dev < 1e-9, f'align_gauge broke orthonormality: {dev:.2e}'
    out = mf.copy()
    out.mo_coeff = C
    return out

def _spikes(v, window=7, z=8.0, floor=0.0):
    """Indices where |v[i]| departs from its own neighbourhood.

    A global threshold cannot work on a PES: |diff(e,2)| is legitimately large
    on the repulsive wall and tiny at dissociation, so any single cutoff --
    absolute or median-relative -- either flags the wall or misses a jump on
    the flat tail.  This compares each point to a robust median/MAD of the
    surrounding window *excluding itself*, so a smoothly varying wall is not
    flagged (its neighbours are equally large) while an isolated departure is.

    Limitation: sensitivity is set by the local curvature scale, so a jump
    smaller than the local second difference -- e.g. 5 mHa on a steep wall --
    is not detectable by any test of this kind.
    """
    v = np.abs(np.asarray(v, float)); n = v.size
    if n < 5:
        return np.array([], int), 0.0
    hits, worst, h = [], 0.0, window // 2
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        nb = np.delete(v[lo:hi], i - lo)
        med = float(np.median(nb))
        mad = float(np.median(np.abs(nb - med))) * 1.4826
        scale = mad if mad > 1e-15 else max(med, 1e-15)
        score = (v[i] - med) / scale
        if v[i] > floor and score > z:
            hits.append(i); worst = max(worst, score)
    return np.array(hits, int), worst


# ----------------------------------------------------------- PHASE 1 ------
def scan_reference(name, geometry, scan_range, spin=0, charge=0,
                   basis_set='sto-3g', outfile=None, verbose=True,
                   bidirectional=True, align=True):
    """SCF + FCI at every geometry.  Returns a list of per-point records.

    `geometry` is a format string, e.g. 'N 0 0 0; N 0 0 {}'.

    bidirectional -- sweep outward as well as inward and keep the lower
      converged solution at each geometry.  A one-way continuation can carry
      a higher branch a long way before a fresh guess breaks out of it, which
      shows up as a step in the energy with a simultaneous jump in <S^2>.
    align -- rotate each geometry's degenerate orbitals into alignment with
      its neighbour's (see align_gauge).  Leaves every SCF quantity invariant
      but makes E_MHS a continuous function of the geometry.
    """
    def sweep(pts, prev_mf, prev_mol):
        """One pass; returns {distance: (mol, {mode: (mf, info)})}."""
        res, dm_prev = {}, {'rhf': None, 'uhf': None}
        for d in pts:
            mol = gto.M(atom=geometry.format(d), basis=basis_set,
                        spin=spin, charge=charge, verbose=0)
            per = {}
            for mode in ('rhf', 'uhf'):
                mf, info = build_reference(mol, mode, dm_prev=dm_prev[mode])
                dm_prev[mode] = mf.make_rdm1()
                prev_mf[mode], prev_mol[mode] = mf, mol
                per[mode] = (mf, info)
            res[d] = (mol, per)
        return res

    pts_in = sorted(scan_range, reverse=True)
    if verbose:
        print(f'sweep 1/{2 if bidirectional else 1}: inward', flush=True)
    A = sweep(pts_in, {}, {})
    if bidirectional:
        if verbose:
            print('sweep 2/2: outward', flush=True)
        B = sweep(list(reversed(pts_in)), {}, {})
    else:
        B = {}

    out, prev_mf, prev_mol = [], {}, {}
    fci_prev, d_prev, turned = None, None, False
    for d in sorted(scan_range):
        mol, per = A[d]
        if d in B:                       # keep the lower converged solution
            molB, perB = B[d]
            merged = {}
            for mode in ('rhf', 'uhf'):
                a, b = per[mode], perB[mode]
                take_b = (b[1]['converged'] and
                          (not a[1]['converged'] or
                           b[1]['e_tot'] < a[1]['e_tot'] - 1e-10))
                merged[mode] = b if take_b else a
                merged[mode][1]['sweep'] = 'outward' if take_b else 'inward'
            mol, per = (molB if merged['rhf'] is perB['rhf'] else mol), merged

        rec = {'name': name, 'distance': float(d), 'M': int(mol.nao),
               'N': int(mol.nelectron), 'spin': int(mol.spin),
               'charge': int(mol.charge), 'basis_set': basis_set, 'refs': {}}
        # FCI is computed from the *unaligned* reference.  The CI here is full
        # -- no frozen core, no active space -- so for two bases of the same
        # orbital space related by U in U(M), the induced map on Lambda^N V is
        # the unitary Lambda^N U and the spectrum is identical.  Alignment
        # therefore cannot move the eigenvalue; it can only move the Davidson
        # start vector, and an unlucky start either fails to converge or
        # converges onto an excited root.  Measured at R = 2.44 A: the aligned
        # basis returned -107.43921618, which is not an eigenvalue at all
        # (roots 0 and 1 are -107.44090965 and -107.43558482) but a
        # half-converged Rayleigh quotient sitting between them.
        cis = fci.FCI(mol, np.asarray(per['rhf'][0].mo_coeff))
        cis.conv_tol = 1e-12
        cis.max_space = FCI_MAX_SPACE
        cis.max_cycle = FCI_MAX_CYCLE
        e_fci = float(np.atleast_1d(cis.kernel()[0])[0])
        assert np.atleast_1d(cis.converged).all(), (
            name, d, f'FCI Davidson did not converge (E={e_fci:.8f})')
        rec['e_fci'] = e_fci

        for mode in ('rhf', 'uhf'):
            mf, info = per[mode]
            # Re-align against the *merged* neighbour, since the two sweeps
            # were aligned independently and the winner may come from either.
            if align:
                mf = align_gauge(mf, mol, prev_mf.get(mode), prev_mol.get(mode))
            info['gauge_deficit_prev'] = (
                gauge_deficit(prev_mf[mode], prev_mol[mode], mf, mol)
                if mode in prev_mf else 0.0)
            prev_mf[mode], prev_mol[mode] = mf, mol
            rec['refs'][mode] = {'mf': mf, 'info': info}

        r, u = rec['refs']['rhf']['info'], rec['refs']['uhf']['info']
        assert u['e_tot'] <= r['e_tot'] + 1e-7, (name, d, 'UHF above RHF')
        assert rec['e_fci'] <= u['e_tot'] + TOL, (name, d, 'FCI above UHF')

        # cis.converged is necessary but not sufficient: a rotated basis can
        # converge cleanly onto an excited root.  Measured at R = 2.58 A,
        # -107.43889572 was flagged converged against -107.43989652 correct.
        # Past the minimum the exact curve is nondecreasing, so a decrease
        # bounds the error at the *earlier* point from below by its own size:
        #     Ehat(R1) - E(R1) >= Ehat(R1) - Ehat(R2)   for R1 < R2,
        # using only Ehat >= E pointwise and E(R1) <= E(R2).
        if MONOTONE_TAIL and fci_prev is not None:
            if rec['e_fci'] > fci_prev + 1e-9:
                turned = True
            elif turned and rec['e_fci'] < fci_prev - 1e-9:
                raise AssertionError(
                    f'{name} R={d}: FCI decreased {fci_prev:.8f} -> '
                    f'{rec["e_fci"]:.8f} past the minimum; the value at '
                    f'R={d_prev} is too high by at least '
                    f'{(fci_prev - rec["e_fci"]) * 1e3:.3f} mHa')
        fci_prev, d_prev = rec['e_fci'], d

        if verbose:
            print(f"{d:6.3f}  RHF {r['e_tot']:14.8f} S2={r['spin_sq']:6.4f} "
                  f"stab={str(r['stable_internal']):<5} "
                  f"gap={r['gauge_deficit_prev']:.1e} | "
                  f"UHF {u['e_tot']:14.8f} S2={u['spin_sq']:6.4f} "
                  f"stab={str(u['stable_internal']):<5} "
                  f"gap={u['gauge_deficit_prev']:.1e} | "
                  f"FCI {rec['e_fci']:14.8f}", flush=True)
        out.append(rec)

    if outfile:
        with open(outfile, 'wb') as f:
            pickle.dump(out, f)
        print(f'phase 1 -> {outfile} ({len(out)} points)')
    return out


def check_reference(recs, kink_z=8.0, gauge_z=8.0):
    """Objective version of 'does the PySCF curve look fine'.

    Every test is an outlier test against the curve's own local trend, so the
    result does not depend on the grid spacing.
    """
    d = np.array([r['distance'] for r in recs])
    fmt = lambda xs: ', '.join(f'{x:.3f}' for x in xs)
    problems = []
    for mode in ('rhf', 'uhf'):
        e = np.array([r['refs'][mode]['info']['e_tot'] for r in recs])
        s2 = np.array([r['refs'][mode]['info']['spin_sq'] for r in recs])
        g = np.array([r['refs'][mode]['info']['gauge_deficit_prev'] for r in recs])
        conv = [r['refs'][mode]['info']['converged'] for r in recs]
        stab = [r['refs'][mode]['info']['stable_internal'] for r in recs]

        if not all(conv):
            problems.append(f'{mode}: {conv.count(False)} unconverged points')
        if not all(stab):
            problems.append(f'{mode}: not internally stable at R = '
                            f'{fmt([x for x, ok in zip(d, stab) if not ok])}')
        if len(e) > 3:
            idx, zz = _spikes(np.diff(e, 2), z=kink_z)
            if idx.size:
                problems.append(f'{mode}: energy kink at R = {fmt(d[1:-1][idx])} '
                                f'(local z = {zz:.0f})')
        if len(s2) > 1:
            ks = np.abs(np.diff(s2))
            if ks.max() > 0.5:
                problems.append(f'{mode}: <S2> jumps by {ks.max():.2f} at '
                                f'R = {d[1:][ks.argmax()]:.3f}')
        if len(g) > 3:
            idx, zz = _spikes(g[1:], z=gauge_z, floor=1e-2)
            if idx.size:
                problems.append(f'{mode}: orbital gauge rotates at R = '
                                f'{fmt(d[1:][idx])} (local z = {zz:.0f})')
    if len(recs) > 3:
        idx, zz = _spikes(np.diff([r['e_fci'] for r in recs], 2), z=kink_z)
        if idx.size:
            problems.append(f'FCI: kink at R = {fmt(d[1:-1][idx])} -- root flipping')
    return problems


def plot_reference(recs, title=None, savefig=None):
    import matplotlib.pyplot as plt
    d = [r['distance'] for r in recs]
    get = lambda m, k: [r['refs'][m]['info'][k] for r in recs]

    fig, ax = plt.subplots(3, 1, figsize=(9, 11), sharex=True,
                           gridspec_kw={'height_ratios': [3, 2, 2]})
    ax[0].plot(d, get('rhf', 'e_tot'), 'b-o', ms=4, label='RHF/ROHF')
    ax[0].plot(d, get('uhf', 'e_tot'), 'm--s', ms=4, mfc='none', label='UHF')
    ax[0].plot(d, [r['e_fci'] for r in recs], 'g-.', lw=1.5, label='FCI')
    ax[0].set_ylabel('Energy (Ha)'); ax[0].legend(); ax[0].grid(alpha=.3)
    if title: ax[0].set_title(title)

    ax[1].plot(d, get('rhf', 'spin_sq'), 'b-o', ms=4, label='RHF/ROHF')
    ax[1].plot(d, get('uhf', 'spin_sq'), 'm--s', ms=4, mfc='none', label='UHF')
    ax[1].set_ylabel(r'$\langle S^2\rangle$'); ax[1].legend(); ax[1].grid(alpha=.3)

    for m, c in (('rhf', 'b'), ('uhf', 'm')):
        v = np.array(get(m, 'gauge_deficit_prev'))
        ax[2].semilogy(d, np.maximum(v, 1e-18), c + '-o', ms=3, label=m.upper())
    ax[2].axhline(1e-2, color='r', ls=':', lw=1)
    ax[2].set_ylabel('orbital gauge deficit\nvs previous geometry')
    ax[2].set_xlabel(r'Bond length ($\AA$)'); ax[2].legend(); ax[2].grid(alpha=.3)

    plt.tight_layout()
    if savefig: plt.savefig(savefig, dpi=150)
    plt.show()


# ----------------------------------------------------------- PHASE 2 ------
def scan_subspaces(recs, outfile=None, verbose=True):
    """Subspace energies at every geometry, reusing the stored references."""
    rows = []
    for rec in recs:
        d, M, N = rec['distance'], rec['M'], rec['N']
        nq, omegas = 2 * M, feasible_omega(M, N)
        assert rec['spin'] in omegas, (
            f"{rec['name']} R={d}: target omega={rec['spin']} not feasible")
        # Reuse the stored Mole: it carries the right charge, and it is the
        # same object mf.get_hcore() uses, so h1 and h2 cannot disagree.
        mol = rec['refs']['rhf']['mf'].mol
        ovlp = mol.intor('int1e_ovlp')

        for mode in ('rhf', 'uhf'):
            mf = rec['refs'][mode]['mf']
            mo = tuple(mf.mo_coeff) if np.asarray(mf.mo_coeff).ndim == 3 else None
            ov = ovlp if mo is not None else 1
            h1, h2, e_nuc = HamiltonianLoader(nq).from_pyscf(mol, mf)

            def solve(sub, omega, k=1):
                basis = np.asarray(SymmetryBasis(nq, N).build(
                    mode=sub, multiplicity=omega))
                H = build_hamiltonian(h1, h2, basis, show=False)
                dt = str(np.asarray(H.data).dtype)
                assert dt == 'float64', f'{sub} narrowed to {dt}'
                w, V = lowest_states(H, int(basis.size), e_nuc,
                                     k=min(k, basis.size))
                return basis, w, V

            b_p, w_p, V_p = solve('PCS', None, PCS_ROOTS)
            g = int(np.sum(np.abs(w_p - w_p[0]) < DEG_TOL))
            assert g < len(w_p), (rec['name'], d, mode, 'raise PCS_ROOTS')
            Vdeg = V_p[:, :g]

            b_m, w_m, V_m = solve('MS', int(rec['spin']))
            assert abs(w_m[0] - rec['e_fci']) < 1e-10, (
                rec['name'], d, mode, f'MS != FCI by {w_m[0]-rec["e_fci"]:.2e}')
            assert w_p[0] <= rec['e_fci'] + TOL, (rec['name'], d, mode)

            block = {}
            for om in omegas:
                b_h, w_h, V_h = solve('MHS', om)
                assert b_h.size == dim_mhs(M, N, om)
                nD, nE = block_counts(M, N, om)
                block[om] = dict(
                    E=float(w_h[0]), D=int(b_h.size), n_D=nD, n_E=nE,
                    spin_sq=spin_square(V_h[:, 0], b_h, M, mo, ov)[0],
                    fidelity=fidelity_to_eigenspace(V_h[:, 0], b_h, Vdeg, b_p))
            E_hs = min(v['E'] for v in block.values())
            assert block[int(rec['spin'])]['E'] >= E_hs - TOL
            assert E_hs >= w_p[0] - TOL

            base = dict(Molecule=rec['name'], Distance=d, Reference=mode.upper(),
                        M=M, N=N, E_FCI=rec['e_fci'],
                        E_SCF=rec['refs'][mode]['info']['e_tot'],
                        S2_SCF=rec['refs'][mode]['info']['spin_sq'],
                        PCS_Degeneracy=g,
                        Gauge_Deficit=rec['refs'][mode]['info']['gauge_deficit_prev'])
            rows.append(dict(base, Subspace='PCS', Omega=None, Dim=int(b_p.size),
                             Energy=float(w_p[0]), Fidelity=1.0,
                             S2=spin_square(V_p[:, 0], b_p, M, mo, ov)[0],
                             n_D=None, n_E=None))
            rows.append(dict(base, Subspace='MS', Omega=int(rec['spin']),
                             Dim=int(b_m.size), Energy=float(w_m[0]),
                             Fidelity=fidelity_to_eigenspace(V_m[:, 0], b_m,
                                                             Vdeg, b_p),
                             S2=spin_square(V_m[:, 0], b_m, M, mo, ov)[0],
                             n_D=None, n_E=None))
            rows.append(dict(base, Subspace='HS', Omega=None, Dim=None,
                             Energy=E_hs, Fidelity=None, S2=None,
                             n_D=None, n_E=None))
            for om, v in block.items():
                rows.append(dict(base, Subspace=f'MHS(w={om})', Omega=om,
                                 Dim=v['D'], Energy=v['E'],
                                 Fidelity=v['fidelity'], S2=v['spin_sq'],
                                 n_D=v['n_D'], n_E=v['n_E']))
            if verbose:
                print(f"  {d:6.3f} {mode.upper()}  PCS {w_p[0]:13.8f} (g={g})  "
                      f"HS {E_hs:13.8f}  "
                      f"MHS {block[int(rec['spin'])]['E']:13.8f}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    if outfile:
        df.to_csv(outfile, index=False)
        print(f'phase 2 -> {outfile} ({len(df)} rows)')
    return df