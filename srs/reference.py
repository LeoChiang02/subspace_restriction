"""
reference.py -- SCF reference construction for the subspace-restriction rerun.

One converged, stability-certified mean-field object per (molecule, mode),
plus the diagnostics needed to interpret it.

  * `mol` is never mutated. The caller owns it; nothing here writes to it.
  * The spin sector is fixed at mol.spin. No search over sectors.
  * Only *converged* solutions are eligible.
  * Internal instabilities are followed to a certificate.

For spin != 0, scf.RHF(mol) dispatches to ROHF; that is intended, and
mode='rhf' means "spin-restricted" throughout.
"""

import numpy as np
import pyscf
from pyscf import gto, scf

__all__ = ['init_guess_mixed', 'build_reference']


def init_guess_mixed(mol, mixing_parameter=np.pi/4, rhf=None):
    """Broken-symmetry UHF guess by HOMO-LUMO mixing.

    Ansatz of Szabo & Ostlund Sec. 3.8.7, as in PySCF example
    56-h2_symm_breaking.py, with three corrections to that example:
      (1) the frontier pair is located from the occupation count, not by a
          test on mo_occ that never fires;
      (2) columns are indexed over MOs, not AOs;
      (3) mixing is applied to converged RHF orbitals, not to
          core-Hamiltonian eigenvectors.
    Per-spin occupations are passed as occ/2 so that Tr(D S) = N, not 2N.

    Returns (dm, homo, lumo). Only meaningful for mol.spin == 0.
    """
    if rhf is None:
        rhf = scf.RHF(mol).run()
    if not rhf.converged:
        raise RuntimeError('RHF must converge before mixing')

    C, occ = rhf.mo_coeff, rhf.mo_occ
    nocc = int((occ > 0).sum())
    homo, lumo = nocc - 1, nocc
    if homo < 0 or lumo >= C.shape[1]:
        raise RuntimeError(f'no frontier pair: nocc={nocc}, nmo={C.shape[1]}')

    q = mixing_parameter
    ph, pl = C[:, homo], C[:, lumo]
    Ca, Cb = C.copy(), C.copy()
    Ca[:, homo] = np.cos(q)*ph + np.sin(q)*pl
    Cb[:, homo] = np.cos(q)*ph - np.sin(q)*pl
    Ca[:, lumo] = -np.sin(q)*ph + np.cos(q)*pl
    Cb[:, lumo] = np.sin(q)*ph + np.cos(q)*pl

    o = occ / 2.0
    return scf.UHF(mol).make_rdm1((Ca, Cb), (o, o)), homo, lumo


def _follow_internal(mol, mf, rerun, max_follow=5):
    """Follow internal instabilities until stable or the budget runs out.

    `rerun(dm)` returns a mean-field object of the same kind as `mf`.
    Returns (mf, stable_internal, n_follow).
    """
    for n in range(max_follow):
        out = mf.stability(return_status=True)
        mo_i, stable_i = out[0], out[-2]
        if stable_i:
            return mf, True, n
        mf_new = rerun(mf.make_rdm1(mo_i, mf.mo_occ))
        if not mf_new.converged:
            return mf, False, n          # keep the last converged solution
        if mf_new.e_tot > mf.e_tot - 1e-10:
            # following stopped lowering it -- keep whichever is lower
            return (mf_new if mf_new.e_tot < mf.e_tot else mf), False, n + 1
        mf = mf_new
    return mf, False, max_follow


def _external_status(mf):
    """Report external stability without following it. None if unavailable."""
    try:
        out = mf.stability(internal=False, external=True, return_status=True)
        return bool(out[-1])
    except Exception:
        return None


def build_reference(mol, mode, dm_prev=None, max_follow=5,
                    mixing_parameter=np.pi/4, verbose=False):
    """Return (mf, info) for mode in {'rhf', 'uhf'}.

    dm_prev -- optional density from a neighbouring geometry, added as an extra
    trial for PES scans.  Selection is still lowest-converged, so continuation
    can only help: a carried density that finds a worse branch simply loses.

    Raises RuntimeError if no converged solution is found.
    """
    mode = mode.lower()
    if mode not in ('rhf', 'uhf'):
        raise ValueError(f"mode must be 'rhf' or 'uhf', got {mode!r}")

    info = {'mode': mode, 'spin': int(mol.spin), 'nelec': tuple(mol.nelec),
            'nao': int(mol.nao), 'basis': mol.basis, 'guess': None,
            'trials': [], 'n_follow': 0,
            'pyscf_version': pyscf.__version__}

    # dm_prev continues the previous geometry's solution.  It is applied to
    # mf_r only in 'rhf' mode: in 'uhf' mode dm_prev is a spin-resolved UHF
    # density and mf_r is a (RO)RHF object, which cannot consume it.
    _dm0 = dm_prev if (mode == 'rhf' and dm_prev is not None) else None
    mf_r = scf.RHF(mol).run(_dm0) if _dm0 is not None else scf.RHF(mol).run()
    if not mf_r.converged:
        raise RuntimeError(f'RHF/ROHF did not converge for {mol.atom}')
    info['e_rhf'] = float(mf_r.e_tot)

    if mode == 'rhf':
        mf, stable, nf = _follow_internal(
            mol, mf_r, lambda dm: scf.RHF(mol).run(dm), max_follow)
        info.update(guess='default', n_follow=nf, stable_internal=stable,
                    stable_external=_external_status(mf))
        info['trials'] = [('default', float(mf_r.e_tot), True, None)]

    else:
        dm_r = mf_r.make_rdm1()
        trials = [('default', lambda: scf.UHF(mol).run())]
        if dm_prev is not None:
            trials.append(('prev', lambda: scf.UHF(mol).run(dm_prev)))
        if mol.spin == 0:
            try:
                dm_m, h, l = init_guess_mixed(mol, mixing_parameter, rhf=mf_r)
                trials.append(('mixed', lambda: scf.UHF(mol).run(dm_m)))
                info['frontier'] = (h, l)
            except RuntimeError as e:
                info['mixed_unavailable'] = str(e)
        else:
            trials.append(('rohf', lambda: scf.UHF(mol).run(dm_r)))
        # Second-order solver last: slower, but converges cases first-order
        # DIIS cannot, and it belongs in the ladder rather than as a fallback
        # -- a first-order run can converge to a higher stationary point.
        trials.append(('newton',   lambda: scf.UHF(mol).newton().run()))
        trials.append(('newton_r', lambda: scf.UHF(mol).newton().run(dm_r)))

        best, best_name = None, None
        for name, fn in trials:
            try:
                m = fn()
            except Exception as e:
                info['trials'].append((name, None, False, type(e).__name__))
                continue
            info['trials'].append((name, float(m.e_tot), bool(m.converged), None))
            if verbose:
                print(f'  [{name:<9}] conv={m.converged} E={m.e_tot:.10f}')
            if not m.converged:
                continue
            if best is None or m.e_tot < best.e_tot - 1e-10:
                best, best_name = m, name

        if best is None:
            raise RuntimeError(f'no converged UHF for {mol.atom}')

        mf, stable, nf = _follow_internal(
            mol, best, lambda dm: scf.UHF(mol).newton().run(dm), max_follow)
        info.update(guess=best_name, n_follow=nf, stable_internal=stable,
                    stable_external=_external_status(mf),
                    e_before_follow=float(best.e_tot))

    if not mf.converged:
        raise RuntimeError(f'{mode} did not converge for {mol.atom}')

    # A second-order (newton) object is not picklable; undo_soscf returns an
    # ordinary UHF/RHF carrying the same converged orbitals, so the archived
    # object can be reloaded.
    if hasattr(mf, 'undo_soscf'):
        mf = mf.undo_soscf()
    # rhf-mode following uses first-order scf.RHF, so n_follow does not imply
    # a second-order solve; only the newton trials and uhf following do.
    info['second_order'] = str(info['guess']).startswith('newton') or (
        mode == 'uhf' and info['n_follow'] > 0)

    occ = np.asarray(mf.mo_occ)
    if occ.ndim == 2:
        na, nb = int(round(occ[0].sum())), int(round(occ[1].sum()))
    else:
        nb = int(round((occ > 1).sum()))
        na = int(round(occ.sum())) - nb
    info.update(e_tot=float(mf.e_tot),
                spin_sq=float(mf.spin_square()[0]),
                mult=float(mf.spin_square()[1]),
                occ_nelec=(na, nb),
                dE_vs_rhf=float(mf.e_tot - info['e_rhf']),
                converged=bool(mf.converged))

    assert na + nb == mol.nelectron, (na, nb, mol.nelectron)
    assert na - nb == mol.spin, (na, nb, mol.spin)
    assert info['dE_vs_rhf'] <= 1e-7, (
        f"{mode} above RHF by {info['dE_vs_rhf']:.3e} Ha -- wrong "
        f"stationary point")
    return mf, info