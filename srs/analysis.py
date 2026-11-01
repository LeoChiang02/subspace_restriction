"""
analysis.py -- spin expectation and degeneracy-aware fidelity for
determinant-basis subspace vectors.

Encoding convention (matches SymmetryBasis): bit 2p is spatial orbital p with
alpha spin, bit 2p+1 is the same orbital with beta spin.

Both routines embed the subspace vector into the full FCI vector of its own
(Na, Nb) sector and then call PySCF's tested primitives.  Nothing here
reimplements S^2.
"""

import numpy as np
from pyscf.fci import cistring, spin_op


def split_masks(states, M):
    """Return (alpha_mask, beta_mask, Na, Nb) arrays for a bit-encoded basis."""
    s = np.array([int(x) for x in np.asarray(states)])
    a = np.zeros(len(s), dtype=np.int64)
    b = np.zeros(len(s), dtype=np.int64)
    for p in range(M):
        a |= (((s >> (2*p)) & 1) << p)
        b |= (((s >> (2*p + 1)) & 1) << p)
    na = np.array([bin(int(x)).count('1') for x in a])
    nb = np.array([bin(int(x)).count('1') for x in b])
    return a, b, na, nb


def reorder_sign(amask, bmask, M):
    """Sign converting interleaved operator order to PySCF's blocked order.

    SymmetryBasis orders creation operators by bit index -- (0a, 0b, 1a, 1b,
    ...) -- while PySCF's FCI addressing assumes all alpha operators precede
    all beta.  Moving each beta_q past every alpha_p with p > q costs one
    transposition each, so

        sign = (-1)^{ sum_{q in beta-occ} #{p in alpha-occ : p > q} }.

    Energies are blind to this (a diagonal +/-1 similarity, and H is assembled
    in the same convention it is diagonalised in), but any operator borrowed
    from PySCF must see the blocked convention.
    """
    n = 0
    for q in range(M):
        if (bmask >> q) & 1:
            n += bin(amask >> (q + 1)).count('1')
    return -1.0 if n % 2 else 1.0


def to_fcivec(vec, states, M):
    """Embed a subspace vector into full FCI vectors, one per (Na, Nb) sector.

    Returns {(Na, Nb): civec} with civec of shape (C(M,Na), C(M,Nb)) in
    PySCF's string addressing.  The interleaved -> blocked reordering sign is
    applied here, so callers pass vectors in the SymmetryBasis convention.
    """
    a, b, na, nb = split_masks(states, M)
    sg = np.array([reorder_sign(int(a[i]), int(b[i]), M) for i in range(len(a))])
    vec = np.asarray(vec) * sg
    out = {}
    for sector in sorted(set(zip(na.tolist(), nb.tolist()))):
        Na, Nb = sector
        sel = (na == Na) & (nb == Nb)
        ci = np.zeros((cistring.num_strings(M, Na),
                       cistring.num_strings(M, Nb)))
        for idx in np.flatnonzero(sel):
            ci[cistring.str2addr(M, Na, int(a[idx])),
               cistring.str2addr(M, Nb, int(b[idx]))] = vec[idx]
        out[sector] = ci
    return out


def spin_square(vec, states, M, mo_coeff=None, ovlp=1):
    """<S^2> and multiplicity of a determinant-basis vector.

    mo_coeff -- (C_alpha, C_beta) for a UHF reference, None for RHF/ROHF.
    ovlp     -- AO overlap matrix; required whenever mo_coeff is a pair.

    A vector spread over several (Na, Nb) sectors is handled by summing the
    sector contributions, which is exact because S^2 commutes with S_z and N,
    so distinct sectors do not mix.
    """
    tot = 0.0
    for (Na, Nb), ci in to_fcivec(vec, states, M).items():
        w = float(np.vdot(ci, ci))
        if w < 1e-14:
            continue
        ss, _ = spin_op.spin_square(ci / np.sqrt(w), M, (Na, Nb),
                                    mo_coeff=mo_coeff, ovlp=ovlp)
        tot += w * ss
    s = 0.5 * (np.sqrt(1 + 4*tot) - 1)
    return float(tot), float(2*s + 1)


def fidelity_to_eigenspace(vec_sub, states_sub, vecs_ref, states_ref):
    """Projection of a subspace vector onto a (possibly degenerate) eigenspace.

        F = || P_deg psi_sub ||^2 = sum_k |<psi_sub | psi_ref^(k)>|^2

    Invariant under any unitary mixing inside the degenerate multiplet, and
    equal to the plain overlap squared when the eigenspace is one-dimensional.
    vecs_ref has shape (D_ref, g) with orthonormal columns.
    """
    idx = {int(s): i for i, s in enumerate(np.asarray(states_ref))}
    rows, coeff = [], []
    for i, s in enumerate(np.asarray(states_sub)):
        j = idx.get(int(s))
        if j is not None:
            rows.append(j)
            coeff.append(vec_sub[i])
    if not rows:
        return 0.0
    V = np.atleast_2d(np.asarray(vecs_ref))
    if V.shape[0] != len(idx):
        V = V.T
    return float(np.sum(np.abs(np.asarray(coeff) @ V[rows, :])**2))