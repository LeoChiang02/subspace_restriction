"""
stream_encode -- Pauli 1-norm, term count and XBM(half) circuit count of a
padded real symmetric matrix, computed pattern by pattern from the sparse
matrix.  Nothing is stored per term unless export= is given.

Conventions (same as n_encod.py and Encode):
  * q = ceil(log2 D); embedding offset s = floor((2^q - D)/2).
  * Padded matrix A: H on [s, s+D), lam on the padding diagonal, 0 elsewhere.
    This equals embed_as_block(H - lam*I) + lam*I as built in Encode.
  * Pauli label position 0 = most significant bit of the row index.
  * Pattern a = row XOR col;  g_a(r) = A[r, r XOR a].
  * c_{a,b} = 2^{-q} (-1)^{|a&b|/2} WHT[g_a](b) if |a&b| even, else 0.
  * N_XBM = |S(A)|, S(A) = {row XOR col : |A_rc| > s_tol}  (XBM half,
    Kondo et al. Quantum 6, 688 (2022), Sec. 4 and Eq. (2)).
  * Z = sum_a max_r |g_a(r)|  (their App. B, Eq. S.56, real symmetric case).



Usage, where H is the SciPy CSR matrix from SRS_ray after the blocked relabelling:
res = stream_encode(H, pad='mindiag')                    # numbers only, nothing stored
res = stream_encode(H, pad='zero', export='LiH_MHS_omega0.txt')   # also writes strings
res = stream_encode(H22, pad='mindiag', n_workers=30)    # H22: never export

"""
import math
import numpy as np
import scipy.sparse as sp

_CHARS = np.frombuffer(b'IZXY', dtype=np.uint8)      # code = 2*a_k + b_k
_SH = {}                                            # shared state for workers


def _fwht_inplace(x):
    """y[b] = sum_r x[r] (-1)^{popcount(r & b)}, vectorised, in place."""
    n, h = x.shape[0], 1
    while h < n:
        y = x.reshape(-1, 2, h)
        t = y[:, 0, :].copy()
        y[:, 0, :] += y[:, 1, :]
        y[:, 1, :] = t - y[:, 1, :]
        h *= 2
    return x


def _popcount_table(q):
    idx = np.arange(1 << q, dtype=np.int64)
    pc = np.zeros(1 << q, dtype=np.uint8)
    for k in range(q):
        pc += ((idx >> k) & 1).astype(np.uint8)
    return pc


def _pattern(k, want_terms=False):
    """Coefficient statistics of pattern number k (index into _SH['ua'])."""
    S = _SH
    a = int(S['ua'][k]); lo, hi = S['start'][k], S['end'][k]
    N, tol = S['N'], S['tol']
    g = np.zeros(N)
    g[S['r'][lo:hi]] = S['v'][lo:hi]
    _fwht_inplace(g)
    pc = S['POP'][a & S['B']]
    even = (pc & 1) == 0
    c = g[even] * ((1.0 - 2.0 * ((pc[even] >> 1) & 1)) / N)
    absc = np.abs(c)
    keep = absc > tol
    small = (absc != 0.0) & ~keep
    out = dict(L=float(np.sum(absc)), T=int(np.count_nonzero(keep)),
               dn=int(np.count_nonzero(small)), dL=float(np.sum(absc[small])),
               cI=float(c[0]) if a == 0 else None)      # b = 0 is first even b
    if want_terms:
        out['a'] = a
        out['b'] = S['B'][even][keep]
        out['c'] = c[keep]
    return out


def _work(ks):
    return [_pattern(k) for k in ks]


def _write_terms(f, a, bs, cs, q):
    if len(bs) == 0:
        return
    sh = np.arange(q - 1, -1, -1)
    codes = 2 * ((a >> sh) & 1)[None, :] + ((bs[:, None] >> sh[None, :]) & 1)
    labels = _CHARS[codes].view(f'S{q}').ravel()
    f.writelines(f"{lab.decode()}: {val:>12.7f}\n" for lab, val in zip(labels, cs))


def _spectrum_ends(H):
    """(E0, Emax) of H.  Dense for small D, Lanczos otherwise."""
    D = H.shape[0]
    if D <= 4096:
        w = np.linalg.eigvalsh(H.toarray())
        return float(w[0]), float(w[-1])
    from scipy.sparse.linalg import eigsh
    e0 = eigsh(H, k=1, which='SA', tol=0)[0][0]
    em = eigsh(H, k=1, which='LA', tol=0)[0][0]
    return float(e0), float(em)


def stream_encode(H, pad='zero', tol=1e-12, s_tol=1e-12, export=None,
                  spectrum=True, n_workers=30, sym_tol=1e-10, report=True):
    """
    H        : (D, D) real symmetric scipy sparse matrix (electronic block).
    pad      : 'zero', 'mindiag', or a float lam (padding diagonal value).
               'mindiag' uses lam = min_j H_jj with the same guard as Encode:
               if lam - E0 < 1e-9 then lam = E0 + 1e-6.
    tol      : |c_P| > tol counts towards T (as in NorthStar); the rest is
               reported as dropped.  L_tot sums ALL coefficients, no cut.
    s_tol    : entry threshold defining S(A) for the XBM count.
    export   : None (accumulate only) or a .txt path (terms streamed to disk,
               same line format as n_encod.to_txt; requires n_workers=1).
    spectrum : compute E0, Emax and floor = (Emax - E0)/2.
    """
    H = sp.coo_matrix(H, dtype=np.float64)
    H.sum_duplicates()
    D = H.shape[0]
    if H.shape[0] != H.shape[1]:
        raise ValueError(f"expected square matrix, got {H.shape}")
    asym = float(abs(H.tocsr() - H.tocsr().T).max()) if H.nnz else 0.0
    if asym > sym_tol:
        raise ValueError(f"not symmetric: max|H - H^T| = {asym:.3e}")
    if D < 2:                                                     # replaces the raise
        h = float(H.toarray()[0, 0])
        if export is not None:
            with open(export, 'w') as f:
                f.write(f": {h:>12.7f}\n")                        # old to_txt with label ''
        res = dict(D=1, q=0, s=0, lam=0.0, T=int(abs(h) > tol), L_tot=abs(h),
                   c_I=h, L_noI=0.0, N_XBM=int(abs(h) > s_tol), n_patterns_offdiag=0,
                   n_tiny_patterns=0, Z=abs(h), dropped_n=0, dropped_L=0.0,
                   E0=h, Emax=h, floor=0.0, max_asym=0.0, file=export)
        if report:
            print(f"stream_encode: D=1 q=0 | T={res['T']} c_I={h:+.10f} (1x1 block)")
        return res
    if export is not None and n_workers != 1:
        raise ValueError("export requires n_workers=1 (terms are written in order)")

    E0 = Emax = None
    if spectrum or pad == 'mindiag':
        E0, Emax = _spectrum_ends(H.tocsr())
    if pad == 'zero':
        lam = 0.0
    elif pad == 'mindiag':
        lam = float(H.diagonal().min())
        assert lam >= E0 - 1e-10, "Rayleigh-quotient guard failed"
        if lam - E0 < 1e-9:
            lam = E0 + 1e-6
    else:
        lam = float(pad)

    q = int(np.ceil(np.log2(D)))
    N = 1 << q
    s = (N - D) // 2

    # ---- group entries of the padded matrix by pattern a = row ^ col ----
    rows = H.row.astype(np.int64) + s
    cols = H.col.astype(np.int64) + s
    vals = H.data
    if lam != 0.0 and N > D:
        padidx = np.r_[np.arange(0, s), np.arange(s + D, N)].astype(np.int64)
        rows = np.r_[rows, padidx]; cols = np.r_[cols, padidx]
        vals = np.r_[vals, np.full(padidx.size, lam)]
    pat = rows ^ cols
    order = np.argsort(pat, kind='stable')
    pat, r, v = pat[order], rows[order], vals[order]
    ua, start = np.unique(pat, return_index=True)
    end = np.r_[start[1:], pat.size]
    maxabs = np.maximum.reduceat(np.abs(v), start)

    in_S = maxabs > s_tol
    N_XBM = int(np.count_nonzero(in_S))
    n_offdiag = int(np.count_nonzero(in_S & (ua != 0)))      # |Delta|
    n_tiny = int(np.count_nonzero(~in_S & (maxabs > 0)))
    Z = math.fsum(maxabs)

    _SH.clear()
    _SH.update(ua=ua, start=start, end=end, r=r, v=v, N=N, tol=tol,
               POP=_popcount_table(q), B=np.arange(N, dtype=np.int64))
    work = [k for k in range(ua.size) if maxabs[k] > 0.0]

    # ---- stream over patterns ----
    Ls, dLs = [], []
    T = dn = 0
    c_I = 0.0
    if export is not None:
        with open(export, 'w') as f:
            for k in work:
                st = _pattern(k, want_terms=True)
                _write_terms(f, st['a'], st['b'], st['c'], q)
                Ls.append(st['L']); dLs.append(st['dL'])
                T += st['T']; dn += st['dn']
                if st['cI'] is not None: c_I = st['cI']
    else:
        if n_workers == 1:
            stats = [_pattern(k) for k in work]
        else:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            chunks = [work[i::n_workers * 8] for i in range(n_workers * 8)]
            with ProcessPoolExecutor(n_workers, mp_context=mp.get_context('fork')) as ex:
                stats = [st for part in ex.map(_work, chunks) for st in part]
        for st in stats:
            Ls.append(st['L']); dLs.append(st['dL'])
            T += st['T']; dn += st['dn']
            if st['cI'] is not None: c_I = st['cI']
    _SH.clear()

    L_tot = math.fsum(Ls)
    res = dict(D=D, q=q, s=s, lam=lam, T=T, L_tot=L_tot, c_I=c_I,
               L_noI=L_tot - abs(c_I), N_XBM=N_XBM, n_patterns_offdiag=n_offdiag,
               n_tiny_patterns=n_tiny, Z=Z, dropped_n=dn, dropped_L=math.fsum(dLs),
               E0=E0, Emax=Emax,
               floor=None if E0 is None else (Emax - E0) / 2,
               max_asym=asym, file=export)
    if report:
        print(f"stream_encode: D={D} q={q} s={s} lam={lam:+.8f} | T={T} "
              f"L_tot={L_tot:.10f} c_I={c_I:+.10f} L_noI={res['L_noI']:.10f} | "
              f"N_XBM={N_XBM} |Delta|={n_offdiag} Z={Z:.8f} | dropped={dn} "
              f"({res['dropped_L']:.2e})")
    return res