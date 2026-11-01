import psutil, ray, gc, os, time, jax

# 1. Suppress Ray "Accelerator Visible Devices" Warning
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
import warnings
# Filter out this specific warning about os.fork
warnings.filterwarnings("ignore", message="os.fork", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=r"os.fork\(\) was called", category=RuntimeWarning)

import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
from itertools import product, combinations
from collections import deque
try:
    from tqdm.notebook import trange, tqdm
except:
    from tqdm import trange

from openfermion import MolecularData, FermionOperator
from openfermion.transforms import get_fermion_operator

from openfermionpyscf import run_pyscf
from pyscf import gto, scf, ao2mo, fci



# Try to import CuPy (GPU Library)
try:
    import cupy as cp
    import cupyx.scipy.sparse.linalg as cp_linalg
    import cupyx.scipy.sparse as cp_sparse
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

jax.config.update('jax_enable_x64', True)
# TO DO: SWITCH SIZE
MEMORY_LIMIT = 10_000_000
USE_GPU_ = 1E4

@ray.remote(runtime_env={"env_vars": {"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "True"}}, max_calls = 1)
def build_chunk_ray(start_idx, end_idx, basis_ref, h1_ref, h2_ref, internal_batch_size, buffer_limit):

    cpu_device = jax.devices('cpu')[0]
    @jax.jit
    def kernel_1body_opt(state, p, q, val):
        occ_q = (state >> q) & 1
        phase_q = jax.lax.population_count(state & ((1 << q) - 1))
        temp = state ^ (1 << q)
        occ_p = (temp >> p) & 1
        phase_p = jax.lax.population_count(temp & ((1 << p) - 1))
        final = temp | (1 << p)
        valid = (occ_q == 1) & (occ_p == 0)
        parity = (phase_q + phase_p) % 2
        amp = ((-1.0)**parity) * valid * val
    
        # Fix
        is_diag = (p == q)
        final = jnp.where(is_diag, state, final)
        amp   = jnp.where(is_diag, val * occ_q, amp)
    
        keep = (final >= state)
        return final, jnp.where(keep, amp, 0.0)

    @jax.jit
    def kernel_2body_opt(state, p, q, r, s, val):
        occ_s = (state >> s) & 1
        phase_s = jax.lax.population_count(state & ((1 << s) - 1))
        temp = state ^ (1 << s)
        occ_r = (temp >> r) & 1
        phase_r = jax.lax.population_count(temp & ((1 << r) - 1))
        temp = temp ^ (1 << r)
        occ_q = (temp >> q) & 1
        phase_q = jax.lax.population_count(temp & ((1 << q) - 1))
        temp = temp | (1 << q)
        occ_p = (temp >> p) & 1
        phase_p = jax.lax.population_count(temp & ((1 << p) - 1))
        final = temp | (1 << p)
        valid = (occ_s == 1) & (occ_r == 1) & (occ_q == 0) & (occ_p == 0)
        parity = (phase_s + phase_r + phase_q + phase_p) % 2
        amp = ((-1.0)**parity) * valid * val
        
        # JIT Filter
        keep = (final >= state)
        return final, jnp.where(keep, amp, 0.0)

    basis_host = basis_ref

    
    if h1_ref:
        H1_P, H1_Q, v1 = h1_ref
    else:
        H1_P, H1_Q, v1 = [jnp.array([], dtype=np.int64)] * 3
    n_terms_h1 = len(v1)

    if h2_ref:
        H2_P, H2_Q, H2_R, H2_S, v2 = h2_ref
    else:
        H2_P, H2_Q, H2_R, H2_S, v2 = [jnp.array([], dtype=np.int64)] * 5
    n_terms_h2 = len(v2)

    # --- D. BUILD LOOP ---
    n_basis_total = len(basis_host)
    local_csr = scipy.sparse.csr_matrix((n_basis_total, n_basis_total), dtype=np.float64)
    
    buf_rows, buf_cols, buf_vals = [], [], []
    current_buffer_count = 0

    with jax.default_device(cpu_device):
        vmap_h1 = jax.vmap(jax.vmap(kernel_1body_opt, in_axes=(None, 0, 0, 0)), in_axes=(0, None, None, None))
        vmap_h2 = jax.vmap(jax.vmap(kernel_2body_opt, in_axes=(None, 0, 0, 0, 0, 0)), in_axes=(0, None, None, None, None, None))

        for i in range(start_idx, end_idx, internal_batch_size):
            b_end = min(i + internal_batch_size, end_idx)
            states = jnp.array(basis_host[i:b_end])
            
            def process_and_buffer(cands, amps, n_terms):
                c_host, a_host = np.array(cands).ravel(), np.array(amps).ravel()
                
                # 1. Primary Filter: JAX already set invalid/lower-tri terms to 0.0
                mask = (a_host != 0)
                if not np.any(mask): return 0

                nz_cands = c_host[mask]
                nz_amps = a_host[mask]
                
                # 2. Global Index Lookup
                targets = np.searchsorted(basis_host, nz_cands)
                
                # 3. Boundary Check
                valid_idx = (targets < n_basis_total)
                if not np.any(valid_idx): return 0
                
                targets = targets[valid_idx]
                nz_cands = nz_cands[valid_idx]
                nz_amps = nz_amps[valid_idx]
                mask_rows = np.repeat(np.arange(i, b_end), n_terms)[mask][valid_idx]

                # 4. Exact Match Check (Collision validation)
                # Removed 'upper_tri' check because JAX handled it!
                exact_match = (basis_host[targets] == nz_cands)
                
                if np.any(exact_match):
                    buf_rows.append(mask_rows[exact_match])
                    buf_cols.append(targets[exact_match])
                    buf_vals.append(nz_amps[exact_match])
                    return len(buf_vals[-1])
                return 0

            if n_terms_h1 > 0:
                cands, amps = vmap_h1(states, H1_P, H1_Q, v1)
                current_buffer_count += process_and_buffer(cands, amps, n_terms_h1)

            if n_terms_h2 > 0:
                cands, amps = vmap_h2(states, H2_P, H2_Q, H2_R, H2_S, v2)
                current_buffer_count += process_and_buffer(cands, amps, n_terms_h2)

            # Flush Buffer
            if current_buffer_count >= buffer_limit:
                R = np.concatenate(buf_rows)
                C = np.concatenate(buf_cols)
                V = np.concatenate(buf_vals)
                # Merge into local Upper Triangle Accumulator
                mini_csr = scipy.sparse.coo_matrix((V, (R, C)), shape=(n_basis_total, n_basis_total)).tocsr()
                local_csr += mini_csr
                
                del R, C, V, mini_csr
                buf_rows, buf_cols, buf_vals = [], [], []
                current_buffer_count = 0
                gc.collect()

    # Final Flush
    if current_buffer_count > 0:
        R = np.concatenate(buf_rows)
        C = np.concatenate(buf_cols)
        V = np.concatenate(buf_vals)
        mini_csr = scipy.sparse.coo_matrix((V, (R, C)), shape=(n_basis_total, n_basis_total)).tocsr()
        local_csr += mini_csr
        del R, C, V
        gc.collect()
    
    # --- RETURN UPPER TRIANGLE ONLY ---
    return local_csr

import jax
import jax.numpy as jnp
import numpy as np
import ray
import psutil
import gc
import scipy.sparse
# from tqdm import tqdm
from collections import deque

@ray.remote(max_calls = 1)
def merge_chunks_ray(mat_a, mat_b):
    """Worker-side merge to keep driver RAM clean."""
    if mat_a is None: return mat_b
    if mat_b is None: return mat_a
    # Adding CSR matrices is efficient in SciPy
    return mat_a + mat_b


import psutil

def estimate_worker_ram(h1, h2, chunk_size, n_basis, show=True):
    h1_terms = len(h1) if h1 else 0
    h2_terms = len(h2) if h2 else 0
    
    vmap_h1 = chunk_size * h1_terms * 8 * 4
    vmap_h2 = chunk_size * h2_terms * 8 * 4
    sparse_buf = n_basis * 100 * 8
    overhead = 2e9
    
    total = vmap_h1 + vmap_h2 + sparse_buf + overhead
    
    if show:
        print(f"  h1 terms: {h1_terms}, h2 terms: {h2_terms}")
        print(f"  vmap_h2 cost: {vmap_h2/1e9:.1f} GB (dominant)")
        print(f"  Estimated worker RAM: {total/1e9:.1f} GB")
    return total


def build_hamiltonian(h1, h2, basis, chunk_size=None, show=True):
    print('use tree')
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, 
                 _system_config={"idle_worker_killing_time_threshold_ms": 5000}, 
                 logging_level="ERROR", 
                 include_dashboard=False,
                 _metrics_export_port=None)

    # 1. Adaptive chunk_size based on h2 size
    n_h2_terms = len(h2) if h2 else 0
    if chunk_size is None:
        if n_h2_terms > 10000:
            chunk_size = 150
        elif n_h2_terms > 1000:
            chunk_size = 1000
        else:
            chunk_size = 2500

    # 2. Prepare Shared Memory
    basis_ref = ray.put(np.array(basis, dtype=np.int64))
    
    def pack_h(h_dict, dims):
        if not h_dict: return None
        keys = np.array(list(h_dict.keys()), dtype=np.int64)
        vals = np.array(list(h_dict.values()), dtype=np.float64)
        return (*[keys[:, i] for i in range(dims)], vals)

    h1_ref = ray.put(pack_h(h1, 2))
    h2_ref = ray.put(pack_h(h2, 4))

    # 3. Resource Management
    buffer_limit_terms = 15_000_000
    est_worker_ram = estimate_worker_ram(h1, h2, chunk_size, len(basis), show=show)
    available_ram = psutil.virtual_memory().available
    safe_ram = available_ram * 0.6

    concurrency_target = max(1, int(safe_ram / est_worker_ram))
    MAX_IN_FLIGHT = concurrency_target

    n_basis = len(basis)
    all_chunks = deque([(i, min(i + chunk_size, n_basis)) 
                        for i in range(0, n_basis, chunk_size)])

    if show:
        pbar = tqdm(total=len(all_chunks), desc="Smart Building")
        print(f"[Smart Controller] System RAM:          {available_ram/1e9:.1f} GB")
        print(f"                   Est. Worker RAM:     {est_worker_ram/1e9:.1f} GB")
        print(f"                   Safe Concurrency:    {concurrency_target} Workers")
        print(f"                   Queue Limit:         {MAX_IN_FLIGHT} Tasks")
        print(f"                   Chunk Size:          {chunk_size}")
        print(f"                   Total Chunks:        {len(all_chunks)}")

    # --- PHASE 1: COMPUTE CHUNKS ---
    active_futures = []
    initial_results = []

    RAM_SAFE_LIMIT    = 60.0
    RAM_CRITICAL_LIMIT = 80.0

    # while all_chunks or active_futures:
    #     mem_percent = psutil.virtual_memory().percent

    #     # Submit only if RAM safe AND slots available
    #     while (all_chunks and
    #            len(active_futures) < MAX_IN_FLIGHT and
    #            mem_percent < RAM_SAFE_LIMIT):
    stall_polls = 0
    while all_chunks or active_futures:
        mem_percent = psutil.virtual_memory().percent

        # Progress guarantee: with nothing in flight, waiting for memory can
        # never free any, so force one chunk through rather than spin forever.
        # (The original loop livelocked here when Ray spilled to a tmpfs /tmp,
        # which counts as used RAM and so held mem_percent above the gate.)
        if not active_futures and all_chunks and mem_percent >= RAM_SAFE_LIMIT:
            stall_polls += 1
            if stall_polls > 20:
                print(f'[Smart Controller] RAM at {mem_percent:.0f}% with no '
                      f'tasks in flight; forcing one chunk to make progress.',
                      flush=True)
                mem_percent = 0.0          # admit exactly one submission
                stall_polls = 0
            else:
                time.sleep(1.0)
        else:
            stall_polls = 0

        # Submit only if RAM safe AND slots available
        while (all_chunks and
               len(active_futures) < MAX_IN_FLIGHT and
               mem_percent < RAM_SAFE_LIMIT):
            start, end = all_chunks.popleft()
            f = build_chunk_ray.remote(
                start, end, basis_ref, h1_ref, h2_ref, chunk_size, buffer_limit_terms
            )
            active_futures.append(f)
            time.sleep(0.005)
            mem_percent = psutil.virtual_memory().percent  # re-check after each submit

        if active_futures:
            if mem_percent > RAM_CRITICAL_LIMIT:
                num_wait = len(active_futures)  # drain all
                if show: print(f"  ⚠ RAM critical ({mem_percent:.1f}%), draining {num_wait} tasks...")
            else:
                num_wait = 1

            ready_ids, active_futures = ray.wait(
                active_futures,
                num_returns=num_wait,
                timeout=0.1
            )
            for ref in ready_ids:
                initial_results.append(ref)
                if show: pbar.update(1)

            if mem_percent > RAM_CRITICAL_LIMIT:
                gc.collect()

    if show: pbar.close()

    # --- PHASE 2: HIERARCHICAL TREE REDUCE ---
    if show: print(f" -> Starting Tree Reduce with {len(initial_results)} nodes...")

    current_level_refs = list(initial_results)
    level_count = 0

    while len(current_level_refs) > 1:
        next_level_refs = []
        for i in range(0, len(current_level_refs), 2):
            if i + 1 < len(current_level_refs):
                m_f = merge_chunks_ray.remote(current_level_refs[i], current_level_refs[i+1])
                next_level_refs.append(m_f)
            else:
                next_level_refs.append(current_level_refs[i])

        current_level_refs = next_level_refs
        level_count += 1
        if show: print(f"    Tree Level {level_count} dispatched. Nodes remaining: {len(current_level_refs)}")
        gc.collect()

    # --- PHASE 3: FINAL ASSEMBLY ---
    if show: print(" -> Fetching final result and symmetrizing...")
    final_matrix_upper = ray.get(current_level_refs[0])

    # Symmetrize: H = H_upper + H_upper.T, halve diagonal (added twice)
    final_matrix_full = final_matrix_upper + final_matrix_upper.transpose()
    diag_vals = final_matrix_full.diagonal()
    final_matrix_full.setdiag(diag_vals * 0.5)

    if final_matrix_full.nnz == 0:
        return BCSR.from_scipy_sparse(final_matrix_full)

    if n_basis > USE_GPU_ and HAS_GPU:
        with jax.default_device(jax.devices('gpu')[0]):
            H_csr = BCSR.from_scipy_sparse(final_matrix_full)
    else:
        with jax.default_device(jax.devices('cpu')[0]):
            H_csr = BCSR.from_scipy_sparse(final_matrix_full)

    del final_matrix_upper, final_matrix_full, diag_vals
    gc.collect()
    return H_csr

class SymmetryBasis:
    def __init__(self, n_qubits, n_electrons):
        self.n_qubits = int(n_qubits)
        self.n_electrons = int(n_electrons)
        self.n_spatial = self.n_qubits // 2

        # Spatial -> Interleaved
        self._spread_table = self._precompute_spread_table()
        self.states = None
    
    def _precompute_spread_table(self):
        table_size = 2**self.n_spatial
        indices = np.arange(table_size, dtype=np.int64)
        
        table = np.zeros(table_size, dtype=np.int64)
        for b in range(self.n_spatial):
            table |= (indices & (1 << b)) << b
            
        return table

    from functools import lru_cache

    @lru_cache(maxsize=128)
    def build(self, mode='PCS', multiplicity=None):
        mode = mode.upper()
        sectors = []
        if mode in ['MS', 'MHS']:
            if multiplicity is None: 
                raise ValueError("Multiplicity Required.")
            if (self.n_electrons + multiplicity) % 2 != 0:
                self.states = jnp.array([], dtype=jnp.int64)
                return self.states
            sectors.append(((self.n_electrons + multiplicity) // 2, (self.n_electrons - multiplicity) // 2))
        else:
            for na in range(min(self.n_electrons, self.n_spatial) + 1):
                nb = self.n_electrons - na
                if na <= self.n_spatial and nb <= self.n_spatial:
                    sectors.append((na, nb))

        final_states_list = []
        for na, nb in sectors:
            if mode in ['HS', 'MHS']:
                for alpha_indices in combinations(range(self.n_spatial), na):
                    a_mask = sum(1 << i for i in alpha_indices)
                    phys_a = self._spread_table[a_mask]
                    if nb > len(alpha_indices): continue
                    b_masks = [sum(1 << i for i in sub) for sub in combinations(alpha_indices, nb)]
                    if not b_masks: continue
                    phys_b = self._spread_table[np.array(b_masks, dtype=np.int64)] << 1
                    final_states_list.append(phys_a | phys_b)
            else:
                alphas = [sum(1 << i for i in c) for c in combinations(range(self.n_spatial), na)]
                betas = [sum(1 << i for i in c) for c in combinations(range(self.n_spatial), nb)]
                if not alphas or not betas: continue
                phys_a = self._spread_table[np.array(alphas, dtype=np.int64)]
                phys_b = self._spread_table[np.array(betas, dtype=np.int64)] << 1
                final_states_list.append((phys_a[:, None] | phys_b[None, :]).flatten())

        if not final_states_list: 
            self.states = jnp.array([], dtype=jnp.int64)
        else:
            full_basis = np.concatenate(final_states_list)
            full_basis.sort()
            self.states = jnp.array(full_basis, dtype=jnp.int64)
        return self.states
    
    # show modular
    def state_to_str(self, state_int, style='block', show_orbitals=None):
        """
        Translates state integer to string, focusing only on 'show_orbitals'.

        Args:
            show_orbitals (list): Indices of spatial orbitals to display (e.g. [2,3,4]).
                                  If None, shows all.
        """
        # Default: Show All
        if show_orbitals is None:
            target_indices = range(self.n_spatial)
        else:
            target_indices = show_orbitals

        if style == 'block':
            # We iterate Reversed so high indices are on the Left (Standard Binary)
            sorted_targets = sorted(target_indices, reverse=True)
            alpha_bits = []
            beta_bits = []

            for i in sorted_targets:
                if i >= self.n_spatial: continue # Safety

                # Extract Alpha (2*i)
                a_bit = (state_int >> (2 * i)) & 1
                alpha_bits.append(str(a_bit))

                # Extract Beta (2*i + 1)
                b_bit = (state_int >> (2 * i + 1)) & 1
                beta_bits.append(str(b_bit))

            prefix = "..." if (show_orbitals is not None) else ""
            return f"{prefix}|{''.join(beta_bits)}⟩_β |{''.join(alpha_bits)}⟩_α"

        elif style == 'chem':
            occ = []
            # For chem, we sort ascending (0a, 1b...)
            sorted_targets = sorted(target_indices)

            for i in sorted_targets:
                if i >= self.n_spatial: continue

                if (state_int >> (2 * i)) & 1: occ.append(f"{i}α")
                if (state_int >> (2 * i + 1)) & 1: occ.append(f"{i}β")

            return f"|{' '.join(occ)}>"

        elif style == 'interleaved':
            return f"|{state_int:0{self.n_qubits}b}>"

        return str(state_int)

    def print_basis(self, limit=10, style='block', show_orbitals=None):
        """Prints the first 'limit' states in the requested format."""
        if self.states is None:
            print("Basis not built yet.")
            return
       
        vis_tag = f"Orbitals {show_orbitals}" if show_orbitals else "ALL"
        print(f"--- Basis View (Total: {len(self.states)}, Visible: {vis_tag}, Style: {style}) ---")
        count = 0

        for s in np.array(self.states):
            if count >= limit: break
            s_int = int(s)
            print(f"{count:<4} {self.state_to_str(s_int, style, show_orbitals)}")
            count += 1
        if len(self.states) > limit:
            print(f"... ({len(self.states) - limit} more) ...")

    # TO DO: limit the number of showing vectors and provide output file
    def decompose_wavefunction(self, vector, threshold=1e-6, style='block', show_orbitals=None):
        """
        Prints the wavefunction composition, optionally hiding frozen orbitals.
        """
        if self.states is None:
            print("❌ Basis not built.")
            return

        vector_host = np.array(vector)
        states_host = np.array(self.states)

        # Header Info
        vis_tag = f"Orbitals {show_orbitals}" if show_orbitals else "ALL"
        print(f"--- Wavefunction ({style.upper()} | Visible: {vis_tag}) ---")

        sorted_indices = np.argsort(np.abs(vector_host))[::-1]

        count = 0
        norm = 0.0

        for idx in sorted_indices:
            amp = vector_host[idx]
            if abs(amp) < threshold: break

            phys_int = states_host[idx]

            s_str = self.state_to_str(phys_int, style=style, show_orbitals=show_orbitals)

            sign = "+" if amp >= 0 else "-"
            print(f" {sign} {abs(amp):.6f}  {s_str}")

            norm += abs(amp)**2
            count += 1

        print(f"--- Shown: {count} | Norm: {norm:.4f} ---")

    def plot_wavefunction_nmr(self, vector, top_k=20, threshold=1e-6, style='block', show_orbitals=None):
        import matplotlib.pyplot as plt
        import numpy as np

        if self.states is None:
            print("❌ Basis not built.")
            return

        vector_host = np.array(vector)
        
        # 1. Grab top K indices by absolute magnitude
        idx_sorted = np.argsort(np.abs(vector_host))[::-1][:top_k]
        
        data = []
        for idx in idx_sorted:
            amp = vector_host[idx]
            if abs(amp) < threshold: 
                continue
                
            s = int(self.states[idx])
            
            # 2. Count alpha and beta (interleaved bits: alpha = even, beta = odd)
            n_alpha = sum((s >> (2 * i)) & 1 for i in range(self.n_spatial))
            n_beta = sum((s >> (2 * i + 1)) & 1 for i in range(self.n_spatial))
            sz = 0.5 * (n_alpha - n_beta)
            
            label = self.state_to_str(s, style=style, show_orbitals=show_orbitals)
            
            data.append({
                'amp': amp,
                'abs_amp': abs(amp),
                'label': label,
                'sz': sz
            })

        if not data:
            print("No states above threshold to plot.")
            return

        # 3. CRITICAL: Sort by Sz first (to group zones), then by absolute amplitude
        data.sort(key=lambda x: (x['sz'], -x['abs_amp']))
        
        amps = [d['amp'] for d in data]
        labels = [d['label'] for d in data]
        sz_values = [d['sz'] for d in data]

        # 4. Setup Plot
        plt.figure(figsize=(12, 6))
        x = np.arange(len(amps))
        
        unique_sz = sorted(list(set(sz_values)))
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_sz)))
        sz_color_map = {sz: colors[i] for i, sz in enumerate(unique_sz)}
        
        seen_sz = set() # Track for legend

        # 5. Draw the "NMR Sticks"
        for i, (amp, sz) in enumerate(zip(amps, sz_values)):
            lbl = f"$S_z$ = {sz}" if sz not in seen_sz else ""
            seen_sz.add(sz)
            
            plt.vlines(x[i], 0, amp, colors=sz_color_map[sz], lw=2.5, label=lbl)
            plt.plot(x[i], amp, 'o', color=sz_color_map[sz], markersize=5) # Peak tip
            
        # 6. Highlight Zones properly
        for sz in unique_sz:
            zone_indices = [i for i, val in enumerate(sz_values) if val == sz]
            if zone_indices:
                plt.axvspan(min(zone_indices) - 0.5, max(zone_indices) + 0.5, 
                            alpha=0.1, color=sz_color_map[sz])

        # Aesthetics
        plt.axhline(0, color='black', lw=1)
        plt.xticks(x, labels, rotation=45, ha='right', family='monospace', fontsize=9)
        plt.ylabel("Amplitude $c_i$")
        plt.title(f"Wavefunction 'NMR' Spectrum (Top {len(amps)}, Grouped by $S_z$)")
        
        plt.legend(loc='upper right', title="Spin Projections")
        plt.tight_layout()
        plt.show()

class HamiltonianLoader:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits # warining size
        self.h1 = {}
        self.h2 = {}
        self.constant = 0.0

    def from_openfermion(self, of_op):
        """Standard OpenFermion Parser."""
        print(f"[Loader] Loading from OpenFermion (Terms: {len(of_op.terms)}) RHF(if S^2 = 0) ROHF(if S^2 != 0)")
        for term, coeff in of_op.terms.items():
            val = float(coeff.real)
            if len(term) == 0: 
                self.constant += val
            elif len(term) == 2: 
                self.h1[(term[0][0], term[1][0])] = val
            elif len(term) == 4: 
                self.h2[(term[0][0], term[1][0], term[2][0], term[3][0])] = val
        return self.h1, self.h2, self.constant

    def from_pyscf(self, mol, mf):
        """
        Loads 1e and 2e integrals from PySCF.
        Supports both RHF and UHF.
        """
        print(f"[Loader] Loading from PySCF ", end='')

        is_uhf = (len(mf.mo_coeff.shape) == 3)

        self.h1 = {}
        self.h2 = {}
        self.constant = float(mol.energy_nuc())

        def unpack(P):
            i = int(np.sqrt(2 * P + 0.25) - 0.5)
            j = P - i * (i + 1) // 2
            return i, j

        if not is_uhf:
            print("(RHF)")
            mo_coeff = mf.mo_coeff
            n_spatial = mo_coeff.shape[1]

            # --- H1: One-body ---
            h_core_ao = mf.get_hcore()
            h1_spatial = mo_coeff.T @ h_core_ao @ mo_coeff

            for i in range(n_spatial):
                for j in range(n_spatial):
                    val = h1_spatial[i, j]
                    if abs(val) > 1e-10:
                        self.h1[(2*i, 2*j)] = val     # Alpha
                        self.h1[(2*i+1, 2*j+1)] = val # Beta

            # --- H2: Two-body  ---
            eri_compressed = ao2mo.kernel(mol, mo_coeff)
            n_pairs = n_spatial * (n_spatial + 1) // 2

            for idx_A in range(n_pairs):
                i, j = unpack(idx_A)
                for idx_B in range(idx_A + 1):
                    k, l = unpack(idx_B)
                    val = eri_compressed[idx_A, idx_B]
                    if abs(val) < 1e-10: continue

                    phys_val = 0.5 * val

                    # (i,j|k,l) 
                    perms = [(i, j, k, l), (j, i, k, l), (i, j, l, k), (j, i, l, k),
                             (k, l, i, j), (l, k, i, j), (k, l, j, i), (l, k, j, i)
                             ]
                    
                    unique_perms = set(perms)

                    for a, b, c, d in unique_perms:
                        # Alpha-Alpha
                        self.h2[(2*a, 2*c, 2*d, 2*b)] = phys_val
                        # Beta-Beta
                        self.h2[(2*a+1, 2*c+1, 2*d+1, 2*b+1)] = phys_val
                        # Alpha-Beta
                        self.h2[(2*a, 2*c+1, 2*d+1, 2*b)] = phys_val
                        # Beta-Alpha
                        self.h2[(2*a+1, 2*c, 2*d, 2*b+1)] = phys_val

        else:
            print("(UHF)")
            Ca = mf.mo_coeff[0] # Alpha Orbitals
            Cb = mf.mo_coeff[1] # Beta  Orbitals
            n_spatial = Ca.shape[1]
            
            
            # if n_spatial * 2 >= self.n_qubits:
            #     print(f"⚠️ Warning: Molecule has {n_spatial} orbitals ({n_spatial*2} qubits).")

            # --- H1: One-body (Block Diagonal) ---
            h_core_ao = mf.get_hcore()
            h1_aa = Ca.T @ h_core_ao @ Ca
            h1_bb = Cb.T @ h_core_ao @ Cb

            for p,q in product(range(n_spatial), repeat = 2):
                if abs(h1_aa[p, q]) > 1e-10:
                    self.h1[(2*p, 2*q)] = h1_aa[p, q]
                if abs(h1_bb[p, q]) > 1e-10:
                    self.h1[(2*p+1, 2*q+1)] = h1_bb[p, q]

            # --- H2: Two-body (3 Distinct Blocks) ---
            eri_aa = ao2mo.kernel(mol, (Ca, Ca, Ca, Ca), compact=False).reshape(n_spatial, n_spatial, n_spatial, n_spatial)
            # 2. Beta-Beta Repulsion
            eri_bb = ao2mo.kernel(mol, (Cb, Cb, Cb, Cb), compact=False).reshape(n_spatial, n_spatial, n_spatial, n_spatial)
            # 3. Alpha-Beta Repulsion
            eri_ab = ao2mo.kernel(mol, (Ca, Ca, Cb, Cb), compact=False).reshape(n_spatial, n_spatial, n_spatial, n_spatial)
            
            # (pq|rs) p^ r^ s q
            for p,q,r,s in product(range(n_spatial), repeat = 4):
                val_aa = eri_aa[p, q, r, s]
                if abs(val_aa) > 1e-10:
                    self.h2[(2*p, 2*r, 2*s, 2*q)] = 0.5 * val_aa

                val_bb = eri_bb[p, q, r, s]
                if abs(val_bb) > 1e-10:
                    self.h2[(2*p+1, 2*r+1, 2*s+1, 2*q+1)] = 0.5 * val_bb

                val_ab = eri_ab[p, q, r, s]
                if abs(val_ab) > 1e-10:
                    self.h2[(2*p, 2*r+1, 2*s+1, 2*q)] = 0.5 * val_ab
                    self.h2[(2*r+1, 2*p, 2*q, 2*s+1)] = 0.5 * val_ab

        return self.h1, self.h2, self.constant

def solve_eign(H_jax, nuclear_repulsion, k=1, v0 = None):
    def jax_sparse_to_cupy(H_jax_bcsr):
        H_jax_bcsr = H_jax_bcsr.block_until_ready()
        
        jax_data = H_jax_bcsr.data
        jax_indices = H_jax_bcsr.indices
        jax_indptr = H_jax_bcsr.indptr
        
        cp_data = cp.from_dlpack(jax_data)
        cp_indices = cp.from_dlpack(jax_indices)
        cp_indptr = cp.from_dlpack(jax_indptr)
           
        cp_data = cp_data.ravel()
        cp_indices = cp_indices.ravel()
        cp_indptr = cp_indptr.ravel()

        H_cupy = cp_sparse.csr_matrix((cp_data, cp_indices, cp_indptr), shape = H_jax_bcsr.shape)
        return H_cupy

    N = H_jax.shape[0]
    print(f"[EignSolver] Solving Eigenvalues (Size: {N})", end=' | ')
    try:
        _gpu = H_jax.data.device in jax.devices('gpu')
    except:
        _gpu = False

    if _gpu:
        print(f"Device: NVIDIA GPU (CuPy)", end=' | ')

        H_gpu = jax_sparse_to_cupy(H_jax) 
        print("Method: Sparse (eigsh)")
        
        if v0 is None:
            print('random initial')
            # Normal execution for the very first point
            vals_gpu, vecs_gpu = cp_linalg.eigsh(H_gpu, k=1, which='SA')
            ground_energy = float(vals_gpu[0]) + nuclear_repulsion
            ground_vec = jnp.array(cp.asnumpy(vecs_gpu[:, 0]))
        else:
            v0 = cp.asarray(np.array(v0), dtype=np.float64)
            print('using state tracking (Maximum Overlap)')
            
            # Ask for the lowest 3 states instead of 1 on the GPU
            # (Ensure your matrix dimension N is strictly greater than k)
            vals_gpu, vecs_gpu = cp_linalg.eigsh(H_gpu, k=1, v0=v0, which='SA')
            ground_energy = float(vals_gpu[0]) + nuclear_repulsion
            ground_vec = jnp.array(cp.asnumpy(vecs_gpu[:, 0]))
            

    else:
        print(f"Device: CPU (SciPy)", end=' | ')
        H_scipy = scipy.sparse.csr_matrix(
                (np.array(H_jax.data), 
                 np.array(H_jax.indices), 
                 np.array(H_jax.indptr)),
                 shape=H_jax.shape
                 )
        
        # --- CPU Dense Solver (Small Systems) ---
        if N < 4096:
            print("Method: Dense (eigh)")
            vals, vecs = scipy.linalg.eigh(H_scipy.todense())

            if v0 is not None:
                # NEW: Prevent root flipping by finding the state with max overlap to v0
                overlaps = np.abs(np.dot(vecs.T, np.array(v0)))
                best_idx = np.argmax(overlaps)
                ground_energy = vals[best_idx] + nuclear_repulsion
                ground_vec = jnp.array(vecs[:, best_idx])
            else:
                if len(vals) > 1 and abs(vals[1] - vals[0]) < 1e-8:
                    print('ground state degenerated')
                ground_energy = vals[0] + nuclear_repulsion
                ground_vec = jnp.array(vecs[:, 0])
            
        # --- CPU Sparse Solver (Intermediate Systems) ---
        else:
            print("Method: Sparse (eigsh)")
            if v0 is None:
                print('random initial ')
                vals, vecs = scipy.sparse.linalg.eigsh(
                H_scipy, k=1, which='SA', ncv=min(H_scipy.shape[0], 50), tol=0)
            
                vals, vecs = scipy.sparse.linalg.eigsh(
                H_scipy, k=k, v0=v0, which='SA',
                ncv=min(H_scipy.shape[0], 50), tol=0)
                ground_energy = vals[0] + nuclear_repulsion
                ground_vec = jnp.array(vecs[:, 0])

    print(f" Ground State Energy: {ground_energy:.8f} Ha")
    return ground_energy, ground_vec

class PySCFGroundState:
    def __init__(self, mode = 'uhf',subspace = 'HS', basis = 'sto-3g', uhf_shift = 0.2, allow_gpu = True, platform_selected = "cpu"):
        self.geometry = {}
        self.mode = mode
        self.basis = basis
        self.uhf_shift = uhf_shift
        self.subs = subspace  # MHS HS MS PCS
        self.gpu = allow_gpu
        self.platform_selected = platform_selected

    def subspace_solver(self, mol, mf, show = True, v0 = None):

        print(f'[Backend] ' + self.platform_selected.upper())

        with jax.default_device(jax.devices(self.platform_selected)[0]):
            loader = HamiltonianLoader(n_qubits=mol.nao)
            h1_tgt, h2_tgt, c_tgt = loader.from_pyscf(mol, mf)

            target_subspace = SymmetryBasis(2 * mol.nao, sum(mol.nelec))
            # target_subspace_basis = target_subspace.build(mode=self.subs.upper(), 
            #                                             multiplicity = int(round(mf.spin_square()[0])))
            target_subspace_basis = target_subspace.build(mode=self.subs.upper(), 
                                                        multiplicity = mol.spin)

        H_sparse = build_hamiltonian(h1_tgt, h2_tgt, target_subspace_basis, show = show)
        eign_value, eign_vector = solve_eign(H_sparse, c_tgt, k = 1, v0 = v0)
        return eign_value, eign_vector, H_sparse, target_subspace_basis
        # return eig_value, (eig_vector, target_subspace_basis), H_sparse

    def solve(self, geometry, v0 = None, spin = None):
        if spin is None:
            spin = 0
            print('[PySCFGroundState] Auto set spin to 0')

        mol = gto.M(atom = geometry, basis=self.basis , spin = spin, verbose=0)
        temp_mol = mol.copy()
        if self.mode == 'uhf':
            hf_results = 10000
            possible_spins = [mol.spin + s for s in [0, 2, 4] if ((mol.nelec[0] + s) <= mol.nao and (mol.nelec[1] - s) >= 0) ]
            for i in possible_spins:
                temp_mol.spin = i
                mf = scf.UHF(temp_mol)
                dm_pert = mf.get_init_guess()
                if dm_pert[1].shape[0] > 0: 
                    dm_pert[0][0,0] += self.uhf_shift  # Alpha density
                    dm_pert[1][0,0] -= self.uhf_shift  # Beta density
                mf.kernel(dm_pert)
                
                if mf.e_tot < hf_results:
                    hf_results = mf.e_tot
                    save_mf = mf #Na - Nb

            eign_value, eign_vector, H_sparse, target_subspace_basis = self.subspace_solver(mol, save_mf, v0 = v0)
            
        elif self.mode == 'rhf':
            hf_mf = scf.HF(mol)
            hf_mf.kernel()
            eign_value, eign_vector, H_sparse, target_subspace_basis = self.subspace_solver(mol, hf_mf, v0 = v0)
            save_mf = hf_mf
            hf_results = hf_mf.e_tot

        mol = gto.M(atom = geometry, basis=self.basis , spin = spin, verbose=0)
        hf_mf = scf.HF(mol)
        hf_mf.kernel()
        cisolver = fci.FCI(mol, hf_mf.mo_coeff)
        e_fci_val, _ = cisolver.kernel()

        Ref = {f'{self.mode}' : hf_results,
               'fci' : e_fci_val}
        del H_sparse
        gc.collect()
        if HAS_GPU:  
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            
        return eign_value, (eign_vector, target_subspace_basis), 0, Ref, save_mf

    def scan(self, geometry, scan_range = np.linspace(0.5, 4.0, 50), mol_name='', ref = 'uhf'):
        import os
        import contextlib

        spin_scan = []
        subs_energy = []
        uhf_energy = []
        rhf_energy = []
        fci_energy = []

        scan_range = scan_range[::-1]
        dm_guess = None
        for d in tqdm(scan_range):
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                mol = gto.M(atom = geometry.format(d), basis=self.basis , spin = self.spin, verbose = 0)
                spin = 0

                mf_rhf = scf.RHF(mol).run()
                rhf_energy.append(mf_rhf.e_tot)
                
                cisolver = fci.FCI(mol, mf_rhf.mo_coeff)
                e_fci_val, _ = cisolver.kernel()
                fci_energy.append(e_fci_val)

                uhf_mf = scf.UHF(mol)
                if dm_guess is None:
                    dm_guess = self.get_general_fragment_guess(mol)
                else:
                    dm_guess = (dm_guess + self.get_general_fragment_guess(mol)) / 2

                uhf_mf.kernel(dm0=dm_guess)
                dm_guess = uhf_mf.make_rdm1()
                
                if ref == 'uhf':
                  eig_value, _, _ , _ = self.subspace_solver(mol, uhf_mf, show=False)
                else:
                  ref = 'rhf'
                  eig_value, _, _ , _= self.subspace_solver(mol, mf_rhf, show=False)

                subs_energy.append(eig_value)
                spin_scan.append(spin)
                uhf_energy.append(uhf_mf.e_tot)
            
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

        # Plot Styling: Use hollow markers to distinguish overlapping data points
        m_every = 2
        kwargs = {'markevery': m_every, 'markerfacecolor': 'none', 'markeredgewidth': 1.5}

        # --- Plot 1: Potential Energy Surface ---
        ax1.plot(scan_range, subs_energy, 'r--x', **kwargs, markersize=8, label='{} ({}))'.format(self.subs, ref))
        ax1.plot(scan_range, uhf_energy, 'm:s', **kwargs, markersize=10, label='UHF', alpha=0.6)
        ax1.plot(scan_range, rhf_energy, 'b-o', **kwargs, markersize=7, label='RHF')
        ax1.plot(scan_range, fci_energy, 'g-.+', **kwargs, markersize=6, label='FCI')

        ax1.set_ylabel('Energy (Hartree)')
        ax1.set_title(f'{mol_name} Dissociation Curve ({self.basis.upper()})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Plot 2: Spin Contamination ---
        # Shows where the single-determinant approximation fails (S^2 deviating from 0)
        ax2.plot(scan_range, spin_scan, 'r--x', **kwargs, markersize=8, label='UHF')
        

        ax2.set_xlabel('Bond Length (Å)')
        ax2.set_ylabel(r"Spin Expectation Value $\langle S^2 \rangle$")
        ax2.set_ylim(-0.1, 1.2)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return scan_range, rhf_energy, uhf_energy, fci_energy, subs_energy, spin_scan

    def get_spin(self, symbol):       
        symbol = symbol.capitalize()
        
        # Dictionary of spins (N_unpaired)
        spins = {
            'H': 1, 'He': 0,
            'Li': 1, 'Be': 0,
            'B': 1, 'C': 2, 'N': 3, 'O': 2, 'F': 1, 'Ne': 0,
            'Na': 1, 'Mg': 0,
            'Al': 1, 'Si': 2, 'P': 3, 'S': 2, 'Cl': 1, 'Ar': 0,
            'K': 1, 'Ca': 0
        }
    
        return spins.get(symbol, 0) # Default to 0 if unknown

    def get_general_fragment_guess(self, mol, atom_orientations=None):
        coords = mol.atom_coords()
        symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
        if atom_orientations is None:
            atom_orientations = ['up' if i % 2 == 0 else 'down' for i in range(mol.natm)]
        alpha_blocks, beta_blocks = [], []
        for i, sym in enumerate(symbols):
            atom_mol = gto.M(atom=[[sym, coords[i]]], basis=mol.basis, spin=self.get_spin(sym), charge=0, verbose=0)
            dm_atom = scf.ROHF(atom_mol).run().make_rdm1()
            orientation = atom_orientations[i].lower()[0]
            if orientation == 'u' or orientation == 'a': 
                alpha_blocks.append(dm_atom[0]); beta_blocks.append(dm_atom[1])
            else: 
                alpha_blocks.append(dm_atom[1]); beta_blocks.append(dm_atom[0])
        guess_alpha = scipy.linalg.block_diag(*alpha_blocks)
        guess_beta  = scipy.linalg.block_diag(*beta_blocks)
        return np.array([guess_alpha, guess_beta])
    
def Fidelity(data_load, mol_name, method, subs):
    SymB = SymmetryBasis(2*data_load[mol_name][method]['PCS']['mf'].mol.nao, sum(data_load[mol_name][method]['PCS']['mf'].mol.nelec))
    SymB.build()
    V_ref, C_ref = data_load[mol_name][method]['PCS']['eign_vec']
    ref = dict(list(zip(np.array(C_ref),np.array(V_ref))))

    norm = 0
    for v,c in list(zip(data_load[mol_name][method][subs]['eign_vec'])):
        if c in C_ref:
            norm += v * ref[c]
    return norm


