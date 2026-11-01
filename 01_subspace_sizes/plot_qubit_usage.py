"""Figure 2: qubit requirements q = ceil(log2 D) versus number of spatial orbitals M.
Only M with N = cM an even integer are used (seniority-zero requires paired electrons).
Curves for c and 2-c coincide (particle-hole symmetry), so each panel shows both."""
from math import comb, ceil, log2
import matplotlib.pyplot as plt

def q(D):  # qubits for a D-dimensional space
    return 0 if D <= 1 else ceil(log2(D))

panels = [(0.25, 1.75), (0.5, 1.5), (0.75, 1.25), (1.0,)]
fig, axes = plt.subplots(1, 4, figsize=(18, 4.3), sharey=True)
for ax, cs in zip(axes, panels):
    c = cs[0]
    Ms = [M for M in range(0, 101) if abs(c*M - round(c*M)) < 1e-9 and round(c*M) % 2 == 0]
    data = {'JW': [], 'HCB': [], 'PCS': [], 'MHS': []}
    for M in Ms:
        N = round(c*M)
        # check particle-hole symmetry with the partner filling 2-c
        for c2 in cs[1:]:
            N2 = round(c2*M); assert comb(2*M, N) == comb(2*M, N2) and comb(M, N//2) == comb(M, N2//2)
        data['JW'].append(2*M); data['HCB'].append(M)
        data['PCS'].append(q(comb(2*M, N))); data['MHS'].append(q(comb(M, N//2)))
    ax.plot(Ms, data['JW'], color='black', label='Jordan–Wigner')
    ax.plot(Ms, data['HCB'], color='tab:blue', label='HCB')
    ax.plot(Ms, data['PCS'], color='tab:red', label='PCS')
    ax.plot(Ms, data['MHS'], color='tab:green', label='MHS(0)')
    ax.set_title('$c = ' + ', '.join(f'{x:g}' for x in cs) + '$', fontsize=13)
    ax.set_xlim(0, 100); ax.set_ylim(0, 200)
    ax.set_xlabel('Number of spatial orbitals ($M$)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    print(cs, 'points', len(Ms), 'M=100 ->' if Ms[-1]==100 else f'M={Ms[-1]} ->', {k: v[-1] for k, v in data.items()})
axes[0].set_ylabel('Qubit usage', fontsize=12)
axes[0].legend(loc='upper left', fontsize=10)
fig.tight_layout()
fig.savefig('qubit_usage_comparison.png', dpi=300)
fig.savefig('qubit_usage_comparison.pdf')
