# Code by Calvin Ku; original repository: https://github.com/calvinku96/subspace_restriction (commit 8f4143af24c433cf7ea0b4d837fe05ac463e1644)
import pyscf
from pyscf import fci, gto, scf

# Define the spin = 0 (singlet) systems with experimental/standard geometries
molecules = {
    "H2": """
        H 0.0 0.0 0.0
        H 0.0 0.0 0.7414
    """,
    "HF": """
        H 0.0 0.0 0.0
        F 0.0 0.0 0.9168
    """,
    "LiH": """
        Li 0.0 0.0 0.0
        H  0.0 0.0 1.5949
    """,
}

basis_set = "sto-3g"

print(
    f"{'System':<8} | {'RHF Energy (E_hf)':<18} | {'FCI Energy (E_fci)':<18} | {'Correlation Energy':<18}"
)
print("-" * 70)

for name, coords in molecules.items():
    # 1. Build the molecule (spin=0 denotes 2S = 0, i.e., singlet state)
    mol = gto.M(atom=coords, basis=basis_set, spin=0, verbose=0)

    # 2. Run Restricted Hartree-Fock (RHF)
    mf = scf.RHF(mol)
    e_hf = mf.kernel()

    # 3. Perform Full Configuration Interaction (FCI) on top of RHF
    fci_solver = fci.FCI(mf)
    e_fci, ci_vec = fci_solver.kernel()

    # 4. Calculate correlation energy (E_fci - E_hf)
    e_corr = e_fci - e_hf

    print(f"{name:<8} | {e_hf:<18.8f} | {e_fci:<18.8f} | {e_corr:<18.8f}")
