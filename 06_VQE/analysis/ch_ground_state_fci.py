# Note: fixed spin_square call for pyscf 2.14 (was scf.uhf.spin_square(mf), which expects MO coefficients).
import numpy as np
from pyscf import fci, gto, scf

# 1. Define the molecule in STO-3G
# (Same geometry as CH in data/mol_info.py.)
mol = gto.Mole()
mol.atom = "C 0.0 0.0 0.0; H 0.0 0.0 1.1199"
mol.basis = "sto-3g"
mol.spin = 1  # 2S = 1 (N_alpha - N_beta = 1, i.e., N_alpha=4, N_beta=3)
mol.charge = 0
mol.build()

# 2. Run ROHF and check SCF energy and spin
mf = scf.ROHF(mol)
mf.run()
ss_scf = mf.spin_square()[0]
print(f"ROHF Energy: {mf.e_tot:.8f} Ha | <S^2>_SCF: {ss_scf:.4f}")

# 3. Solve for the lowest multiple roots in FCI within (N_alpha=4, N_beta=3)
cisolver = fci.FCI(mf)
nroots = 5
cisolver.nroots = nroots

energies, fcivecs = cisolver.kernel()

print("\n--- FCI Roots in (N_alpha=4, N_beta=3) Sector ---")
for i, (e, c) in enumerate(zip(energies, fcivecs)):
  ss, mult = cisolver.spin_square(c, mol.nao, mol.nelec)
  print(f"Root {i}: Energy = {e:.8f} Ha | <S^2> = {ss:.4f} (Multiplicity ~ {mult:.2f})")