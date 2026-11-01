"""Table 7 (Appendix): spin purity <S^2> of the block ground state of every MHS(Omega) block, RHF and UHF references.

Reads subspace_results.csv (written by runner.py). S = Omega/2, target S(S+1).
Run from this folder:  python table_spin_purity.py   ->  table7_spin_purity.csv
"""
import csv, os
from fractions import Fraction

ORDER = ["H2", "LiH", "BeH2", "CH", "NH", "BH3", "OH", "HF", "H2O", "NH3", "CH4",
         "CN", "N2", "CO", "CNH", "NO", "O2", "NF", "H2CO", "H2O2"]   # as in Table 5

here = os.path.dirname(os.path.abspath(__file__))
s2 = {}
for r in csv.DictReader(open(os.path.join(here, "subspace_results.csv"))):
    if r["Subspace"].startswith("MHS"):
        s2[(r["Molecule"], int(float(r["Omega"])), r["Reference"])] = float(r["Spin_Sq_Subspace"])


def f4(x):  # 4 decimals; round(...) + 0.0 prints -0.0000 (numerical noise) as 0.0000
    return f"{round(x, 4) + 0.0:.4f}"


rows = []
for mol in ORDER:
    for om in sorted({o for (m, o, _) in s2 if m == mol}):
        S = Fraction(om, 2)
        rows.append({"Molecule": mol, "Omega": om, "S": str(S), "S(S+1)": f"{float(S * (S + 1)):.2f}",
                     "RHF": f4(s2[(mol, om, "RHF")]), "UHF": f4(s2[(mol, om, "UHF")])})

with open(os.path.join(here, "table7_spin_purity.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print("  ".join(f"{v:>8}" for v in r.values()))
print(f"{len(rows)} blocks")
