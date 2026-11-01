"""Tables 1 and 5: subspace dimensions and qubit counts (STO-3G), from the closed forms of the paper.

  PCS        = C(2M, N)
  MS         = C(M, N_a) C(M, N_b)               fixed M_s = S, N_a - N_b = 2S
  HS         = sum_{k=0}^{floor(N/2)} C(M, N-k) C(N-k, k)          (Prop. SizeEstimation)
  MHS(2S)    = C(M, (N+2S)/2) C((N+2S)/2, (N-2S)/2)                (block listed in the tables)
  qubits     = ceil(log2 D);  JW = 2M
M, N and 2S (= mol_info 'spin') come from data/mol_info.py; H22 is the 1 Angstrom chain of 03_H22.
Run from this folder:  python table_sizes.py   ->  table1.csv, table5.csv
"""
import csv, os, sys
from math import comb
from pyscf import gto

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
import mol_info

TABLE5 = ["H2", "LiH", "BeH2", "CH", "NH", "BH3", "OH", "HF", "H2O", "NH3", "CH4",
          "CN", "N2", "CO", "CNH", "NO", "O2", "NF", "H2CO", "H2O2", "H22"]
TABLE1 = ["HF", "H2O", "NH3", "CH4", "O2", "H2O2"]


def q(D):  # qubits for a D-dimensional space, ceil(log2 D) in exact integer arithmetic
    return (D - 1).bit_length()


def molecule(name):
    if name == "H22":
        info = {"atom": "; ".join(f"H 0 0 {i * 1.0:.2f}" for i in range(22)), "charge": 0, "spin": 0}
    else:
        info = getattr(mol_info, name)
    mol = gto.M(atom=info["atom"], basis="sto-3g", spin=info["spin"], charge=info["charge"], verbose=0)
    return mol.nao, mol.nelectron, info["spin"]


def sizes(M, N, two_s):
    na, nb = (N + two_s) // 2, (N - two_s) // 2
    return dict(PCS=comb(2 * M, N),
                MS=comb(M, na) * comb(M, nb),
                HS=sum(comb(M, N - k) * comb(N - k, k) for k in range(N // 2 + 1)),
                MHS=comb(M, na) * comb(na, nb))


rows5, rows1 = [], []
for name in TABLE5:
    M, N, two_s = molecule(name)
    d = sizes(M, N, two_s)
    rows5.append({"Molecule": name, "N_so": 2 * M, "N_e": N, "2S": two_s,
                  **{f"{k}_size": v for k, v in d.items()}, **{f"{k}_qubits": q(v) for k, v in d.items()}})
    if name in TABLE1:
        rows1.append({"Molecule": name, "MHS": q(d["MHS"]), "HS": q(d["HS"]), "MS": q(d["MS"]),
                      "PCS": q(d["PCS"]), "JW": 2 * M})
rows1.sort(key=lambda r: TABLE1.index(r["Molecule"]))

here = os.path.dirname(os.path.abspath(__file__))
for fname, rows in (("table5.csv", rows5), ("table1.csv", rows1)):
    with open(os.path.join(here, fname), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"== {fname}")
    for r in rows:
        print("  " + "  ".join(f"{k}={v}" for k, v in r.items()))
