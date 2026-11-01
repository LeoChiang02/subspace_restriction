# Independent re-implementation; the original generator is not available.
# Agrees with the committed 02_resources/hcb_hamiltonians/*_HCB_omega0.txt to 4e-7 elementwise.
# Those committed files remain the data behind Tables 2-3.
# Source: HCB_vs_SRS/hcb_hamiltonian.py. Changes: the path constants below, and SYSTEMS entries for H2, BeH2
# and H2O (geometries from data/mol_info.py). For all five molecules q, T, N_XBM and Lambda match Tables 2-3.
# Extra requirements: tequila-basic==1.9.12, pennylane==0.42.3 (see 02_resources/README.md).
"""Hamiltonians for the HCB-vs-SRS comparison (Omega = 0).

Energy convention (used everywhere in HCB_vs_SRS): ELECTRONIC energies.
  * The SRS .txt Hamiltonians of the certified repo are electronic-only.
  * The Tequila HCB Hamiltonian carries E_nuc in its identity coefficient; we
    subtract it, so that E_min(m=0) = 0 (the empty pair state).
  * Total energy = electronic + E_nuc (E_nuc from *_records.json); reported
    alongside, never used for |E - E0|, which is convention-independent.

Qubit ordering (HCB): Tequila's condensed HCB maps spatial orbital k -> qubit k,
|1> = pair occupied.  qml.from_openfermion keeps index k -> wire k.  PennyLane
wire 0 is the most significant bit of a basis-state index.
"""

import json
import os
import sys

import numpy as onp
import pennylane as qml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
VQE_DIR = os.path.join(REPO, "06_VQE", "mhs_hea")
RECORDS_DIR = os.path.join(REPO, "data", "records")
if VQE_DIR not in sys.path:
    sys.path.insert(0, VQE_DIR)

# Geometries copied from subspace_restriction/encoding/molecule_info.py (Angstrom).
SYSTEMS = {
    "HF":  {"geometry": "F 0.0 0.0 0.0\nH 0.0 0.0 0.9168", "n_pairs": 5, "M": 6,
            "srs_file": "HF_MHS_omega0.txt", "srs_q": 3, "srs_label": "HF_w0"},
    "LiH": {"geometry": "Li 0.0 0.0 0.0\nH 0.0 0.0 1.5949", "n_pairs": 2, "M": 6,
            "srs_file": "LiH_MHS_omega0.txt", "srs_q": 4, "srs_label": "LiH_w0"},
    # Added for Tables 2-3; geometries from data/mol_info.py.
    "H2":  {"geometry": "H 0.0 0.0 0.0\nH 0.0 0.0 0.7414", "n_pairs": 1, "M": 2,
            "srs_file": "H2_MHS_omega0.txt", "srs_q": 1, "srs_label": "H2_w0"},
    "BeH2": {"geometry": "Be 0.0 0.0 0.0\nH 0.0 0.0 1.3264\nH 0.0 0.0 -1.3264", "n_pairs": 3, "M": 7,
             "srs_file": "BeH2_MHS_omega0.txt", "srs_q": 6, "srs_label": "BeH2_w0"},
    "H2O": {"geometry": "O 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692", "n_pairs": 5, "M": 7,
            "srs_file": "H2O_MHS_omega0.txt", "srs_q": 5, "srs_label": "H2O_w0"},
}
BASIS = "sto-3g"


def record(mol):
    """Omega = 0, shift = 'zero' entry of <mol>_records.json (the one the SRS runs used)."""
    with open(os.path.join(RECORDS_DIR, f"{mol}_records.json")) as f:
        recs = json.load(f)
    hits = [r for r in recs if r["omega"] == 0 and r["shift"] == "zero"]
    assert len(hits) == 1, hits
    return hits[0]


def srs_hamiltonian(mol):
    """Certified SRS Hamiltonian, loaded with the certified loader (tool.py)."""
    from tool import qml_Hamli_from_file
    return qml_Hamli_from_file(os.path.join(VQE_DIR, "hamiltonians", SYSTEMS[mol]["srs_file"]))


def tequila_molecule(mol):
    import tequila as tq
    # frozen_core=False: Tequila 1.9 freezes the core by default with PySCF; the
    # records use all M = 6 spatial orbitals.  Orbitals are canonical RHF ("hf").
    return tq.Molecule(geometry=SYSTEMS[mol]["geometry"], basis_set=BASIS, backend="pyscf",
                       frozen_core=False)


def hcb_hamiltonian(mol, electronic=True):
    """H_HCB via Tequila -> OpenFermion -> PennyLane.

    Returns (H, E_nuc_removed).  With electronic=True the Tequila constant
    E_nuc is subtracted from the identity term.
    """
    tqm = tequila_molecule(mol)
    H_tq = tqm.make_hardcore_boson_hamiltonian(condensed=True)
    H_of = H_tq.to_openfermion()
    H = qml.from_openfermion(H_of)
    coeffs, ops = H.terms()
    coeffs = [float(onp.real(c)) for c in coeffs]
    assert tqm.n_orbitals == SYSTEMS[mol]["M"] and tqm.integral_manager._orbital_type == "hf"
    e_nuc = float(tqm.get_integrals()[0]) if electronic else 0.0  # constant = E_nuc (no frozen core)
    # Shift identity coefficient by -E_nuc
    new_c, new_o, found = [], [], False
    for c, o in zip(coeffs, ops):
        if isinstance(o, qml.Identity) or o.name == "Identity":
            c = c - e_nuc
            found = True
        new_c.append(c)
        new_o.append(o)
    if not found and e_nuc != 0.0:
        new_c.append(-e_nuc)
        new_o.append(qml.Identity(0))
    M = SYSTEMS[mol]["M"]
    # Map wires to 0..M-1 explicitly (from_openfermion uses integer labels already)
    H = qml.Hamiltonian(new_c, new_o).simplify()
    return H, e_nuc


def pair_number_operator(M):
    """N_pair = sum_i (1 - Z_i)/2 on wires 0..M-1."""
    return qml.Hamiltonian([0.5] * M + [-0.5] * M,
                           [qml.Identity(i) for i in range(M)] + [qml.PauliZ(i) for i in range(M)])


def pair_numbers(M):
    """Pair number of each computational basis index (wire 0 = MSB)."""
    return onp.array([bin(k).count("1") for k in range(2 ** M)])


def sector_minima(H, M):
    """E_min(m) for m = 0..M by exact diagonalisation of the m-pair block."""
    Hm = qml.matrix(H, wire_order=list(range(M)))
    assert onp.allclose(Hm, Hm.conj().T)
    npair = pair_numbers(M)
    off = onp.abs(Hm[npair[:, None] != npair[None, :]]).max()
    out = {}
    for m in range(M + 1):
        idx = onp.where(npair == m)[0]
        out[m] = float(onp.linalg.eigvalsh(Hm[onp.ix_(idx, idx)])[0])
    return out, float(off)


def hf_pair_bitstring(M, n):
    """HF pair occupation: n lowest orbitals (qubits 0..n-1) occupied."""
    return [1] * n + [0] * (M - n)
