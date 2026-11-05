# Code by Calvin Ku; original repository: https://github.com/calvinku96/subspace_restriction (commit 8f4143af24c433cf7ea0b4d837fe05ac463e1644)
"""
VQE benchmark driver — reproduces Tables 2 & 3.

The molecule configuration below wires every benchmarked
molecule to BOTH of its custom-encoding Hamiltonian files (MHS and HS), whose
qubit counts were verified against the paper's Table 4:

    H2 : MHS = hamiltonians/H2_central_MH.txt   (1 qubit)
         HS  = hamiltonians/H2_Hund.txt         (2 qubits)
    HF : MHS = hamiltonians/HF_central_MH.txt   (3 qubits)
         HS  = hamiltonians/HF_central.txt      (5 qubits)
    LiH: MHS = hamiltonians/LiH_central_MH.txt  (4 qubits)
         HS  = hamiltonians/LiH_central.txt     (7 qubits)

The JW Hamiltonian is built on the fly via qml.qchem.molecular_hamiltonian and needs no file.
"""

import csv
import os
import sys
import time

import pennylane as qml
from pennylane import numpy as np

from tool import VQE_UCCSD, qml_Hamli_from_file  # was: from .tool import (package-relative in the original repo)

HAM_DIR = "hamiltonians"

# --- 1. CONFIGURATION ---
# Each molecule lists its custom encodings as {label: (filename, qubits)}.
EXAMPLE_MOLECULES = {
    "H2": {
        "symbols": ["H", "H"],
        "coordinates": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.7414]).reshape(2, 3),
        "exact_energy": -1.13727017,  # FCI, STO-3G
        "nelec": 2,
        "custom_encodings": {
            "MHS": (os.path.join(HAM_DIR, "H2_central_MH.txt"), 1),
            "HS": (os.path.join(HAM_DIR, "H2_Hund.txt"), 2),
        },
    },
    "HF": {
        "symbols": ["H", "F"],
        "coordinates": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9168]).reshape(2, 3),
        "exact_energy": -98.59658658,  # FCI, STO-3G
        "nelec": 10,
        "custom_encodings": {
            "MHS": (os.path.join(HAM_DIR, "HF_central_MH.txt"), 3),
            "HS": (os.path.join(HAM_DIR, "HF_central.txt"), 5),
        },
    },
    "LiH": {
        "symbols": ["H", "Li"],
        "coordinates": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5949]).reshape(2, 3),
        "exact_energy": -7.88240341,  # FCI, STO-3G
        "nelec": 4,
        "custom_encodings": {
            "MHS": (os.path.join(HAM_DIR, "LiH_central_MH.txt"), 4),
            "HS": (os.path.join(HAM_DIR, "LiH_central.txt"), 7),
        },
    },
    "OH": {
        "symbols": ["O", "H"],
        "coordinates": np.array([0, 0, 0, 0, 0, 0.9697]).reshape(2, 3),
        "exact_energy": -74.38713413,
        "nelec": 9,
        "mult": 2,
        "custom_encodings": {},
    },
    # CH: supplied by the authors (not in commit 8f4143a); geometry as in data/mol_info.py.
    "CH": {
        "symbols": ["C", "H"],
        "coordinates": np.array([0, 0, 0, 0, 0, 1.1199]).reshape(2, 3),
        "exact_energy": -37.81134935,  # FCI, lowest root of the (4a, 3b) sector (quartet)
        "nelec": 7,
        "mult": 2,
        "custom_encodings": {},
    },
}


# Encodings to run: any of "JW", "MHS", "HS".
ENCODINGS_TO_RUN = ["JW"]
OPTIMIZERS = ["lbfgs", "adam"]
LAYERS_TO_TEST = [1]
NUM_RANDOM_TRIALS = 15
SUCCESS_THRESHOLD = 0.9


QUALITATIVE_ACCURACY = 0.01
CHEMICAL_ACCURACY = 1.6e-3

MAX_ITERATIONS = 1000
CONV_TOL = 1e-6
ADAM_LR = 0.1

RAW_TRIALS_CSV = "benchmark_raw_trials_uccsd.csv"
SUMMARY_STATUS_CSV = "benchmark_summary_status_uccsd.csv"
SUMMARY_DETAILED_CSV = "benchmark_summary_detailed_uccsd.csv"


# --- 2. BENCHMARK ---
def run_benchmarks(molecules=EXAMPLE_MOLECULES, encodings=ENCODINGS_TO_RUN):
    header = [
        "molecule",
        "encoding",
        "optimizer",
        "layers",
        "trial_num",
        "seed",
        "final_energy",
        "iterations",
        "converged",
        "wall_time_s",
        "status",
    ]
    with open(RAW_TRIALS_CSV, "w", newline="") as f:
        csv.writer(f).writerow(header)

    for molecule_name, config in molecules.items():
        print(f"\n--- Starting benchmark for {molecule_name} ---")

        # Assemble the requested Hamiltonians for this molecule.
        hamiltonians = {}
        if "JW" in encodings:
            jw_h, jw_q = qml.qchem.molecular_hamiltonian(
                config["symbols"],
                config["coordinates"],
                unit="angstrom",
                charge=config.get("charge", 0),
                mult=config.get("mult", 1),
                method="openfermion",
            )
            hamiltonians["JW"] = (jw_h, jw_q)
        for label, (fname, qubits) in config["custom_encodings"].items():
            if label in encodings:
                hamiltonians[label] = (qml_Hamli_from_file(fname), qubits)

        for encoding_name, (hamiltonian, qubits) in hamiltonians.items():
            for optimizer_name in OPTIMIZERS:
                print(f"Testing: Encoding={encoding_name}, Optimizer={optimizer_name}")
                for layers in LAYERS_TO_TEST:
                    print(f"  Trying {layers} layer(s)...")
                    for trial in range(NUM_RANDOM_TRIALS):
                        seed = trial  # 0..NUM_RANDOM_TRIALS-1; same seed value reused
                        # across molecule/encoding/optimizer/layers so trial
                        # index alone identifies the initialization used.
                        lr = ADAM_LR if optimizer_name == "adam" else None
                        start_time = time.perf_counter()
                        energy_history, _, converged = VQE_UCCSD(
                            Hamiltonian=hamiltonian,
                            qubits=qubits,
                            electrons=config["nelec"],
                            seed=seed,
                            # layers=layers,
                            optimizer=optimizer_name,
                            max_iterations=MAX_ITERATIONS,
                            conv_tol=CONV_TOL,
                            learning_rate=lr,
                            GPU=False,
                        )
                        wall_time_s = time.perf_counter() - start_time
                        final_energy = energy_history[-1]
                        error = abs(final_energy - config["exact_energy"])
                        iterations = len(energy_history) - 1

                        status = "Failure"
                        if converged:
                            if error < CHEMICAL_ACCURACY:
                                status = "Chemically Accurate"
                            elif error < QUALITATIVE_ACCURACY:
                                status = "Qualitatively Accurate"

                        with open(RAW_TRIALS_CSV, "a", newline="") as f:
                            csv.writer(f).writerow(
                                [
                                    molecule_name,
                                    encoding_name,
                                    optimizer_name,
                                    layers,
                                    trial + 1,
                                    seed,
                                    f"{final_energy:.6f}",
                                    iterations,
                                    converged,
                                    f"{wall_time_s:.4f}",
                                    status,
                                ]
                            )
                        sys.stdout.flush()


# --- 3. ANALYSIS (csv-only, no pandas) ---
def analyze_and_save_summaries(raw_filename, molecule_name, molecule):
    def save_to_csv(filename, header, rows):
        with open(filename, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)

    try:
        with open(raw_filename, "r") as f:
            all_trials = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"\nError: Raw data file '{raw_filename}' not found.")
        return

    trials = [t for t in all_trials if t["molecule"] == molecule_name]
    summary_status_list, summary_detailed_list = [], []

    groups = {}
    for trial in trials:
        groups.setdefault((trial["encoding"], trial["optimizer"]), []).append(trial)

    exact = molecule["exact_energy"]
    for (encoding, optimizer), group in groups.items():
        chem_found, qual_found = False, False
        for layers in LAYERS_TO_TEST:
            layer_trials = [t for t in group if int(t["layers"]) == layers]
            if not layer_trials:
                continue
            chem = [t for t in layer_trials if t["status"] == "Chemically Accurate"]
            if not chem_found and len(chem) / NUM_RANDOM_TRIALS >= SUCCESS_THRESHOLD:
                summary_status_list.append(
                    {
                        "Method": encoding,
                        "Layers": layers,
                        "Optimizer": optimizer,
                        "Status": "Chemically Accurate",
                    }
                )
                rep = chem[0]
                summary_detailed_list.append(
                    {
                        "Method": encoding,
                        "Layers": layers,
                        "Optimizer": optimizer,
                        "Final Energy (Ha)": rep["final_energy"],
                        "Error vs. Exact": f"{abs(float(rep['final_energy']) - exact):.6f}",
                        "Status": "Chemically Accurate",
                    }
                )
                chem_found = True
            qual = [
                t
                for t in layer_trials
                if t["status"] in ("Qualitatively Accurate", "Chemically Accurate")
            ]
            if not qual_found and len(qual) / NUM_RANDOM_TRIALS >= SUCCESS_THRESHOLD:
                summary_status_list.append(
                    {
                        "Method": encoding,
                        "Layers": layers,
                        "Optimizer": optimizer,
                        "Status": "Qualitatively Accurate",
                    }
                )
                qual_found = True
        if not qual_found:
            summary_status_list.append(
                {
                    "Method": encoding,
                    "Layers": "N/A",
                    "Optimizer": optimizer,
                    "Status": "Failure",
                }
            )
        if not chem_found:
            best = min(group, key=lambda t: abs(float(t["final_energy"]) - exact))
            summary_detailed_list.append(
                {
                    "Method": encoding,
                    "Layers": best["layers"],
                    "Optimizer": optimizer,
                    "Final Energy (Ha)": best["final_energy"],
                    "Error vs. Exact": f"{abs(float(best['final_energy']) - exact):.6f}",
                    "Status": "Failure (Local Minimum)",
                }
            )

    save_to_csv(
        SUMMARY_STATUS_CSV,
        ["Method", "Layers", "Optimizer", "Status"],
        summary_status_list,
    )
    save_to_csv(
        SUMMARY_DETAILED_CSV,
        [
            "Method",
            "Layers",
            "Optimizer",
            "Final Energy (Ha)",
            "Error vs. Exact",
            "Status",
        ],
        summary_detailed_list,
    )
    print(f"\nSummaries saved to '{SUMMARY_STATUS_CSV}' and '{SUMMARY_DETAILED_CSV}'")


if __name__ == "__main__":
    run_benchmarks()
    print(f"\nBenchmark complete. Raw data saved to {RAW_TRIALS_CSV}")
    for mol in EXAMPLE_MOLECULES:
        analyze_and_save_summaries(RAW_TRIALS_CSV, molecule_to_analyze=mol)
