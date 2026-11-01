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

import os
import csv
import time
import pennylane as qml
from pennylane import numpy as np

from tool import VQE_HEA_RY, qml_Hamli_from_file

HAM_DIR = "hamiltonians" 

# --- 1. CONFIGURATION ---
# Each molecule lists its custom encodings as {label: (filename, qubits)}.
# MOLECULES = {
#     "H2": {
#         "symbols": ["H", "H"],
#         "coordinates": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.7414]).reshape(2, 3),
#         "exact_energy": -1.851024168,           # FCI, STO-3G
#         "custom_encodings": {
#             "MHS": (os.path.join(HAM_DIR, "H2_central_MH.txt"), 1),
#             "HS":  (os.path.join(HAM_DIR, "H2_Hund.txt"),       2),
#         },
#     },
#     # "HF": {
#     #     "symbols": ["H", "F"],
#     #     "coordinates": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.9168]).reshape(2, 3),
#     #     "exact_energy": -103.791389043825,       # FCI, STO-3G
#     #     "custom_encodings": {
#     #         "MHS": (os.path.join(HAM_DIR, "HF_central_MH.txt"), 3),
#     #         "HS":  (os.path.join(HAM_DIR, "HF_central.txt"),    5),
#     #     },
#     # },
#     # "LiH": {
#     #     "symbols": ["H", "Li"],
#     #     "coordinates": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5949]).reshape(2, 3),
#     #     "exact_energy": -8.87339056,             # FCI, STO-3G
#     #     "custom_encodings": {
#     #         "MHS": (os.path.join(HAM_DIR, "LiH_central_MH.txt"), 4),
#     #         "HS":  (os.path.join(HAM_DIR, "LiH_central.txt"),    7),
#     #     },
#     # },
# }

MOLECULES = {
    # MHS sectors. exact_energy is the electronic subspace ground state (E0 from
    # *_records.json); the .txt Hamiltonians are electronic-only, so no E_nuc.
    # "H2_w0": {
    #     "exact_energy": -1.8510241683485207,
    #     "custom_encodings": {"MHS": (os.path.join(HAM_DIR, "H2_MHS_omega0.txt"), 1)},
    # },
    # "HF_w0": {
    #     "exact_energy": -103.78651942347501,
    #     "custom_encodings": {"MHS": (os.path.join(HAM_DIR, "HF_MHS_omega0.txt"), 3)},
    # },
    # "HF_w2": {
    #     "exact_energy": -103.35696033951213,
    #     "custom_encodings": {"MHS": (os.path.join(HAM_DIR, "HF_MHS_omega2.txt"), 4)},
    # },
    # "LiH_w0": {
    #     "exact_energy": -8.873388605959928,
    #     "custom_encodings": {"MHS": (os.path.join(HAM_DIR, "LiH_MHS_omega0.txt"), 4)},
    # },
    # "LiH_w2": {
    #     "exact_energy": -8.761504365993739,
    #     "custom_encodings": {"MHS": (os.path.join(HAM_DIR, "LiH_MHS_omega2.txt"), 6)},
    # },
    # "LiH_w4": {
    #     "exact_energy": -6.698872338461917,
    #     "custom_encodings": {"MHS": (os.path.join(HAM_DIR, "LiH_MHS_omega4.txt"), 4)},
    # },
    # "OH_w1": {
    #     "exact_energy": -78.7449247887357,
    #     "custom_encodings": {"MHS": (os.path.join(HAM_DIR, "OH_MHS_omega1.txt"), 5)},
    # },
    # "OH_w3": {
    #     "exact_energy": -78.35908641062225,
    #     "custom_encodings": {"MHS": (os.path.join(HAM_DIR, "OH_MHS_omega3.txt"), 5)},
    # },
    "CH_w1": {
        "exact_energy": -40.62895820401409,
        "custom_encodings": {"MHS": (os.path.join(HAM_DIR, "CH_MHS_omega1.txt"), 6)},
    },
    "CH_w3": {
        "exact_energy": -40.63675349533744,
        "custom_encodings": {"MHS": (os.path.join(HAM_DIR, "CH_MHS_omega3.txt"), 6)},
    },
    "CH_w5": {
        "exact_energy": -40.003312975179895,
        "custom_encodings": {"MHS": (os.path.join(HAM_DIR, "CH_MHS_omega5.txt"), 3)},
    },

}

# ENCODINGS_TO_RUN = ["MHS"]
# LAYERS_TO_TEST = range(1, 8)
# OPTIMIZERS = ["adam"]
# NUM_RANDOM_TRIALS = 15
# L_NUM_RANDOM_TRIALS = range(7,NUM_RANDOM_TRIALS)

# Encodings to run: any of "JW", "MHS", "HS".
# ENCODINGS_TO_RUN = ["JW", "HS", "MHS"]
ENCODINGS_TO_RUN = ["MHS"]
OPTIMIZERS = ["lbfgs", "adam"]
LAYERS_TO_TEST = range(1, 8)
NUM_RANDOM_TRIALS = 15
L_NUM_RANDOM_TRIALS = range(NUM_RANDOM_TRIALS)
SUCCESS_THRESHOLD = 0.9


QUALITATIVE_ACCURACY = 0.01
CHEMICAL_ACCURACY = 1.6e-3

MAX_ITERATIONS = 1000
CONV_TOL = 1e-6
ADAM_LR = 0.1

RAW_TRIALS_CSV = "benchmark_raw_trials.csv"
SUMMARY_STATUS_CSV = "benchmark_summary_status.csv"
SUMMARY_DETAILED_CSV = "benchmark_summary_detailed.csv"


# --- 2. BENCHMARK ---
def run_benchmarks(encodings=ENCODINGS_TO_RUN):
    header = ['molecule', 'encoding', 'optimizer', 'layers', 'trial_num', 'seed',
              'final_energy', 'iterations', 'converged', 'wall_time_s', 'status']
    with open(RAW_TRIALS_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(header)

    for molecule_name, config in MOLECULES.items():
        print(f"\n--- Starting benchmark for {molecule_name} ---")

        # Assemble the requested Hamiltonians for this molecule.
        hamiltonians = {}
        if "JW" in encodings:
            jw_h, jw_q = qml.qchem.molecular_hamiltonian(config["symbols"],
                                                         config["coordinates"], unit="Angstrom")
            hamiltonians["JW"] = (jw_h, jw_q)
        for label, (fname, qubits) in config["custom_encodings"].items():
            if label in encodings:
                hamiltonians[label] = (qml_Hamli_from_file(fname), qubits)

        for encoding_name, (hamiltonian, qubits) in hamiltonians.items():
            for optimizer_name in OPTIMIZERS:
                print(f"Testing: Encoding={encoding_name}, Optimizer={optimizer_name}")
                for layers in LAYERS_TO_TEST:
                    print(f"  Trying {layers} layer(s)...")
                    for trial in L_NUM_RANDOM_TRIALS:
                        seed = trial  # 0..NUM_RANDOM_TRIALS-1; same seed value reused
                                      # across molecule/encoding/optimizer/layers so trial
                                      # index alone identifies the initialization used.
                        lr = ADAM_LR if optimizer_name == 'adam' else None
                        start_time = time.perf_counter()
                        energy_history, _, converged = VQE_HEA_RY(
                            Hamiltonian=hamiltonian, qubits=qubits, seed=seed,
                            layers=layers, optimizer=optimizer_name,
                            max_iterations=MAX_ITERATIONS, conv_tol=CONV_TOL,
                            learning_rate=lr, GPU=True,
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

                        with open(RAW_TRIALS_CSV, 'a', newline='') as f:
                            csv.writer(f).writerow(
                                [molecule_name, encoding_name, optimizer_name,
                                 layers, trial + 1, seed, f"{final_energy:.6f}",
                                 iterations, converged, f"{wall_time_s:.4f}", status])


# --- 3. ANALYSIS (csv-only, no pandas) ---
def analyze_and_save_summaries(raw_filename, molecule_to_analyze):
    def save_to_csv(filename, header, rows):
        with open(filename, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)

    try:
        with open(raw_filename, 'r') as f:
            all_trials = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"\nError: Raw data file '{raw_filename}' not found.")
        return

    trials = [t for t in all_trials if t['molecule'] == molecule_to_analyze]
    summary_status_list, summary_detailed_list = [], []

    groups = {}
    for trial in trials:
        groups.setdefault((trial['encoding'], trial['optimizer']), []).append(trial)

    exact = MOLECULES[molecule_to_analyze]['exact_energy']
    for (encoding, optimizer), group in groups.items():
        chem_found, qual_found = False, False
        for layers in LAYERS_TO_TEST:
            layer_trials = [t for t in group if int(t['layers']) == layers]
            if not layer_trials:
                continue
            chem = [t for t in layer_trials if t['status'] == "Chemically Accurate"]
            if not chem_found and len(chem) / NUM_RANDOM_TRIALS >= SUCCESS_THRESHOLD:
                summary_status_list.append({'Method': encoding, 'Layers': layers,
                                            'Optimizer': optimizer, 'Status': 'Chemically Accurate'})
                rep = chem[0]
                summary_detailed_list.append({
                    'Method': encoding, 'Layers': layers, 'Optimizer': optimizer,
                    'Final Energy (Ha)': rep['final_energy'],
                    'Error vs. Exact': f"{abs(float(rep['final_energy']) - exact):.6f}",
                    'Status': 'Chemically Accurate'})
                chem_found = True
            qual = [t for t in layer_trials if t['status'] in ("Qualitatively Accurate", "Chemically Accurate")]
            if not qual_found and len(qual) / NUM_RANDOM_TRIALS >= SUCCESS_THRESHOLD:
                summary_status_list.append({'Method': encoding, 'Layers': layers,
                                            'Optimizer': optimizer, 'Status': 'Qualitatively Accurate'})
                qual_found = True
        if not qual_found:
            summary_status_list.append({'Method': encoding, 'Layers': 'N/A',
                                        'Optimizer': optimizer, 'Status': 'Failure'})
        if not chem_found:
            best = min(group, key=lambda t: abs(float(t['final_energy']) - exact))
            summary_detailed_list.append({
                'Method': encoding, 'Layers': best['layers'], 'Optimizer': optimizer,
                'Final Energy (Ha)': best['final_energy'],
                'Error vs. Exact': f"{abs(float(best['final_energy']) - exact):.6f}",
                'Status': 'Failure (Local Minimum)'})

    save_to_csv(SUMMARY_STATUS_CSV, ['Method', 'Layers', 'Optimizer', 'Status'],
                summary_status_list)
    save_to_csv(SUMMARY_DETAILED_CSV,
                ['Method', 'Layers', 'Optimizer', 'Final Energy (Ha)', 'Error vs. Exact', 'Status'],
                summary_detailed_list)
    print(f"\nSummaries saved to '{SUMMARY_STATUS_CSV}' and '{SUMMARY_DETAILED_CSV}'")


if __name__ == "__main__":
    run_benchmarks()
    print(f"\nBenchmark complete. Raw data saved to {RAW_TRIALS_CSV}")
    for mol in MOLECULES:
        analyze_and_save_summaries(RAW_TRIALS_CSV, molecule_to_analyze=mol)