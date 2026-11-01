# Code by Calvin Ku; original repository: https://github.com/calvinku96/subspace_restriction (commit 8f4143af24c433cf7ea0b4d837fe05ac463e1644)
import jax
from jax import numpy as np

# jax.config.update("jax_platform_name", "cpu")


# BACKEND = jax.default_backend()
BACKEND = "cpu"
print(f"Backend: {BACKEND} (set in tool.py)")
jax.config.update("jax_enable_x64", True)

import optax
import pennylane as qml
from pennylane import numpy as np


def Pauli2qml(String, position):
    Pauli2qml = {
        "I": qml.Identity(position),
        "X": qml.PauliX(position),
        "Y": qml.PauliY(position),
        "Z": qml.PauliZ(position),
    }
    return Pauli2qml[String]


def EncodedHamiltonian2qml(Dic):
    coeffs = []
    obs = []
    for key in Dic:
        for i in range(0, len(key)):
            if i == 0:
                temp = Pauli2qml(key[i], 0)
            else:
                temp = Pauli2qml(key[i], i) @ temp
        # coeffs.append(complex(Dic[key]))
        coeffs.append(float(Dic[key]))
        obs.append(temp)
    return coeffs, obs


def qml_Hamli_from_file(name):
    # with open('LiH_Hund.txt', 'r') as f:
    with open(name, "r") as f:
        lines = f.readlines()

    dic = {}
    for line in lines:
        key, value = line.strip().split(":")
        dic[key] = value

    return qml.Hamiltonian(
        EncodedHamiltonian2qml(dic)[0], EncodedHamiltonian2qml(dic)[1]
    )


def VQE_HEA_RY(
    Hamiltonian,
    qubits,
    seed=None,
    layers=10,
    show=False,
    max_iterations=400,
    conv_tol=1e-06,
    optimizer="adagrad",
    GPU=False,
    learning_rate=0.5,
):
    import numpy as onp

    rng = onp.random.default_rng(seed)
    theta = np.array(rng.uniform(0, 2 * onp.pi, size=qubits * (layers + 1)))

    def circuit(param, wires, layer=layers):
        # Reshape parameters for easier access: (layers, num_gate_types, qubits)
        # Each layer has RZ and RY gates, so 2 types.
        # p = param.reshape(layer, 2, wires)
        p = param.reshape(layer + 1, 1, wires)

        for k in range(layer):  # Loop over layers
            # Apply RZ gates
            for i in range(wires):
                # Apply RZ gates
                # qml.RZ(p[k, 0, i], wires=i)
                # Apply RY gates
                qml.RY(p[k, 0, i], wires=i)

            # Entangling block
            for i in range(wires - 1):
                qml.CNOT(wires=[i, i + 1])
            # if wires > 1: # Add cyclic entanglement
            #     qml.CNOT(wires=[wires - 1, 0])

        for i in range(wires):
            qml.RY(p[-1, 0, i], wires=i)

    # print circuits
    if show:
        drawer = qml.draw(circuit)
        print(drawer(theta, qubits, layer=layers))

    global BACKEND

    # define device and cost function
    if BACKEND == "gpu" and GPU:
        print("lightning.gpu")
        dev = qml.device("lightning.gpu", wires=qubits)

    else:
        dev = qml.device("lightning.qubit", wires=qubits)
    dev = qml.device("lightning.qubit", wires=qubits)

    @jax.jit
    @qml.qnode(dev, interface="jax", diff_method="adjoint")
    def cost_fn(param):
        circuit(param, wires=qubits)
        return qml.expval(Hamiltonian)

    # VQE main process
    if optimizer == "lbfgs":
        print("using lbfgs")
        opt = optax.lbfgs()

    elif optimizer == "adam":
        print(f"using adam {learning_rate}")
        opt = optax.adam(learning_rate=learning_rate)

    elif optimizer == "adagrad":
        print(f"using adagrad {learning_rate}")
        opt = optax.adagrad(learning_rate=learning_rate)

    else:
        opt = optax.sgd(learning_rate=learning_rate)
        print("using sgd")

    energy = [cost_fn(theta)]
    angle = [theta]
    opt_state = opt.init(theta)

    @jax.jit
    def optimization_step(params, opt_state, opti=optimizer, opt=opt):
        if opti != "lbfgs":
            # Calculate gradients using the cost function (qnode)
            grads = jax.grad(cost_fn)(params)

            # Get updates from the optimizer
            updates, new_opt_state = opt.update(grads, opt_state)

            # Apply updates to the parameters
            new_params = optax.apply_updates(params, updates)

        else:
            value, grad = optax.value_and_grad_from_state(cost_fn)(
                params, state=opt_state
            )
            updates, new_opt_state = opt.update(
                grad, opt_state, params, value=value, grad=grad, value_fn=cost_fn
            )

            new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    ############################

    for n in range(max_iterations):
        theta, opt_state = optimization_step(theta, opt_state)

        angle.append(theta)
        energy.append(cost_fn(theta))

        conv = np.abs(energy[-1] - energy[-2])

        if n % 5 == 0:
            print(f"Step = {n:>5},  Energy = {energy[-1]:.8f} Ha")

        if conv <= conv_tol:
            break

    converged = bool(np.abs(energy[-1] - energy[-2]) <= conv_tol)
    if not converged:
        print("Not converged")

    print(f"\nFinal value of the ground-state energy = {energy[-1]:.8f} Ha")
    print(f"\nOptimal value of the circuit parameter =\n {np.round(angle[-1], 3)}")

    return energy, angle, converged


def VQE_UCCSD(
    Hamiltonian,
    qubits,
    electrons,
    seed=None,
    show=False,
    max_iterations=400,
    conv_tol=1e-06,
    optimizer="adagrad",
    GPU=False,
    learning_rate=0.5,
):
    # 1. Define excitation indices & initial parameters
    s_excitations, d_excitations = qml.qchem.excitations(electrons, qubits)
    s_wires, d_wires = qml.qchem.excitations_to_wires(s_excitations, d_excitations)
    singles_count = len(s_excitations)
    doubles_count = len(d_excitations)
    num_params = singles_count + doubles_count

    rng = np.random.default_rng(seed)
    theta = np.array(rng.uniform(-0.01, 0.01, size=num_params))

    # Initial reference state (Hartree-Fock)
    hf_state = qml.qchem.hf_state(electrons, qubits)

    # 2. Define the UCCSD circuit
    def circuit(params):
        qml.UCCSD(
            weights=params,
            wires=range(qubits),
            s_wires=s_wires,
            d_wires=d_wires,
            init_state=hf_state,
        )

    if show:
        drawer = qml.draw(circuit)
        print(drawer(theta))

    # 3. Device selection
    if BACKEND == "gpu" and GPU:
        dev = qml.device("lightning.gpu", wires=qubits)
    else:
        dev = qml.device("lightning.qubit", wires=qubits)

    @jax.jit
    @qml.qnode(dev, interface="jax", diff_method="adjoint")
    def cost_fn(param):
        circuit(param)
        return qml.expval(Hamiltonian)

    # 4. Optimization process
    if optimizer == "lbfgs":
        print("using lbfgs")
        opt = optax.lbfgs()
    elif optimizer == "adam":
        print(f"using adam {learning_rate}")
        opt = optax.adam(learning_rate=learning_rate)
    elif optimizer == "adagrad":
        print(f"using adagrad {learning_rate}")
        opt = optax.adagrad(learning_rate=learning_rate)
    else:
        opt = optax.sgd(learning_rate=learning_rate)
        print("using sgd")

    energy = [cost_fn(theta)]
    angle = [theta]
    opt_state = opt.init(theta)

    @jax.jit
    def optimization_step(params, opt_state, opti=optimizer, opt=opt):
        if opti != "lbfgs":
            grads = jax.grad(cost_fn)(params)
            updates, new_opt_state = opt.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
        else:
            value, grad = optax.value_and_grad_from_state(cost_fn)(
                params, state=opt_state
            )
            updates, new_opt_state = opt.update(
                grad, opt_state, params, value=value, grad=grad, value_fn=cost_fn
            )
            new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state

    for n in range(max_iterations):
        theta, opt_state = optimization_step(theta, opt_state)

        angle.append(theta)
        energy.append(cost_fn(theta))

        conv = np.abs(energy[-1] - energy[-2])

        if n % 5 == 0:
            print(f"Step = {n:>5},  Energy = {energy[-1]:.8f} Ha")

        if conv <= conv_tol:
            break

    converged = bool(np.abs(energy[-1] - energy[-2]) <= conv_tol)
    if not converged:
        print("Not converged")

    print(f"\nFinal value of the ground-state energy = {energy[-1]:.8f} Ha")
    print(f"\nOptimal value of the circuit parameter =\n {np.round(angle[-1], 3)}")

    return energy, angle, converged
