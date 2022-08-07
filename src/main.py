"""
Authored by team: PauliZee
"""
from typing import List
from helpers import get_probs
import numpy as np
from naive import unitarize, make_circuit, simulate, simulate_measurement
from trotter import construct_heisenberg
from helpers import error
from qiskit import IBMQ, execute
import timeit
import config as config
from qiskit.providers import provider
from matplotlib import pyplot as plt
import qiskit.quantum_info as qi

qubits_neighbours = [0, 1, 3, 5, 4]
t, noise = 2, np.random.uniform(-1, 1, 7)
noise = [0.0] * 7


def authenticate() -> provider.ProviderV1:
    IBMQ.save_account(config.TOKEN)
    IBMQ.load_account()
    provider = IBMQ.get_provider(
        hub=config.HUB, group=config.GROUP, project=config.PROJECT
    )
    return provider


def initialize(n):
    global noise, qubits_neighbours, t, r, num_qubits, state
    num_qubits = n
    qubits_neighbours = list(range(num_qubits))
    noise = [0.0] * num_qubits
    state = [0] * (2**num_qubits)
    state[0] = 1
    state = np.array(state)
    state = state / np.linalg.norm(state)


def get_matrix_rep(circuit):
    return qi.Operator(circuit)


def get_trotter_state(get_matrix=False) -> np.ndarray:
    circuit = construct_heisenberg(num_qubits, qubits_neighbours, t, r, noise, state)
    if get_matrix:
        matrix = get_matrix_rep(circuit)
    state_vector = simulate(circuit)
    return state_vector


def simulate_trotter(
    num_qubits, qubits_neighbours, t, r, noise, state, shots=1024
) -> np.ndarray:
    circuit = construct_heisenberg(num_qubits, qubits_neighbours, t, r, noise, state)

    counts = simulate_measurement(circuit, shots)
    print(f"classical measurements:{counts}")
    return counts


def get_naive_state(state, num_qubits, get_matrix=False) -> np.ndarray:
    unitary = unitarize(t, num_qubits)
    circ = make_circuit(unitary, state, t, num_qubits)
    if get_matrix:
        matrix = get_matrix_rep(circ)
    return simulate(circ)


def get_naive_unsimulated(state) -> np.ndarray:
    unitary = unitarize(t, num_qubits)
    result_state = unitary @ state
    return result_state


def run_circuit_on_quantum(circuit, backend, shots=1024):
    print("running circuit")
    result = execute(circuit, backend=backend, optimization_level=0, shots=shots)
    assert result is not None
    counts = result.result().get_counts(circuit)
    return counts


def get_probs_from_count(counts, num_qubits):
    probs = np.zeros(2**num_qubits)
    for key in counts.keys():
        key_val = int(key, 2)
        probs[key_val] = counts[key]
    probs = probs / np.sum(probs)
    return probs


def run_on_quantum_computer(
    num_qubits,
    qubits_neighbours,
    t,
    r,
    noise,
    state,
    backend_name,
    provider,
    shots=1024,
) -> np.ndarray:
    backend = provider.get_backend(backend_name)

    circuit = construct_heisenberg(num_qubits, qubits_neighbours, t, r, noise, state)
    # transpiled_circuit = transpile(circuit, backend, optimization_level=0)

    return run_circuit_on_quantum(circuit, backend, shots)
    # return run_circuit_on_quantum(transpiled_circuit, backend)


def compare_on_quantum_computer(
    range_qubits: List[int], run_quantum=False, shots=1024, r=1
) -> dict:
    print(f"running on r={r}")

    backend_name = "ibm_perth"
    provider = None
    if run_quantum:
        print("Queried for auth")
        provider = authenticate()
        print("Auth complete")

    all_probs_cc = []
    state_lists = []
    for num_qubits in range_qubits:
        state = np.random.rand(2**num_qubits)
        state = state / np.linalg.norm(state)
        state_lists.append(state)

    for ind, state_vec in enumerate(state_lists):
        num_qubits = range_qubits[ind]
        # cur_neighbours = qubits_neighbours[:num_qubits]
        cur_neighbours = list(range(num_qubits))

        probs_c = get_probs(get_naive_state(state_vec, num_qubits))
        all_probs_cc.append(probs_c)

    if not run_quantum:
        return {"cc": all_probs_cc}

    all_probs_qq = []
    all_probs_qc = []
    num_qubits = 7  # setting the number of qubits to the same as ibm_perth.
    for ind, state in enumerate(state_lists):
        num_qubits = range_qubits[ind]
        print(f"running for {num_qubits} on quantum computer")
        cur_neighbours = qubits_neighbours[:ind]
        probs_q = get_probs_from_count(
            run_on_quantum_computer(
                num_qubits,
                cur_neighbours,
                t,
                r,
                noise,
                state,
                backend_name,
                provider,
                shots=shots,
            ),
            num_qubits,
        )
        all_probs_qq.append(probs_q)
        probs_q = get_probs_from_count(
            simulate_trotter(num_qubits, cur_neighbours, t, r, noise, state), num_qubits
        )
        all_probs_qc.append(probs_q)

    return {"qq": all_probs_qq, "cc": all_probs_cc, "qc": all_probs_qc}


def normalize(vec):
    if np.linalg.norm(vec) != 0:
        return vec / np.linalg.norm(vec)
    else:
        return 0


def performance(l, r, is_normalize=False, get_matrix=False):
    timestamps_naive = []
    timestamps_trotter = []
    timestamps_unsimulated = []
    for n in range(l, r):
        initialize(n)
        start_time = timeit.default_timer()
        trotter_state = get_trotter_state(get_matrix)
        timestamps_trotter.append(timeit.default_timer() - start_time)

        start_time = timeit.default_timer()
        naive_state = get_naive_state(state, get_matrix)
        timestamps_naive.append(timeit.default_timer() - start_time)

        start_time = timeit.default_timer()
        naive_unsimulated = get_naive_unsimulated(state)
        timestamps_unsimulated.append(timeit.default_timer() - start_time)

    if is_normalize:
        timestamps_naive = normalize(timestamps_naive)
        timestamps_trotter = normalize(timestamps_trotter)
        timestamps_unsimulated = normalize(timestamps_unsimulated)
    plt.plot(range(l, r), timestamps_trotter, label="Trotter")
    plt.plot(range(l, r), timestamps_naive, label="Naive")
    plt.plot(range(l, r), timestamps_unsimulated, label="Unsimualted")
    plt.legend()
    plt.xlabel("Number of qubits")
    plt.ylabel("Time")
    plt.title("Hamiltonian Simulation")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    performance(3, 10, is_normalize=False, get_matrix=False)
