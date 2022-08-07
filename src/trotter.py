import numpy as np
from qiskit.providers.aer import QasmSimulator
from qiskit import QuantumCircuit, transpile, Aer
from typing import List


def apply_one_noise(circuit: QuantumCircuit, qubit: int, angle: float) -> None:
    """
    Applies noise to given qubits.
    Returns: None
    """
    circuit.rz(angle, qubit)


def apply_one_hamiltonian(
    circuit: QuantumCircuit, qubits: List[int], angle: float
) -> None:
    """
    Applies the required actual gates part of the Hamiltonian for the given qubit
    Returns: None
    """
    circuit.rxx(angle, *qubits)
    circuit.ryy(angle, *qubits)
    circuit.rzz(angle, *qubits)


def construct_heisenberg(
    num_qubits: int,
    qubits_neighbours: List[int],
    time: float,
    r: float,
    noise: List[float],
    initial_state: np.ndarray,
) -> QuantumCircuit:
    """
    Takes in the quantum computer's neighbouring qubits and constructs the Hamiltonian.

    Returns: QuantumCircuit.
    """
    circuit = QuantumCircuit(num_qubits, num_qubits)
    circuit.initialize(initial_state, qubits=circuit.qubits)

    coeff = 2 * (time / r)
    for _ in np.arange(time / r, time + (time / r), time / r):
        # iterating over time steps
        for ind, cur_qubit in enumerate(qubits_neighbours[:-1]):
            # Iterating over the neighbours
            apply_one_hamiltonian(
                circuit, [cur_qubit, qubits_neighbours[ind + 1]], coeff
            )
        for ind, cur_qubit in enumerate(qubits_neighbours):
            apply_one_noise(circuit, cur_qubit, coeff * noise[ind])

    circuit.measure(list(range(num_qubits)), list(range(num_qubits)))
    return circuit


if __name__ == "__main__":
    qubits_neighbours = [0, 1, 2]
    time, r, noise = 2, 10000, np.random.uniform(-1, 1, 7)
    noise = [0.0] * 3
    state = np.array([0, 1, 0, 0, 0, 0, 1, 0])
    state = state / np.linalg.norm(state)
    circuit = construct_heisenberg(3, qubits_neighbours, time, r, noise, state)
