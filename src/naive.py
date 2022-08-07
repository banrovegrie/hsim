# -*- coding: utf-8 -*-
"""
Authored by team: PAuliZee

Here, we are naively trying to simulate a Hamiltonian.
We are directly producing the matrix e^{-iHt} using
eigenvalue decomposition of the hamiltonian H.
"""

import time
from matplotlib import pyplot as plt
from helpers import error
import qiskit.quantum_info as qi
import numpy as np
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit, transpile, Aer


"""
Defining the Hamiltonian
H = \sum_j [X(j)X(j+1) + Y(j)Y(j+1) + Z(j)Z(j+1) + v(j)Z(j)]
where v(j) is a scalar uniformly random between [-1, 1]
"""


def hamiltonian(num_wires, noise=False):
    H = np.zeros((2**num_wires, 2**num_wires), dtype="complex")
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, complex(0, -1)], [complex(0, 1), 0]])
    z = np.array([[1, 0], [0, -1]])
    I = np.array([[1, 0], [0, 1]])

    for i in range(0, num_wires - 1):
        X = np.array([[1]])
        Y = np.array([[1]])
        Z = np.array([[1]])
        W = np.array([[1]])
        if noise:
            v = np.random.random() * 2 - 1
        else:
            v = 0

        for k in range(0, num_wires):
            if k == i:
                X = np.kron(X, x)
                Y = np.kron(Y, y)
                Z = np.kron(Z, z)
                W = np.kron(W, z)
            elif k == i + 1:
                X = np.kron(X, x)
                Y = np.kron(Y, y)
                Z = np.kron(Z, z)
                W = np.kron(W, I)
            else:
                X = np.kron(X, I)
                Y = np.kron(Y, I)
                Z = np.kron(Z, I)
                W = np.kron(W, I)

        H += X + Y + Z + v * W

    W = np.array([[1]])
    if noise:
        v = np.random.random() * 2 - 1
    else:
        v = 0
    for k in range(0, num_wires - 1):
        W = np.kron(W, I)
    W = np.kron(W, z)
    H += v * W
    return H


"""
Unitary for e^{-iHt}
"""


def unitarize(t, num_wires, noise=False):
    H = hamiltonian(num_wires, noise)
    eigenval, eigenvec = np.linalg.eig(H)
    return (
        eigenvec
        @ np.diag(np.exp(complex(0, -1) * t * eigenval))
        @ np.linalg.inv(eigenvec)
    )


"""
Defining the circuit
"""


def make_circuit(U, statevector, t, num_wires):
    circuit = QuantumCircuit(num_wires, num_wires)
    circuit.initialize(statevector, circuit.qubits)
    circuit.unitary(U, list(range(0, num_wires)), label=f"exp(i{t}H)")
    return circuit


"""
Simulate a given ciruit
"""


def simulate(circuit):
    backend = Aer.get_backend("statevector_simulator")
    job = backend.run(circuit)
    result = job.result()
    output = result.get_statevector(circuit, decimals=3)
    return np.array(output)


def simulate_measurement(circuit, shots=1024):
    backend = Aer.get_backend("qasm_simulator", shots=shots)
    job = backend.run(circuit)
    result = job.result()
    output = result.get_counts()
    return output


"""
Main execution
"""


def main(n=3, t=2, state=None):
    np.random.seed(42)
    start_time = time.time()

    if state == None:
        state = [0] * (2**n)
        state[0] = 1
        state = state / np.linalg.norm(state)
    num_wires = int(np.log2(len(state)))

    unitary = unitarize(t, num_wires)
    circ = make_circuit(unitary, state, t, num_wires)

    res = simulate(circ).tolist()
    return (time.time() - start_time), circ, res


"""
Performance comparision
"""


def performance():
    timestamps = []
    for n in range(2, 6):
        total_time, circ, res = main(n=n, t=2, state=None)
        timestamps.append(total_time)
    plt.plot(range(2, 6), timestamps)
    plt.xlabel("Number of qubits")
    plt.ylabel("Time")
    plt.title("Naive hamiltonian simulation")
    plt.show()


if __name__ == "__main__":
    performance()
