import time
from matplotlib import pyplot as plt
import qiskit.quantum_info as qi
import numpy as np
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit, transpile, Aer
from qiskit import QuantumRegister, ClassicalRegister


def normalize_vec(vec):
    val = np.linalg.norm(vec)
    if val > 0:
        return vec / val
    return vec


def normalize_mat(mat):
    new_mat = mat.copy()
    n, m = new_mat.shape
    for i in range(m):
        if np.sum(new_mat[:, i]) > 0:
            new_mat[:, i] /= np.sum(new_mat[:, i])
    return new_mat


def get_vector(j, n):
    vec = np.zeros((n), dtype="float")
    vec[j] = 1
    return vec


def U(P):
    N = P.shape[1]
    new_P = np.sqrt(P)
    psi = [None for _ in range(N)]
    for j in range(N):
        arr = normalize_vec(
            np.array(sum([new_P[j][k]*get_vector(k, N) for k in range(N)])))
        psi[j] = np.array([np.kron(get_vector(j, N), arr)])
    Pi = sum([np.outer(psi[i], psi[i].conj()) for i in range(N)])
    S = sum(
        [
            sum(
                [
                    np.outer(
                        np.kron(get_vector(j, N), get_vector(k, N)),
                        np.kron(get_vector(k, N), get_vector(j, N))) for k in range(N)
                ]) for j in range(N)
        ]
    )
    U = S @ (2*Pi-np.eye(N**2))
    return U


def is_unitary(m):
    return np.allclose(np.eye(len(m)), m.dot(m.T.conj()))


if __name__ == "__main__":
    H = normalize_mat(np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ], dtype="float"
    ))
    u = U(H)
    print(u)
    print(is_unitary(u))
