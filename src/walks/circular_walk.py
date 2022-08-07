# -*- coding: utf-8 -*-
"""
Simulating a circular walk
"""

import time
from matplotlib import pyplot as plt
import qiskit.quantum_info as qi
import numpy as np
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit, transpile, Aer
from qiskit import QuantumRegister, ClassicalRegister

def simulate(circuit):
    backend = Aer.get_backend("statevector_simulator")
    job = backend.run(circuit)
    result = job.result()
    output = result.get_statevector(circuit, decimals=3)
    return np.array(output)

def normalize(vec):
    return vec / np.linalg.norm(vec)

n = 3
times = 4

qnodes = QuantumRegister(n,'qr')
qsubnodes = QuantumRegister(1,'qanc')
cnodes = ClassicalRegister(n,'cr')
csubnodes = ClassicalRegister(1,'canc')

circuit = QuantumCircuit(qnodes, qsubnodes, cnodes, csubnodes)
initial_state = [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
initial_state = normalize(initial_state)
circuit.initialize(initial_state, circuit.qubits)
print(circuit)


def increment(circuit, q, qanc):
    circuit.mcx([q[0], q[1], q[2]], qanc)
    circuit.mcx([q[1], q[2]], qanc)
    circuit.mcx([q[2]], qanc)
    circuit.barrier()
    return circuit

def decrement(circuit, q, qanc):
    circuit.x(qanc)
    circuit.x(q[2])
    circuit.x(q[1])

    circuit.mcx([q[0], q[1], q[2]], qanc)
    circuit.x(q[1])

    circuit.mcx([q[1], q[2]], qanc)
    circuit.x(q[2])

    circuit.mcx([q[2]], qanc)
    circuit.x(qanc)
    
    circuit.barrier()
    return circuit

def run_walk(circuit, times):
    for i in range(times):
        circuit.h(qsubnodes[0])
        increment(circuit, qnodes, qsubnodes[0])
        decrement(circuit, qnodes, qsubnodes[0])
    return circuit

circuit = run_walk(circuit, times)
print(circuit)

final_state = simulate(circuit)
print(final_state)

probs = [abs(i) ** 2 for i in final_state]
plt.plot(range(len(probs)), probs)
plt.xlabel("Position")
plt.ylabel("Probability")
plt.title("Distribution after t-walks")
plt.show()