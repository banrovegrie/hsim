# Fast Hamiltonian Simulation

The problem of efficiently simulating a Hamiltonian is one of the long standing problems in computational physics. Today, the problem of Hamiltonian simulation stands as one of the most impacting and significant contribution of quantum computers. Furthermore since it is a BQP-complete problem, devising any efficient classical algorithm for the same is believed to be intractable.

We simulated the Hamiltonian of the Heisenberg Spin Chain model using topologically optimized Trotterisation on a 7-qubit IBM quantum computer. We then compare the results of this simulation with purely-classical Hamiltonian simulation and classically Trotterised Hamiltonian simulation.

![comparision](images/benchmarking.png)

### References

1. Richard P Feynman, [Simulating physics with computers](http://www.sciencemag.org/cgi/content/abstract/273/5278/1073), International Journal of Theoretical Physics, 1982.
    
1. Andrew M. Childs, [On the relationship between continuous- and discrete-time quantum walk](https://arxiv.org/abs/0810.0312), Communications in Mathematical Physics 294, 581-603 (2010).
    
1. Dominic W. Berry, Andrew M. Childs, [Black-box Hamiltonian simulation and unitary implementation](https://arxiv.org/abs/0910.4157), Quantum Information and Computation 12, 29 (2012).

1. Andrew M. Childs, [Theory of Trotter Error with Commutator Scaling
](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.011020), Phys. Rev. X 11, 011020 (2021).
    
1. Guang Hao Low and Isaac L. Chuang, [Optimal Hamiltonian Simulation by Quantum Signal Processing](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.010501), Phys. Rev. Lett. 118, 010501 (2017).
    
1. Guang Hao Low and Isaac Chuang, [Hamiltonian Simulation by Qubitization](https://quantum-journal.org/papers/q-2019-07-12-163), Quantum 3, 163 (2019).
    
1. L. Clinton, J. Bausch and T. Cubitt, [Hamiltonian simulation algorithms for near-term quantum hardware](https://www.nature.com/articles/s41467-021-25196-0), Nature Communications volume 12, article 4989 (2021).
    
1. Yulong Dong, K. Birgitta Whaley and Lin Lin, [A Quantum Hamiltonian Simulation Benchmark](https://arxiv.org/pdf/2108.03747.pdf).
