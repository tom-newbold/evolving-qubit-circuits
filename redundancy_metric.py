from linear_genetic_programming import Genotype, ProblemParameters
from quantum_fourier_transform import GATE_SET
from toffoli_gate_generation import ToffoliGeneration

if __name__=="__main__":
    app = ToffoliGeneration(GATE_SET)
    app.print_gate_set()
    g = Genotype(app, 'A1A1F04A2B1A2F04E02B1E02')#A1A1A2B1A2C02B1C02
    print(f'fitness: {g.get_fitness()}')
    import numpy as np
    from qiskit.quantum_info import Operator
    c = Genotype.static_remove_redundant_gates(g.to_circuit())
    g.from_circuit(c)
    print(f'fitness: {g.get_fitness()}')

    for i in range(10):
        print('-----')
        g = Genotype(app)
        print(np.abs(np.trace(np.matmul(Operator(g.to_circuit()),app.M_in_tr))/(2**3)) > g.get_fitness())