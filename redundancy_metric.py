from linear_genetic_programming import Genotype, ProblemParameters
from quantum_fourier_transform import GATE_SET
from toffoli_gate_generation import ToffoliGeneration

if __name__=="__main__":
    app = ToffoliGeneration(GATE_SET)
    app.print_gate_set()
    g = Genotype(app, '000102')#A1A1C04A2B1A2C04E02B1E02
    c = Genotype.static_remove_redundant_gates(g.to_circuit(), True)
    print(len(c.data))
    #g.from_circuit(c)

    #import numpy as np
    #for i in range(10):
    #    print('-----')
    #    g = Genotype(app)
    #    print(np.abs(np.trace(np.matmul(Operator(g.to_circuit()),app.M_in_tr))/(2**3)) > g.get_fitness())