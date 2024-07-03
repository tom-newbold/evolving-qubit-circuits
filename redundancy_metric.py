from linear_genetic_programming import Genotype, ProblemParameters
from quantum_fourier_transform import GATE_SET
from toffoli_gate_generation import ToffoliGeneration

if __name__=="__main__":
    app = ToffoliGeneration(GATE_SET)
    app.print_gate_set()
    g = Genotype(app, 'F17E01K027K027K122D0E20F14D0E02H0H0A0A1I126I126I126A2')#A1A1F04A2B1A2F04E02B1E02
    c = Genotype.static_remove_redundant_gates(g.to_circuit(), True)
    #g.from_circuit(c)

    #import numpy as np
    #for i in range(10):
    #    print('-----')
    #    g = Genotype(app)
    #    print(np.abs(np.trace(np.matmul(Operator(g.to_circuit()),app.M_in_tr))/(2**3)) > g.get_fitness())