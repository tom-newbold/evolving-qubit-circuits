from qiskit.circuit.library import QFT as QFT_blueprint
from qiskit.circuit.library import *

from linear_genetic_programming import AppliedProblemParameters, Evolution
    
GATE_SET = [HGate(), XGate(), YGate(), ZGate(), CXGate(), PhaseGate(0), 
            RGate(0, 0), TGate(), TdgGate(), CHGate(), CPhaseGate(0)]
GATE_SET_SIMPLE = [HGate(), CXGate(), TGate(), TdgGate()]

def QFTGeneration(set_of_gates, N=3):
    '''creates a ProblemParameters object with the desired input and output states,
       a sample of the specified size generated based on the number of qubits'''
    return AppliedProblemParameters(set_of_gates, QFT_blueprint(N),
                                    genotype_len_bounds=[10,25],
                                    genotype_length_falloff='linear')
    
if __name__=="__main__":
    print(QFT_blueprint(3).decompose().draw('text'))
    QFT_GEN = QFTGeneration(GATE_SET, 3)
    QFT_GEN.print_gate_set()

    E = Evolution(QFT_GEN, sample_percentage=0.1, gen_mulpilier=6)
    
    null_f = QFT_GEN.get_null_circuit_fitness()
    E.stochastic_hill_climb()
    population = E.evolutionary_search(MINIMUM_FITNESS=min(null_f, 0),
                                       use_double_point_crossover=True)[0]
    #random_sample_size=10
        