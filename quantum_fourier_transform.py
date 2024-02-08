from linear_genetic_programming import AppliedProblemParameters, Evolution, ProblemParametersMatrix, ProblemParametersCombined

from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT as QFT_blueprint

def QFTGeneration(set_of_gates, N=3, t=0.05):
    '''creates a ProblemParameters object with the desired input and output states,
       a sample of the specified size generated based on the number of qubits'''
    input_states_sample = [Statevector.from_int(i, 2**N) for i in range(2**N)]

    return AppliedProblemParameters(set_of_gates, input_states_sample, QFT_blueprint(N)) #TODO comment line test new fitness...
    #return ProblemParametersMatrix(set_of_gates, QFT_blueprint(N), N)
    return ProblemParametersCombined(set_of_gates, input_states_sample, QFT_blueprint(N), t)

from qiskit.circuit.library import *
    
GATE_SET = [HGate(), XGate(), YGate(), ZGate(), CXGate(), PhaseGate(0), 
            RGate(0, 0), TGate(), TdgGate(), CHGate(), CPhaseGate(0)]
    
if __name__=="__main__":
    QFT_GEN = QFTGeneration(GATE_SET, 3, t=0.5)#, 16)
    #E = Evolution(QFT_GEN, individuals_per_generation=300, alpha=3, beta=6, gamma=4)
    E = Evolution(QFT_GEN, sample_percentage=0.1, gen_mulpilier=2.5)
    
    null_f = QFT_GEN.get_null_circuit_fitness()
    population = E.evolutionary_search(min_length=10, max_length=25, MINIMUM_FITNESS=min(null_f, 0),
                                       remove_duplicates=True, use_double_point_crossover=True)
    #random_sample_size=10