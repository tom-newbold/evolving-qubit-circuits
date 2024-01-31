from linear_genetic_programming import AppliedProblemParameters, Evolution, Genotype, ProblemParametersMatrix, ProblemParametersCombined
from grid_search import grid_search

from qiskit.quantum_info import Operator, Statevector, random_statevector
from qiskit.circuit.library import QFT as QFT_blueprint

import math
def to_state(x):
    '''DEPRECATED'''
    i = 0
    for j in range(len(x)):
        i += x[j] * 2**j
    return Statevector.from_int(i, 2**len(x))

def qft(state):
    '''DEPRECATED: returns quantum fourier transform applied to the state'''
    N = 2**state.num_qubits
    y = []
    for k in range(N):
        probability = 0
        for i in range(N):
            alpha = state.probabilities()[i]
            probability += alpha * (math.e**(2j*math.pi/N))**(i*k)
        probability /= math.sqrt(N)
        y.append(probability)
    return Statevector(y)

def QFTGeneration(set_of_gates, N=3, number_of_states_to_check=10, t=0.05):
    '''creates a ProblemParameters object with the desired input and output states,
       a sample of the specified size generated based on the number of qubits'''
    input_states_sample = [random_statevector(2**N) for _ in range(number_of_states_to_check)]
    #[to_state([i//4 %2, i//2 %2, i%2]) for i in range(8)]
    output_states_sample = [s.evolve(Operator(QFT_blueprint(N))) for s in input_states_sample]
    #[qft(s) for s in input_states_sample]

    #return AppliedProblemParameters(set_of_gates, input_states_sample, output_states_sample, N)
    #return ProblemParametersMatrix(set_of_gates, QFT_blueprint(N), N)
    return ProblemParametersCombined(set_of_gates, input_states_sample, QFT_blueprint(N), t)

    
if __name__=="__main__":
    from qiskit.circuit.library import *

    GATE_SET_SIMPLE = [HGate(), CHGate(), CXGate(), PhaseGate(0), CPhaseGate(0)]
    
    GATE_SET = [HGate(), XGate(), YGate(), CXGate(), PhaseGate(0), RGate(0, 0), 
                TGate(), TdgGate(), CHGate(), CPhaseGate(0)]
    GATE_SET = {'A': HGate(), 'B': XGate(), 'C': YGate(), 'D': ZGate(), 'E': CXGate(), 'F': PhaseGate(0),
                'G': RGate(0, 0), 'H': TGate(), 'I': TdgGate(), 'J': CHGate(), 'K': CPhaseGate(0)}

    QFT_GEN = QFTGeneration(GATE_SET, 3, t=0.5)#, 16)
    #E = Evolution(QFT_GEN, individuals_per_generation=300, alpha=3, beta=6, gamma=4)
    E = Evolution(QFT_GEN)
        
    null_circuit_fitness = Genotype(QFT_GEN, '').get_msf()
    population = E.evolutionary_search(min_length=10, max_length=25, MINIMUM_FITNESS=null_circuit_fitness,
                                       random_sample_size=10, remove_duplicates=True, use_double_point_crossover=True)

    
    
    #grid_search(Evolution(QFT_GEN),lengths=([0,10,20,30],[10,20,25,30,40]),
    #            falloff=['linear','logarithmic','reciprocal'], iterations=1,
    #            MINIMUM_FITNESS=null_circuit_fitness, remove_duplicates=True, random_sample_size=25)