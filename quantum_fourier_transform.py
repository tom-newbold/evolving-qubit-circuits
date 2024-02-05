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
    #input_states_sample = [random_statevector(2**N) for _ in range(number_of_states_to_check)]
    input_states_sample = [to_state([i//4 %2, i//2 %2, i%2]) for i in range(8)]
    output_states_sample = [s.evolve(Operator(QFT_blueprint(N))) for s in input_states_sample]
    #[qft(s) for s in input_states_sample]

    #return AppliedProblemParameters(set_of_gates, input_states_sample, output_states_sample, N)
    return AppliedProblemParameters(set_of_gates, input_states_sample, QFT_blueprint(N))
    #return ProblemParametersMatrix(set_of_gates, QFT_blueprint(N), N)
    return ProblemParametersCombined(set_of_gates, input_states_sample, QFT_blueprint(N), t)

    
if __name__=="__main__":
    from qiskit.circuit.library import *
    
    GATE_SET = [HGate(), XGate(), YGate(), ZGate(), CXGate(), PhaseGate(0), 
                RGate(0, 0), TGate(), TdgGate(), CHGate(), CPhaseGate(0)]
    #GATE_SET = {'α': XGate(), 'β': CXGate(), 'γa': YGate(), 'δ': HGate(), 'ε': PhaseGate(0), 'ζ': ZGate(),
    #            'η': RGate(0, 0), 'θ': TGate(), 'λa': TdgGate(), 'μ': CHGate(), 'φ': CPhaseGate(0)}

    QFT_GEN = QFTGeneration(GATE_SET, 3, t=0.5)#, 16)
    #E = Evolution(QFT_GEN, individuals_per_generation=300, alpha=3, beta=6, gamma=4)
    E = Evolution(QFT_GEN, sample_percentage=0.1, number_of_generations=25, individuals_per_generation=100, gen_mulpilier=2.5)
    
    null_f = QFT_GEN.get_null_circuit_fitness()
    population = E.evolutionary_search(min_length=10, max_length=25, MINIMUM_FITNESS=null_f,
                                       remove_duplicates=True, use_double_point_crossover=True)
    #random_sample_size=10

    
    #grid_search(Evolution(QFT_GEN), 1, [0, null_f], [0,2,5], [0.025,0.05,0.1], [0.025,0.1,0.5]) # try multiples of null fitness