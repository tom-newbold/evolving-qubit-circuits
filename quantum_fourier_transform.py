from linear_genetic_programming import AppliedProblemParameters, Evolution, Genotype, ProblemParametersMatrix
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

def QFTGeneration(set_of_gates, N=3, number_of_states_to_check=10):
    '''creates a ProblemParameters object with the desired input and output states,
       a sample of the specified size generated based on the number of qubits'''
    input_states_sample = [random_statevector(2**N) for _ in range(number_of_states_to_check)]
    #[to_state([i//4 %2, i//2 %2, i%2]) for i in range(8)]
    output_states_sample = [s.evolve(Operator(QFT_blueprint(N))) for s in input_states_sample]
    #[qft(s) for s in input_states_sample]

    #return AppliedProblemParameters(set_of_gates, input_states_sample, output_states_sample, N)
    return ProblemParametersMatrix(set_of_gates, QFT_blueprint(N), N)

    
if __name__=="__main__":
    GATE_SET_SIMPLE = [{'label':'had','inputs':1},
                       {'label':'chad','inputs':2},
                       {'label':'cnot','inputs':2},
                       {'label':'phase','inputs':1,'parameters':1},
                       {'label':'cphase','inputs':2,'parameters':1}]
                              
    GATE_SET = [{'label':'had','inputs':1},
                {'label':'not','inputs':1},
                {'label':'cnot','inputs':2},
                {'label':'phase','inputs':1,'parameters':1},
                {'label':'t','inputs':1},
                {'label':'t_prime','inputs':1},
                {'label':'chad','inputs':2},
                {'label':'cphase','inputs':2,'parameters':1}]
    

    QFT_GEN = QFTGeneration(GATE_SET, 3, 16)
    E = Evolution(QFT_GEN, individuals_per_generation=300, alpha=4, beta=6, gamma=3)

    """
    simple_set_qft_genotype = '024122014024401200202220202'
    g = Genotype(QFT_GEN, simple_set_qft_genotype)
    states = [random_statevector(2**3) for _ in range(30)]
    fitness_from_sample = QFT_GEN.msf(g.to_circuit(), states, [s.evolve(Operator(QFT_blueprint(3))) for s in states])
    print(f'{100 * fitness_from_sample / QFT_GEN.specific_msf(QFT_blueprint(3))} %')
    """

        
    null_circuit_fitness = Genotype(QFT_GEN, '201201').get_msf()
    population = E.evolutionary_search(min_length=10, max_length=25, MINIMUM_FITNESS=null_circuit_fitness,
                                       random_sample_size=25, remove_duplicates=True, use_double_point_crossover=True)
    
    print(f'Best generated / Best Possible: {100 * population[0].msf / QFT_GEN.specific_msf(QFT_blueprint(3))} %')
    states = [random_statevector(2**3) for _ in range(50)]
    fitness_from_sample = QFT_GEN.msf(population[0].to_circuit(), states, [s.evolve(Operator(QFT_blueprint(3))) for s in states])
    print(f'From sample percentage: {100 * fitness_from_sample / QFT_GEN.specific_msf(QFT_blueprint(3))} %')
    
    
    #grid_search(Evolution(QFT_GEN),lengths=([0,10,20,30],[10,20,25,30,40]),
    #            falloff=['linear','logarithmic','reciprocal'], iterations=1,
    #            MINIMUM_FITNESS=null_circuit_fitness, remove_duplicates=True, random_sample_size=25)