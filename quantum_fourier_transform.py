from linear_genetic_programming import ProblemParameters, Evolution, Genotype
from grid_search import grid_search

from qiskit.quantum_info import Operator, Statevector, random_statevector
from qiskit.circuit.library import QFT as QFT_blueprint

import math
def to_state(x):
    i = 0
    for j in range(len(x)):
        i += x[j] * 2**j
    return Statevector.from_int(i, 2**len(x))

def qft(state):
    '''returns quantum fourier transform applied to the state'''
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

class QFTGeneration(ProblemParameters):
    def __init__(self, set_of_gates, N=3, number_of_states_to_check=10):
        super().__init__(N, set_of_gates)

        self.input_states = [random_statevector(2**N) for _ in range(number_of_states_to_check)]
        #self.output_states = [qft(s) for s in self.input_states]
        self.output_states = [s.evolve(Operator(QFT_blueprint(N))) for s in self.input_states]

    def specific_msf(self, candidate_circuit):
        """overrides with the required truth table"""
        return self.msf(candidate_circuit, self.input_states, self.output_states)
    
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
    
    QFT_GEN = QFTGeneration(GATE_SET_SIMPLE, 3, 16)
    E = Evolution(QFT_GEN, individuals_per_generation=250, alpha=4, beta=6, gamma=3)

    """
    simple_set_qft_genotype = '00401442040141244208402402220202'#'004102420401421202'
    g = Genotype(QFT_GEN, simple_set_qft_genotype)
    print(g.to_circuit())
    print(g.get_msf())

    c = QFT_blueprint(3)
    print(c)
    print(QFT_GEN.specific_msf(c))
    """
        
    null_circuit_fitness = Genotype(QFT_GEN, '201201').get_msf()
    #population = E.evolutionary_search(min_length=10, max_length=25, MINIMUM_FITNESS=null_circuit_fitness,
    #                                   random_sample_size=25, remove_duplicates=True)
    
    grid_search(Evolution(QFT_GEN),lengths=([0,10,20,30],[10,20,25,30,40]),
                falloff=['linear','logarithmic','reciprocal'], iterations=1,
                MINIMUM_FITNESS=null_circuit_fitness, remove_duplicates=True ,random_sample_size=25)