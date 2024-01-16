from linear_genetic_programming import ProblemParameters, Evolution

from qiskit.quantum_info import Statevector, random_statevector

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

class QFT3Generation(ProblemParameters):
    def __init__(self, set_of_gates, number_of_states_to_check=10):
        super().__init__(3, set_of_gates)

        self.input_states = [random_statevector(2**3) for _ in range(number_of_states_to_check)]
        self.output_states = [qft(s) for s in self.input_states]
    
    def specific_msf(self, candidate_circuit):
        """overrides with the required truth table"""
        return self.msf(candidate_circuit, self.input_states, self.output_states)
    
if __name__=="__main__":
    GATE_SET_SIMPLE = [{'label':'had','inputs':1},
                       {'label':'cnot','inputs':2},
                       {'label':'t','inputs':1},
                       {'label':'t_prime','inputs':1}]
                              
    GATE_SET = [{'label':'had','inputs':1},
                {'label':'not','inputs':1},
                {'label':'cnot','inputs':2},
                {'label':'phase','inputs':1,'parameters':1},
                {'label':'t','inputs':1},
                {'label':'t_prime','inputs':1},
                {'label':'chad','inputs':2},
                {'label':'cphase','inputs':2,'parameters':1}]
    
    QFT_GEN = QFT3Generation(GATE_SET, 20)
    E = Evolution(QFT_GEN, sample=50)
    
    #population = E.random_search()
    #population = E.stochastic_hill_climb()
    population = E.evolutionary_search(MINIMUM_FITNESS=0.05)