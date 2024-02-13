from linear_genetic_programming import AppliedProblemParameters, Evolution, ProblemParametersCombined, plot_many_averages

#from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT as QFT_blueprint
from qiskit.circuit.library import *
    
GATE_SET = [HGate(), XGate(), YGate(), ZGate(), CXGate(), PhaseGate(0), 
            RGate(0, 0), TGate(), TdgGate(), CHGate(), CPhaseGate(0)]

def QFTGeneration(set_of_gates, N=3, t=0.05):
    '''creates a ProblemParameters object with the desired input and output states,
       a sample of the specified size generated based on the number of qubits'''
    return AppliedProblemParameters(set_of_gates, QFT_blueprint(N)) # just msf
    #return ProblemParametersCombined(set_of_gates, QFT_blueprint(N), mdf_tolerance=t) # msf and mdf
    
if __name__=="__main__":
    QFT_GEN = QFTGeneration(GATE_SET, 3, t=0.5)#, 16)
    #E = Evolution(QFT_GEN, individuals_per_generation=300, alpha=3, beta=6, gamma=4)
    E = Evolution(QFT_GEN, sample_percentage=0.1, gen_mulpilier=10)
    
    null_f = QFT_GEN.get_null_circuit_fitness()
    population = E.evolutionary_search(min_length=10, max_length=25, MINIMUM_FITNESS=min(null_f, 0),
                                       remove_duplicates=True, use_double_point_crossover=True)[0]
    #random_sample_size=10
        