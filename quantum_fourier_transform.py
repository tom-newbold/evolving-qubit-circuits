from qiskit.circuit.library import QFT as QFT_blueprint
from qiskit.circuit.library import *

from linear_genetic_programming import AppliedProblemParameters, Evolution
from bulk_runs import multiple_runs
    
#GATE_SET = [HGate(), XGate(), YGate(), ZGate(), CXGate(), PhaseGate(0), 
#            RGate(0, 0), TGate(), TdgGate(), CHGate(), CPhaseGate(0),
#            RYGate(0), RYGate(0).control()]
GATE_SET = [HGate(), XGate(), CXGate(), PhaseGate(0),
            TGate(), TdgGate(), CPhaseGate(0), RYGate(0).control()]
GATE_SET_SIMPLE = [HGate(), CXGate(), TGate(), TdgGate()]
from toffoli_gate_generation import UNIVERSAL_GATE_SET

def QFTGeneration(set_of_gates=GATE_SET, N=3):
    '''creates a ProblemParameters object with the desired input and output states,
       a sample of the specified size generated based on the number of qubits'''
    lower_bound = N**2 + N
    return AppliedProblemParameters(set_of_gates, QFT_blueprint(N),
                                    genotype_len_bounds=[2*lower_bound, 4*lower_bound],
                                    genotype_length_falloff='linear')
    
if __name__=="__main__":
    N=3
    #print(QFT_blueprint(3).decompose().draw('text'))
    QFT_GEN = QFTGeneration(GATE_SET, N)
    QFT_GEN.print_gate_set()

    E = Evolution(QFT_GEN, individuals_per_generation=25 * 2**(N-1), number_of_generations=10,
                  sample_percentage=0.1, gen_mulpilier=8, beta=4)
    
    
    #null_f = QFT_GEN.get_null_circuit_fitness()
    #MINIMUM_FITNESS=min(null_f, 0),
    population = E.evolutionary_search(use_double_point_crossover=True, plot_fitness=False)[0]#,plot_fitness=False

    '''

    to_plot, stats = multiple_runs(E, iterations=10)#, MINIMUM_FITNESS=min(null_f, 0))
    ##plot_many_averages(to_plot, 'Generations', 'Circuit Fitness', legend=False, reference_line=(2**N-1)/(2**N))
    print(f"convergence: {stats['generations_taken_to_converge']}")
    '''