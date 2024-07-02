from qiskit.circuit.library import QFT as QFT_blueprint
from qiskit.circuit.library import *

from linear_genetic_programming import AppliedProblemParameters, Evolution, Genotype
from linear_genetic_programming_utils import plot_many_averages, list_avr
from bulk_runs import multiple_runs, remaining_time_calc
    
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
    N=3
    #print(QFT_blueprint(3).decompose().draw('text'))
    QFT_GEN = QFTGeneration(GATE_SET, N)
    QFT_GEN.print_gate_set()

    E = Evolution(QFT_GEN, number_of_generations=100, sample_percentage=0.1, gen_mulpilier=8, beta=5)
    
    null_f = QFT_GEN.get_null_circuit_fitness()
    population = E.evolutionary_search(MINIMUM_FITNESS=min(null_f, 0),
                                       use_double_point_crossover=True,
                                       plot_fitness=False)[0]
    c = Genotype.remove_redundant_gates(population[0].to_circuit())
    population[0].from_circuit(c)
    print(population[0].to_circuit())

    

    #to_plot, stats = multiple_runs(E, iterations=10, MINIMUM_FITNESS=min(null_f, 0))
    #print(f'average runtime: {remaining_time_calc(list_avr(stats["runtime"]))}')
    #plot_many_averages(to_plot, 'Generations', 'Circuit Fitness', legend=False, reference_line=(2**N-1)/(2**N))
        