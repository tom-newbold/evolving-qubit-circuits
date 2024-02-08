from linear_genetic_programming import AppliedProblemParameters, Evolution, ProblemParametersCombined, basis_states
from time import time
#from grid_search_old import grid_search, remaining_time_calc

from qiskit import QuantumCircuit

def ToffoliGeneration(set_of_gates):
    """creates a ProblemParameters object using the toffoli gate truth table"""
    toffoli_inputs = [[i//4 %2, i//2 %2, i%2] for i in range(8)]
    
    toffoli_outputs = []
    for i in range(8):
        x = toffoli_inputs[i].copy()
        if bool(x[0]) and bool(x[1]):
            x[2] = int(not bool(x[2]))
        toffoli_outputs.append(x)
    
    #a = AppliedProblemParameters(set_of_gates, [list_to_state(x) for x in toffoli_inputs],
    #                                [list_to_state(y) for y in toffoli_outputs], 3)
    #return a
    c = QuantumCircuit(3)
    c.ccx(0,1,2)
    #from qiskit.quantum_info import random_statevector
    #input_states_sample = [random_statevector(2**3) for _ in range(10)]
    return AppliedProblemParameters(set_of_gates, basis_states(), c)
    #return ProblemParametersCombined(set_of_gates, basis_states(), c)


if __name__=="__main__":
    from qiskit.circuit.library import *
    
    GATE_SET = [HGate(), XGate(), CXGate(), PhaseGate(0),
                TGate(), TdgGate(), CHGate(), CPhaseGate(0)]
    
    TOFFOLI = ToffoliGeneration(GATE_SET)
    E = Evolution(TOFFOLI, sample_percentage=0.1, individuals_per_generation=50, number_of_generations=20, beta=1, gamma=1)

    #g = Genotype(TOFFOLI, '022125220242212522024142201024051201')
    #print(g.genotype_str)
    #print(g.to_circuit())
    
    #population = E.random_search()
    #population = E.stochastic_hill_climb()
    population = E.evolutionary_search(MINIMUM_FITNESS=0, remove_duplicates=True)#, random_sample_size=5)
    print(population[0].to_list())

    #grid_search(Evolution(TOFFOLI, sample=10, number_of_generations=20,
    #                      individuals_per_generation=50, alpha=1, beta=2))

    #grid_search_threaded(Evolution(TOFFOLI, sample=10, number_of_generations=20,
    #                               individuals_per_generation=50, alpha=1, beta=2))

    #grid_search(Evolution(TOFFOLI),lengths=([0,15,30,45],[30,45,60]),
    #            falloff=['linear','logarithmic','reciprocal'], iterations=3)