from linear_genetic_programming import AppliedProblemParameters, Evolution, ProblemParametersCombined
from linear_genetic_programming_utils import basis_states
from qiskit import QuantumCircuit

from qiskit.circuit.library import *
    
GATE_SET = [HGate(), XGate(), CXGate(), PhaseGate(0),
            TGate(), TdgGate(), CHGate(), CPhaseGate(0)]

def ToffoliGeneration(set_of_gates):
    """creates a ProblemParameters object using the toffoli gate truth table"""
    c = QuantumCircuit(3)
    c.ccx(0,1,2)
    #from qiskit.quantum_info import random_statevector
    #input_states_sample = [random_statevector(2**3) for _ in range(10)]
    return AppliedProblemParameters(set_of_gates, c, basis_states())
    #return ProblemParametersCombined(set_of_gates, c, basis_states())


if __name__=="__main__":
    
    TOFFOLI = ToffoliGeneration(GATE_SET)
    E = Evolution(TOFFOLI, sample_percentage=0.1, number_of_generations=25, gen_mulpilier=2, alpha=2, beta=3, gamma=3)

    #g = Genotype(TOFFOLI, '022125220242212522024142201024051201')
    #print(g.genotype_str)
    #print(g.to_circuit())
    
    #population = E.random_search()
    #population = E.stochastic_hill_climb()
    population = E.evolutionary_search(MINIMUM_FITNESS=0, remove_duplicates=True)[0]#, random_sample_size=5)
    print(population[0].to_list())

    #grid_search(Evolution(TOFFOLI, sample=10, number_of_generations=20,
    #                      individuals_per_generation=50, alpha=1, beta=2))

    #grid_search_threaded(Evolution(TOFFOLI, sample=10, number_of_generations=20,
    #                               individuals_per_generation=50, alpha=1, beta=2))

    #grid_search(Evolution(TOFFOLI),lengths=([0,15,30,45],[30,45,60]),
    #            falloff=['linear','logarithmic','reciprocal'], iterations=3)