from qiskit import QuantumCircuit
from qiskit.circuit.library import *

from linear_genetic_programming import AppliedProblemParameters, Evolution
from linear_genetic_programming_utils import basis_states
    
GATE_SET = [HGate(), XGate(), CXGate(), PhaseGate(0),
            TGate(), TdgGate(), CHGate(), CPhaseGate(0)]

def genericToffoliConstructor(n=3):
    """constructs a generic controlled-not gate, to be used as blueprint for evolution"""
    c = QuantumCircuit(n)
    if n==3:
        c.ccx(0,1,2)
    elif n==4:
        c.append(
            C3XGate(),
            [0,1,2,3]
        )
    elif n==5:
        c.append(
            C4XGate(),
            [0,1,2,3,4]
        )
    else:
        raise ValueError('Invalid n for Toffoli constructor')
    return c


def ToffoliGeneration(set_of_gates, n=3):
    """creates a ProblemParameters object using the toffoli gate truth table"""
    return AppliedProblemParameters(set_of_gates, genericToffoliConstructor(n), basis_states(n),
                                    genotype_len_bounds=[30,45], genotype_length_falloff='linear')

if __name__=="__main__":
    for n in range(3,6):
        print(genericToffoliConstructor(n))

    TOFFOLI = ToffoliGeneration(GATE_SET)
    E = Evolution(TOFFOLI, sample_percentage=0.1, gen_mulpilier=5, alpha=2, beta=3, gamma=3)

    #g = Genotype(TOFFOLI, '022125220242212522024142201024051201')
    #print(g.genotype_str)
    #print(g.to_circuit())
    
    population = E.random_search()[0]
    population = E.evolutionary_search(MINIMUM_FITNESS=0)[0]#, random_sample_size=5)
    print(population[0].to_list())