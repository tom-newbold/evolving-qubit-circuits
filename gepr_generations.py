import math

from qiskit.circuit.library import *
from qiskit.quantum_info import Statevector

from linear_genetic_programming import AppliedProblemParameters, Evolution

CLIFFORD_T = [HGate(), XGate(), SGate(), SdgGate(), CXGate(), TGate(), TdgGate()]

if __name__=="__main__":
    N = 6

    #state_in = [Statevector.from_int(0, 2**N)]
    coef=math.pow(2, -0.5)
    #states_in = [Statevector.from_int(i, 2**N) for i in range(4)]
    #states_out = [Statevector([coef, 0, 0, (-1)**i]) for i in range(2)] + [Statevector([0, coef, (-1)**i, 0]) for i in range(2)]
    states_in = [Statevector.from_int(i, 2**N) for i in range(1)]
    states_out = [Statevector([math.pow(2, -0.5*N) for _ in range(2**N)]) for i in range(1)]
    app = AppliedProblemParameters(CLIFFORD_T, N=N, input_states=states_in, output_states=states_out,
                                   genotype_len_bounds=[3,30], genotype_length_falloff='linear')
    
    E = Evolution(app, sample_percentage=0.1, gen_mulpilier=8, alpha=2, beta=2, gamma=2)
    #              sorting_function_override=lambda genotype: (1+genotype.get_fitness()) / (1+len(genotype.to_circuit().data)))#genotype.get_depth())
    population = E.evolutionary_search()[0]
