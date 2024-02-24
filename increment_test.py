"""TO REMOVE"""

from qiskit.circuit.library import *
from qiskit.quantum_info import Statevector

from linear_genetic_programming import AppliedProblemParameters, Evolution
from grid_search import multiple_runs
    
GATE_SET = [HGate(), XGate(), YGate(), ZGate(), CXGate(), PhaseGate(0), 
            RGate(0, 0), TGate(), TdgGate(), CHGate(), CPhaseGate(0)]

if __name__=="__main__":
    N = 4
    GEN = AppliedProblemParameters(GATE_SET, input_states=[Statevector.from_int(i, 2**N) for i in range(2**N)],
                                   output_states=[Statevector.from_int((i+1)%2**N, 2**N) for i in range(2**N)], N=N)
    E = Evolution(GEN, sample_percentage=0.1)

    #population = E.evolutionary_search(remove_duplicates=True)
    multiple_runs(E, method='random')