from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import GroverOperator
from qiskit import transpile
from quantum_fourier_transform import GATE_SET
from linear_genetic_programming import AppliedProblemParameters, Evolution

def generic_grover_operator(marked_qubit=0, N=3):
    oracle = QuantumCircuit(N)
    oracle.z(marked_qubit) # marks target qubit
    return GroverOperator(oracle)

def grover_operator_generation(set_of_gates, marked_qubit=0, N=3):
    grover_op = generic_grover_operator(marked_qubit, N)
    lower_bound = 3 * N**2
    return AppliedProblemParameters(set_of_gates, grover_op,
                                    genotype_len_bounds=[2*lower_bound, 4*lower_bound],
                                    genotype_length_falloff='linear')


if __name__=="__main__":
    G_GEN = grover_operator_generation(GATE_SET, N=2)

    E = Evolution(G_GEN, individuals_per_generation=100, number_of_generations=200,
                    sample_percentage=0.1, gen_mulpilier=8)


    population = E.evolutionary_opsearch(lambda o: None, [], use_double_point_crossover=True)[0]
