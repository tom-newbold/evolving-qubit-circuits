from quantum_fourier_transform import QFTGeneration, GATE_SET
from linear_genetic_programming import Evolution, AppliedProblemParameters

from qiskit.circuit.library import QFT as QFT_blueprint
from qiskit.quantum_info import random_statevector

from grid_search import grid_search, multiple_runs


if __name__=="__main__":
    QFT_GEN = QFTGeneration(GATE_SET, 3, t=0.5)#, 16)
    #QFT_GEN = AppliedProblemParameters(GATE_SET, QFT_blueprint(3),
    #                                   [random_statevector(2**3) for _ in range(10)])
    #E = Evolution(QFT_GEN, individuals_per_generation=300, alpha=3, beta=6, gamma=4)
    E = Evolution(QFT_GEN, sample_percentage=0.1, gen_mulpilier=7.5)

    multiple_runs(E, iterations=10)
    #multiple_runs(E, method='stochastic')
    
    #null_f = QFT_GEN.get_null_circuit_fitness()
    #grid_search(E, 10, [0], [0], [0.1])
    #grid_search(E, 3, [0], [0], [0.05,0.1,0.15,0.2,0.25], [0.025,0.1,0.5,1]) # currently not iterating random sample size
    # try multiples of null fitness