from quantum_fourier_transform import QFTGeneration, GATE_SET
from linear_genetic_programming import Evolution

from grid_search import grid_search


if __name__=="__main__":
    QFT_GEN = QFTGeneration(GATE_SET, 3, t=0.5)#, 16)
    #E = Evolution(QFT_GEN, individuals_per_generation=300, alpha=3, beta=6, gamma=4)
    E = Evolution(QFT_GEN, sample_percentage=0.1, gen_mulpilier=2.5)
    
    null_f = QFT_GEN.get_null_circuit_fitness()
    grid_search(E, 1, [0], [0], [0.05,0.1,0.15,0.2,0.25], [0.025,0.1,0.5]) # currently not iterating random sample size
    # try multiples of null fitness