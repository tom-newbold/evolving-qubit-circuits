from quantum_fourier_transform import QFTGeneration, GATE_SET, GATE_SET_SIMPLE
from linear_genetic_programming import Evolution, AppliedProblemParameters
from grid_search import multiple_runs


if __name__=="__main__":
    for set in [GATE_SET_SIMPLE, GATE_SET]:
        QFT_GEN = QFTGeneration(set, 3)
        E = Evolution(QFT_GEN, sample_percentage=0.1, gen_mulpilier=7.5)

        multiple_runs(E)
