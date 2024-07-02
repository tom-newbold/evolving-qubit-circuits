from linear_genetic_programming import Genotype, ProblemParameters
from quantum_fourier_transform import GATE_SET
from toffoli_gate_generation import ToffoliGeneration

if __name__=="__main__":
    app = ToffoliGeneration(GATE_SET)
    app.print_gate_set()
    g = Genotype(app, 'A1A1F04A2B1A2F04E02B1E02')#A1A1A2B1A2C02B1C02
    print(f'fitness: {g.get_fitness()}')
    c = Genotype.static_remove_redundant_gates(g.to_circuit())
    g.from_circuit(c)
    print(f'fitness: {g.get_fitness()}')