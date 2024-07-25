from circuit_unoptimiser import unoptimiser
from quantum_fourier_transform import QFT_blueprint, QFTGeneration, GATE_SET
from linear_genetic_programming import Genotype, Evolution
from qiskit import transpile

N = 4
qft = QFT_blueprint(N)
qft = transpile(qft, basis_gates=[gate.name for gate in GATE_SET], optimization_level=0)

qft_gen = QFTGeneration(GATE_SET, N)
qft_gen.print_gate_set()

E = Evolution(qft_gen, sorting_function_override=lambda genotype: 100*genotype.get_fitness() - len(genotype.to_circuit().data),
              number_of_generations=100, gen_mulpilier=8)


'''
circuit_population = [unoptimiser(qft, qft_gen, N) for i in range(E.SAMPLE_SIZE)]
population = []
for c in circuit_population:
    print(c)
    g = Genotype(qft_gen)
    g.from_circuit(c)
    population.append(g)
'''
g = Genotype(qft_gen)
c = unoptimiser(qft, qft_gen, N)
print(f'r_unopt = {c.depth()/qft.depth()}')
g.from_circuit(c)
print(g.to_circuit())
print(g.get_fitness())
population = [g for _ in range(E.SAMPLE_SIZE)]

population = E.evolutionary_optimisation(population, plot_depth=True, insert_delete_proportion=0.5)[0]

print(population[0].to_circuit())
print(population[0].get_fitness())

print(f'r_opt = {population[0].to_circuit().depth()/qft.depth()}')

qiskit_optimised = transpile(g.to_circuit(), basis_gates=[gate.name for gate in GATE_SET], optimization_level=3)
print(qiskit_optimised)
print(f'r_opt (qiskit) = {qiskit_optimised.depth()/qft.depth()}')

print('unoptimised')
print(c)
print(f'r_unopt = {c.depth()/qft.depth()}')