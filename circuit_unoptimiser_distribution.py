from circuit_unoptimiser import unoptimiser
from quantum_fourier_transform import QFT_blueprint, QFTGeneration, GATE_SET
from linear_genetic_programming import Genotype, Evolution
from qiskit import transpile

from pandas import DataFrame
import matplotlib.pyplot as plt

SAMPLES = 100

data = []
labels = []
for N in range(3, 8):
    print(f'--- N={N} ---')
    qft = QFT_blueprint(N)
    qft = transpile(qft, basis_gates=[gate.name for gate in GATE_SET], optimization_level=0)
    #print(qft)

    qft_gen = QFTGeneration(GATE_SET, N)

    circuits = []
    for i in range(SAMPLES):
        print(f'unoptimising: {"#"*(i+1)}{"-"*(SAMPLES-i-1)}', end='\r')
        circuits.append(unoptimiser(qft, qft_gen, N)[0])
    print('')
    #r_unopt = []
    #for i, c in enumerate(circuits):
    #    print(f'calculating ratios: {"#"*(i+1)}{"-"*(SAMPLES-i-1)}', end='\r')
    #    r_unopt.append(c.depth()/qft.depth())
    #print('')
    r_unopt = [c.depth()/qft.depth() for c in circuits]
    data.append(r_unopt)
    labels.append(f'N={N}')

data_dict = {}
for i, label in enumerate(labels):
    column = label#.split('=')[1]
    data_dict[str(column)] = data[i]
df = DataFrame.from_dict(data_dict)

with open(f'out/unoptimisasion_ratios.csv','w') as file:
    # writes dataframe to unique file, statistical analysis and further plots can be carried out externally
    file.write(DataFrame.to_csv(df))
    file.close()

plt.title('unoptimiser produced ratios')
plt.boxplot(data, labels=labels)
plt.ylabel('r_unopt')
plt.show()