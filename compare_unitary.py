from qiskit.quantum_info import Operator, Statevector
from qiskit import transpile
import numpy as np

from toffoli_gate_generation import ToffoliGeneration
from quantum_fourier_transform import GATE_SET, QFTGeneration, QFT_blueprint
from linear_genetic_programming import Genotype
from linear_genetic_programming_utils import basis_states, list_avr

QFT_GEN = QFTGeneration(GATE_SET)
genotype = Genotype(QFT_GEN, '2016017410260224062112027011311015000202220')
#022202026101620201612102

qft = QFT_blueprint(3).decompose()
qft = transpile(qft, basis_gates=[gate.name for gate in GATE_SET], optimization_level=0)

print(genotype.to_circuit())
print(qft)

for psi in range(7):
    s_1 = Statevector.from_int(psi, 2**3).evolve(Operator(genotype.to_circuit()))
    s_2 = Statevector.from_int(psi, 2**3).evolve(Operator(qft))
    print(s_1.equiv(s_2))

    global_phases = [np.round(s_1[i]/s_2[i], 5) for i in range(len(s_1))]
    print(global_phases)
    print(list_avr(global_phases))

matrix_genotype = np.array(Operator(genotype.to_circuit()))
matrix_blueprint = np.array(Operator(qft))

#print(matrix_genotype==-matrix_blueprint)
#print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in matrix_genotype]))
#print('---')
#print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in matrix_blueprint]))

matrix_combined = np.round(np.matmul(matrix_genotype.conj().T, matrix_blueprint), 5)
#print('---')
print('\n'.join(['\t\t'.join([str(cell) for cell in row]) for row in matrix_combined]))