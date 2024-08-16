from qiskit.quantum_info import Operator, Statevector
from qiskit import transpile
import numpy as np

from toffoli_gate_generation import ToffoliGeneration
from quantum_fourier_transform import GATE_SET, QFTGeneration, QFT_blueprint
from linear_genetic_programming import Genotype
from linear_genetic_programming_utils import basis_states

QFT_GEN = QFTGeneration(GATE_SET)
genotype = Genotype(QFT_GEN, '022202026101620201612102')

qft = QFT_blueprint(3).decompose()
qft = transpile(qft, basis_gates=[gate.name for gate in GATE_SET], optimization_level=0)

print(genotype.to_circuit())
print(qft)

matrix_genotype = np.array(Operator(genotype.to_circuit()))
matrix_blueprint = np.array(Operator(qft))

#print(matrix_genotype==-matrix_blueprint)
#print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in matrix_genotype]))
#print('---')
#print('\n'.join(['\t'.join([str(cell) for cell in row]) for row in matrix_blueprint]))

matrix_combined = np.round(np.matmul(matrix_genotype.conj().T, matrix_blueprint), 5)
#print('---')
print('\n'.join(['\t\t'.join([str(cell) for cell in row]) for row in matrix_combined]))