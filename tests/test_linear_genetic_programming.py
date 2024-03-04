import unittest
from qiskit import QuantumCircuit
from qiskit.circuit.library import *
from linear_genetic_programming import *

class test_lgp(unittest.TestCase):
    def test_genotype_init(self):
        control_circuit = QuantumCircuit(3)
        control_circuit.h(0)
        control_circuit.h(1)
        control_circuit.cx(0,2)

        prob_param = AppliedProblemParameters([HGate(), CXGate()],
                                              input_states=[Statevector.from_int(0,2**3)],
                                              output_states=[Statevector.from_int(4,2**3)])
        g = Genotype(prob_param,"0001102")
        self.assertEqual(g.to_circuit(),control_circuit)
        self.assertEqual(g.to_list(),["00","01","102"])

        for _ in range(10):
            g = Genotype(prob_param,min_length=20,max_length=40)
            self.assertEqual(len(g.genotype_str), g.get_depth())
            self.assertGreaterEqual(len(g.genotype_str),20)
            self.assertLess(len(g.genotype_str),43) # max length can be exceed by at most 1 gate worth of characters