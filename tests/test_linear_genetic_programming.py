import unittest
from qiskit import QuantumCircuit
from qiskit.circuit.library import *
from linear_genetic_programming import *

class test_lgp(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.prob_param = AppliedProblemParameters([HGate(), CXGate()],
                                                   input_states=[Statevector.from_int(0,2**3)],
                                                   output_states=[Statevector.from_int(4,2**3)])
        self.g = Genotype(self.prob_param,"0001102")

    def test_genotype_init(self):
        control_circuit = QuantumCircuit(3)
        control_circuit.h(0)
        control_circuit.h(1)
        control_circuit.cx(0,2)

        self.assertEqual(self.g.to_circuit(),control_circuit)
        self.assertEqual(self.g.to_list(),["00","01","102"])

        for _ in range(10):
            random_g = Genotype(self.prob_param,min_length=20,max_length=40)
            self.assertEqual(len(random_g.genotype_str), random_g.get_depth())
            self.assertGreaterEqual(len(random_g.genotype_str),20)
            self.assertLess(len(random_g.genotype_str),43) # max length can be exceed by at most 1 gate worth of characters

    def test_genotype_single_crossover(self):
        for _ in range(10):
            single_crossover_uniform = Genotype.single_crossover(self.g,self.g)
            self.assertEqual(single_crossover_uniform[0].genotype_str,self.g.genotype_str)
            self.assertEqual(single_crossover_uniform[1].genotype_str,self.g.genotype_str)
            #single_crossover_non_uniform = Genotype.single_crossover(g,g,False)

    def test_genotype_double_crossover(self):
        for _ in range(10):
            double_crossover_uniform = Genotype.double_crossover(self.g,self.g)
            self.assertEqual(double_crossover_uniform[0].genotype_str,self.g.genotype_str)
            self.assertEqual(double_crossover_uniform[1].genotype_str,self.g.genotype_str)
            #single_crossover_non_uniform = Genotype.single_crossover(g,g,False)