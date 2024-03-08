import unittest
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT as QFT_blueprint
from qiskit.circuit.library import *
from linear_genetic_programming import *

class test_lgp(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.prob_param = AppliedProblemParameters([HGate(), CXGate()],
                                                   input_states=[Statevector.from_int(0,2**3)],
                                                   output_states=[Statevector.from_int(4,2**3)])
        self.g = Genotype(self.prob_param,"0001102")
        self.a_prob_param = AppliedProblemParameters([HGate(), XGate(), CXGate(), PhaseGate(0), 
                                                      RGate(0, 0), TGate(), TdgGate(), CPhaseGate(0)],
                                                      target_circuit=QFT_blueprint(3))

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
            # max length can be exceed by at most 1 gate worth of characters
            self.assertLess(len(random_g.genotype_str),43)

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

    def test_insertion(self):
        insertion = self.g
        for _ in range(5):
            post_insertion = Genotype.insertion(insertion)
            self.assertGreater(len(post_insertion.genotype_str),len(insertion.genotype_str))
            insertion = post_insertion

    def test_deletion(self):
        deletion = self.g
        for _ in range(2):
            post_deletion = Genotype.deletion(deletion)
            self.assertLess(len(post_deletion.genotype_str),len(deletion.genotype_str))
            deletion = post_deletion
        post_deletion = Genotype.deletion(deletion)
        self.assertEqual(len(post_deletion.genotype_str),len(deletion.genotype_str))

    def test_develop_circuits_uniform(self):
        E = Evolution(self.prob_param)
        population = [Genotype(self.prob_param) for _ in range(20)]
        for g in population:
            g.fitness=0.5

        evolved_population = E.develop_circuits_uniform(population, 20)

        self.assertGreater(len(evolved_population), len(population))
        for g in population:
            self.assertIn(g, evolved_population)

    def test_develop_circuits_random(self):
        E = Evolution(self.prob_param)
        population = [Genotype(self.prob_param) for _ in range(20)]
        for g in population:
            g.fitness=0.5

        evolved_population = E.develop_circuits_random(population, 20)

        self.assertGreaterEqual(len(evolved_population), len(population)+20) # at least as many extra as operations
        for g in population:
            self.assertIn(g, evolved_population)

    def test_truth_table(self):
        with self.assertRaises(ValueError):
            a_prob_param_fail = AppliedProblemParameters([HGate(), XGate()])
            # providing no circuit leads to output_states=[]
        
        # testing truth table generation
        for n in [3,4,5]:
            a_prob_param = AppliedProblemParameters([HGate(), XGate(), CXGate()],
                                                    target_circuit=QFT_blueprint(n))
            self.assertEqual(len(a_prob_param.input_states),2**n)
            self.assertEqual(len(a_prob_param.output_states),len(a_prob_param.input_states))
        
        a_prob_param.set_new_circuit(QFT_blueprint(3))
        self.assertEqual(len(a_prob_param.input_states),2**3)
        self.assertEqual(len(a_prob_param.output_states),len(a_prob_param.input_states))

    def test_circuit_fitness(self):
        with self.assertRaises(ValueError):
            # attempting qubit count mismatch
            self.a_prob_param.circuit_fitness(QFT_blueprint(4))
        self.assertEqual(self.a_prob_param.circuit_fitness(QFT_blueprint(3)),1.0)
        self.assertLess(Genotype(self.a_prob_param,"000102202210").get_fitness(),1.0)

    def test_sort_by_fitness(self):
        population = [Genotype(self.a_prob_param) for _ in range(50)]
        #E = Evolution(self.a_prob_param)
        sorted_population = Evolution.sort_by_fitness(population)
        sorted_population_fitness = [g.get_fitness() for g in sorted_population]
        self.assertEqual(sorted_population_fitness[0],max(sorted_population_fitness))
        self.assertEqual(sorted_population_fitness[-1],min(sorted_population_fitness))

    def test_top_by_fitness(self):
        population = [Genotype(self.a_prob_param) for _ in range(100)]
        E = Evolution(self.a_prob_param)
        developed_population = E.develop_circuits_random(population, 100)
        new_population = E.top_by_fitness(developed_population)

        self.assertLessEqual(len(new_population),100)
        self.assertGreaterEqual(len(developed_population),2*len(new_population))
        
    def test_evolutionary_search(self):
        E = Evolution(self.a_prob_param, number_of_generations=25, gen_mulpilier=2)
        final_population, fitness_trace = E.evolutionary_search(output=False)
        final_population_fitness = [g.get_fitness() for g in final_population]

        for i in range(len(fitness_trace[0])-1):
            self.assertLessEqual(fitness_trace[0][i],fitness_trace[0][i+1])
        self.assertEqual(final_population_fitness[0],max(final_population_fitness))
        