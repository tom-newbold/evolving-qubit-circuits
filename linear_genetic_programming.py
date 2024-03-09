import random, math
import numpy as np
import matplotlib.pyplot as plt
from time import time

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from linear_genetic_programming_utils import *
from bulk_runs import remaining_time_calc

class Genotype:
    def __init__(self, problem_parameters, genotype_string=None, min_length=None, max_length=None, falloff=None):            
        """omitting genotype_string causes string to be randomly generated according to either provided params,
           or params from problem_parameters; falloff takes one of the following values
           ['linear','logarithmic','reciprocal','']"""
        self.genotype_str = genotype_string
        self.circuit = None
        if type(problem_parameters)!=AppliedProblemParameters:
            raise TypeError('First argument to Genotype constructor is not instace of ProblemParameters')
        self.metadata = problem_parameters
        self.fitness = None
        self.depth = None
        if self.genotype_str==None:
            if min_length==None or max_length==None or falloff==None:
                if min_length!=None and max_length!=None:
                    self.generate_random_genotype(min_length, max_length, problem_parameters.genotype_length_falloff)
                elif problem_parameters.genotype_length_bounds!=None and len(problem_parameters.genotype_length_bounds)==2:
                    self.generate_random_genotype(problem_parameters.genotype_length_bounds[0],
                                                  problem_parameters.genotype_length_bounds[1],
                                                  problem_parameters.genotype_length_falloff)
                else:
                    raise ValueError('Random generation parameters missing or only partially specified')
            else:
                self.generate_random_genotype(min_length, max_length, falloff)

    ### ---------- GENOTYPE UTILS ----------

    def to_list(self):
        """takes the genotype and splits using the number of arguments per gate"""
        out = []
        i = 0
        while i<len(self.genotype_str):
            gate = self.genotype_str[i]
            j = i + self.metadata.gate_set[gate].num_qubits + 1
            j += len(self.metadata.gate_set[gate].params)
            k = self.genotype_str[i:j]
            i = j
            out.append(k)
        return out

    def to_circuit(self):
        """decodes a genotype string to circuit form"""
        if self.circuit:
            return self.circuit
        else:
            circuit_instance = QuantumCircuit(self.metadata.qubit_count)
            for k in self.to_list():
                try:
                    circuit_instance = self.construct_gate(k, circuit_instance)
                except ValueError:
                    print(self.genotype_str)
                    print(self.to_list())
                    raise ValueError
            self.circuit = circuit_instance
            return circuit_instance
    
    def construct_gate(self, genotype_string, c_instance):
        """constructs a single gate from a string and appends to the given circuit"""
        g_list = [int(x) for x in genotype_string[1:]]

        g = self.metadata.gate_set[genotype_string[0]].copy()
        if len(g.params) > 0:
            g.params = [math.pi/x for x in g_list[g.num_qubits:]]

        try:
            c_instance.append(
                g,
                g_list[:g.num_qubits]
            )
        except:
            print(g_list)
            raise RuntimeError(f'Cannot decode gate {genotype_string}')
        return c_instance
    
    def generate_random_genotype(self, min_length, max_length, falloff):
        """generates a random genotype according to the given parameters"""
        gradient = -1/(max_length-min_length)
        intercept = -max_length*gradient
        g = ''
        
        if falloff=='linear':
            while True:
                new_gate = random.choice(self.metadata.all_gate_combinations)
                g += new_gate
                for _ in range(len(self.metadata.gate_set[new_gate[0]].params)):
                    g += str(random.randint(1,9))

                if random.random() > intercept + gradient*len(g):
                    break
        elif falloff=='logarithmic':
            while True:
                new_gate = random.choice(self.metadata.all_gate_combinations)
                g += new_gate
                for _ in range(len(self.metadata.gate_set[new_gate[0]].params)):
                    g += str(random.randint(1,9))

                try:
                    if random.random() > math.log10(1-9*(len(g)-max_length)/(max_length-min_length)):
                        break
                except:
                    break
        elif falloff=='reciprocal':
            while True:
                new_gate = random.choice(self.metadata.all_gate_combinations)
                g += new_gate
                for _ in range(len(self.metadata.gate_set[new_gate[0]].params)):
                    g += str(random.randint(1,9))

                if random.random() > min_length/len(g):
                    break
        else:
            while True:
                new_gate = random.choice(self.metadata.all_gate_combinations)
                g += new_gate
                
                for _ in range(len(self.metadata.gate_set[new_gate[0]].params)):
                    g += str(random.randint(1,9))

                if len(g) > max_length:
                    break
        self.genotype_str = g
        self.circuit = None
        self.fitness = None
        self.depth = None

    def get_fitness(self):
        """calculates fitness for genotype and stores"""
        if self.fitness == None:
            self.fitness = self.metadata.circuit_fitness(self.to_circuit())
        return self.fitness
    
    def get_depth(self):
        if self.depth==None:
            self.depth = len(self.genotype_str)
            #self.depth = self.to_circuit().depth # too computationally expensive
        return self.depth
    
    ### ---------- EVOLUTIONARY OPERATORS ----------
    
    @staticmethod
    def single_crossover(genotype_1, genotype_2, uniform=True):
        """computes the random crossover of two genotypes; the point is selected
           separately on each genotype, allowing for length variations if uniform=False"""
        genotype_1_list = genotype_1.to_list()
        if len(genotype_1_list) <= 1:
            g_1_split = 0
        else:
            g_1_split = random.randint(1,len(genotype_1_list)-1)
        genotype_2_list = genotype_2.to_list()
        if len(genotype_2_list) <= 1:
            g_2_split = 0
        else:
            g_2_split = random.randint(1,len(genotype_2_list)-1)
        if g_1_split == 0 and g_2_split == 0:
            return genotype_1, genotype_2
        else:
            if uniform:
                # makes both split points the same value
                g_1_split = g_2_split = min(g_1_split, g_2_split)
            new_genotype_1 = genotype_1_list[:g_1_split] + genotype_2_list[g_2_split:]
            new_genotype_2 = genotype_2_list[:g_2_split] + genotype_1_list[g_1_split:]
            return Genotype(genotype_1.metadata, ''.join(new_genotype_1)), Genotype(genotype_1.metadata, ''.join(new_genotype_2))
        
    @staticmethod
    def double_crossover(genotype_1, genotype_2, uniform=True):
        """computes the random crossover of two genotypes; two points are selected
           on each genotype; length variations can occur if uniform=False"""
        genotype_1_list = genotype_1.to_list()
        if len(genotype_1_list) <= 1:
            g_1_split_left = g_1_split_right = 0
        else:
            g_1_split_left, g_1_split_right = sorted([random.randint(1,len(genotype_1_list)-1) for _ in range(2)])
        genotype_2_list = genotype_2.to_list()
        if len(genotype_2_list) <= 1:
            g_2_split_left = g_2_split_right = 0
        else:
            g_2_split_left, g_2_split_right = sorted([random.randint(1,len(genotype_2_list)-1) for _ in range(2)])
        if g_1_split_left == 0 and g_2_split_left == 0:
            return genotype_1, genotype_2
        else:
            if uniform:
                # makes both split points the same value
                g_1_split_left = g_2_split_left = min(g_1_split_left, g_2_split_left)
                g_1_split_right = g_2_split_right = min(g_1_split_right, g_2_split_right)
            new_genotype_1 = genotype_1_list[:g_1_split_left] + genotype_2_list[g_2_split_left:g_2_split_right] + genotype_1_list[g_1_split_right:]
            new_genotype_2 = genotype_2_list[:g_2_split_left] + genotype_1_list[g_1_split_left:g_1_split_right] + genotype_2_list[g_2_split_right:]
            return Genotype(genotype_1.metadata, ''.join(new_genotype_1)), Genotype(genotype_1.metadata, ''.join(new_genotype_2))
    
    @staticmethod
    def mutation(genotype):
        """mutates a genotype by either chosing a different gate from
           the given gate set, or modifying one of the gate inputs"""
        genotype_list = genotype.to_list()
        mutation_point = random.randint(0,len(genotype_list)-1)
        gate = genotype_list[mutation_point]
        prev_gate = gate
        while gate==prev_gate:
            if random.random() < 0.5:
                # mutate gate
                old_gate_index = gate[0]
                new_gate_index = random.choice(list(genotype.metadata.gate_set))
                if new_gate_index==old_gate_index:
                    continue
                old_input_count = genotype.metadata.gate_set[old_gate_index].num_qubits
                new_input_count = genotype.metadata.gate_set[new_gate_index].num_qubits
                prev_inputs = gate[1:1+old_input_count]
                # adjust inputs
                if len(prev_inputs) > new_input_count:
                    prev_inputs = prev_inputs[:new_input_count]
                else:
                    while len(prev_inputs) < new_input_count:
                        # add inputs to satify new gate
                        i = str(random.randint(0,genotype.metadata.qubit_count-1))
                        if i not in prev_inputs:
                            prev_inputs += i

                old_param_count = len(genotype.metadata.gate_set[old_gate_index].params)
                new_param_count = len(genotype.metadata.gate_set[new_gate_index].params)
                
                prev_params = gate[-old_param_count:] if old_param_count != 0 else ''
                # adjust params
                if len(prev_params) > new_param_count:
                    prev_params = prev_params[:new_param_count]
                else:
                    while len(prev_params) < new_param_count:
                        # add params
                        prev_params += str(random.randint(1,9))
                
                gate = str(new_gate_index) + prev_inputs + prev_params
            else:
                param_count = len(genotype.metadata.gate_set[gate[0]].params)
                if param_count > 0 and random.random() < 0.25:
                    # mutate a parameter
                    mutate_input = False
                    index_to_change = random.randint(0,param_count-1)
                    index_to_change += 1 + genotype.metadata.gate_set[gate[0]].num_qubits
                    new_param = str(random.randint(1,9))
                    gate = gate[:index_to_change] + str(new_param) + gate[index_to_change+1:]
                else:
                    # mutate an input
                    input_count = genotype.metadata.gate_set[gate[0]].num_qubits
                    index_to_change = random.randint(1,input_count)
                    new_input = random.randint(0,genotype.metadata.qubit_count-1)
                    if str(new_input) not in prev_gate[1:1+input_count]:
                        # adds new input if not a duplicate
                        gate = gate[:index_to_change] + str(new_input) + gate[index_to_change+1:]

        genotype_list[mutation_point] = gate
        return Genotype(genotype.metadata, ''.join(genotype_list))
    
    @staticmethod
    def insertion(genotype):
        """inserts a new random gate at a randomly chosen point in the genotype"""
        g_add = random.choice(genotype.metadata.all_gate_combinations)
        new_gate = g_add[0]
        params = []
        while len(params) < len(genotype.metadata.gate_set[new_gate].params):
            params.append(str(random.randint(1,9)))
        g_add += ''.join(params)
        # insert at random position
        genotype_list = genotype.to_list()
        genotype_add_index = random.randint(0,len(genotype_list)-1)
        new_string = ''.join(genotype_list[:genotype_add_index]) + g_add + ''.join(genotype_list[genotype_add_index:])
        return Genotype(genotype.metadata, new_string)

    @staticmethod
    def deletion(genotype):
        """removes a random gate from the genotype"""
        genotype_list = genotype.to_list()
        if len(genotype_list)<=1:
            return genotype
        genotype_remove_index = random.randint(0,len(genotype_list)-1)
        new_string = ''.join(genotype_list[:genotype_remove_index] + genotype_list[genotype_remove_index+1:])
        return Genotype(genotype.metadata, new_string)


from abc import ABC, abstractmethod

class ProblemParameters(ABC):
    def __init__(self, qubits, set_of_gates, genotype_len_bounds=(), genotype_length_falloff=None):
        """if set_of_gates is a dictionary, any invalid keys are remapped;
           if set_of_gates is a list, keys are assigned (single digit ints
           if there are less than 10 gate, otherwise a range of english and
           greek letters are used - the set can contain at most 80 gates);
           genotype_length_falloff takes one of the following values:
           ['linear','logarithmic','reciprocal','']"""
        self.qubit_count = qubits
        self.set_gate_set(set_of_gates)
        if len(genotype_len_bounds)>=2:
            self.genotype_length_bounds = (genotype_len_bounds[0],genotype_len_bounds[1])
        else:
            self.genotype_length_bounds = None
        self.genotype_length_falloff = genotype_length_falloff

    def set_gate_set(self, set_of_gates):
        """parses the input set_of_gates to a valid dictionary"""
        if len(set_of_gates)==0:
            raise ValueError('set_of_gates is empty')
        if type(set_of_gates) == dict:
            # checks chosen symbols
            set_of_gates_dict = {}
            for key in set_of_gates:
                old_key = key
                key = str(key)
                if len(key) > 1:
                    print(f'Gate identifier {key} uses more than one symbol, attempting to remap')
                    for k in key:
                        if k not in set_of_gates:
                            set_of_gates_dict[k] = set_of_gates[old_key]
                            break
                    i=0
                    if k not in set_of_gates_dict:
                        while i<80:
                            if encode_to_letter(i) not in set_of_gates:
                                if encode_to_letter(i) not in set_of_gates_dict:
                                    set_of_gates_dict[encode_to_letter(i)] = set_of_gates[old_key]
                                    break
                            i+=1
                        if i==80:
                            raise RuntimeError('Could not remap gate')
                else:
                    set_of_gates_dict[key] = set_of_gates[old_key]
            self.gate_set = set_of_gates_dict
            self.all_gate_combinations = self.generate_gate_combinations()
        elif type(set_of_gates) == list:
            # assigns symbols
            if len(set_of_gates) > 80:
                raise ValueError('List exceeds inbuilt base-80: Insufficient symbols')
            set_of_gates_dict = {}
            if len(set_of_gates) < 10:
                for i in range(len(set_of_gates)):
                    set_of_gates_dict[str(i)] = set_of_gates[i]
            else:
                for i in range(len(set_of_gates)):
                    set_of_gates_dict[encode_to_letter(i)] = set_of_gates[i]
            self.gate_set = set_of_gates_dict
            self.all_gate_combinations = self.generate_gate_combinations()
        else:
            raise TypeError('set_of_gates is not a dictionary or list')

    def print_gate_set(self):
        """outputs gate set to console with colouring for readability"""
        print('{')
        for symbol in self.gate_set:
            print(f'    {ansi(92)}{symbol}{ansi()} : {ansi(96)}{self.gate_set[symbol].base_class.__name__}{ansi()}')
        print('}')
        
    def generate_gate_combinations(self):
        """iterates through gate set to find all possible gate combinations;
           used for more efficient random circuit creation"""
        double_input_combinations = []
        for i in range(self.qubit_count):
            other = list(range(self.qubit_count))
            other.remove(i)
            for j in other:
                double_input_combinations.append(str(i)+str(j))

        all_gates = []
        for index in self.gate_set:
            gate = self.gate_set[index]
            if gate.num_qubits==1:
                for q in range(self.qubit_count):
                    all_gates.append(str(index)+str(q))
            elif gate.num_qubits==2:
                for q in double_input_combinations:
                    all_gates.append(str(index)+q)
        return all_gates
    
    def __msf(self, candidate_circuit, input_states, output_states):
        """mean square fidelity function over a set of input and output states"""
        M = Operator(candidate_circuit)
        fidelity_sum = 0
        penalty = 1/len(input_states)
        for i, state in enumerate(input_states):
            calc_state = state.evolve(M)
            if calc_state==output_states[i]:
                fidelity_sum += 1.0
            else:
                fidelity_sum += abs(np.inner(output_states[i].data, calc_state.data).item())**2
                fidelity_sum -= penalty
        return fidelity_sum/len(input_states)
    
    @abstractmethod
    def circuit_fitness(self, candidate_circuit):
        pass

    def get_null_circuit_fitness(self):
        # gets fitness of circuit with no gates
        return Genotype(self, '').get_fitness()

class AppliedProblemParameters(ProblemParameters):
    def __init__(self, set_of_gates, target_circuit=None, input_states=[], output_states=[], N=3, genotype_len_bounds=(), genotype_length_falloff=None):
        """if output_states is a circuit object, uses to evaluate truth table;
           otherwise, assumed to be a precalulated list of states;
           genotype_length_falloff takes one of the following values:
           ['linear','logarithmic','reciprocal','']"""
        # sets number of qubits and input states
        try:
            N = target_circuit.num_qubits
        except:
            pass
        if len(input_states) > 0:
            self.input_states = input_states
        else:
            self.input_states = basis_states(N)

        try:
            # tries to calculate the effect of target_circuit on input_states
            self.M = Operator(target_circuit)
            self.output_states = [s.evolve(self.M) for s in self.input_states]
        except:            
            self.output_states = output_states
        super().__init__(N, set_of_gates, genotype_len_bounds, genotype_length_falloff)

        if len(self.input_states)!=len(self.output_states):
            raise ValueError('Inconsistent size of input_states and output_states')
        
    def set_new_circuit(self, new_circuit):
        """sets the target_circuit (matrix) to a match the new circuit"""
        if self.qubit_count!=new_circuit.num_qubits:
            self.qubit_count = new_circuit.num_qubits
            # gate combinations will change with different qubit count
            self.all_gate_combinations = self.generate_gate_combinations()
        self.M = Operator(new_circuit)
        self.recalc_states()

    def recalc_states(self, input_states=[]):
        """resets input and ountput states, when new circuit set"""
        if len(input_states) > 0:
            self.input_states = input_states
        else:
            self.input_states = basis_states(self.qubit_count)
        self.output_states = [s.evolve(self.M) for s in self.input_states]
        
    def circuit_fitness(self, candidate_circuit):
        """overrides with the required truth table"""
        if candidate_circuit.num_qubits!=self.qubit_count:
            raise ValueError('Qubit count mismatch')
        return self._ProblemParameters__msf(candidate_circuit, self.input_states, self.output_states)

class Evolution:
    def __init__(self, problem_parameters, sample_percentage=0.05, number_of_generations=50,
                 individuals_per_generation=100, gen_mulpilier=5, alpha=1, beta=2, gamma=2):
        self.metadata = problem_parameters
        self.SAMPLE_SIZE = int(individuals_per_generation*sample_percentage)
        print(f'sample size: {self.SAMPLE_SIZE}')
        self.GENERATION_COUNT = number_of_generations
        self.GENERATION_SIZE = individuals_per_generation
        self.GENERATION_MULTIPLIER = gen_mulpilier
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    ### ---------- CIRCUIT SELECTION ----------

    @staticmethod
    def sort_by_fitness(population, min_fitness=0, prefer_short_circuits=False, prefer_long_circuits=False, remove_dupe=True):
        """sorts population by fitness, also removed duplicates / sorts by circuit depth if specified"""
        by_fitness = population.copy()
        if remove_dupe:
            by_fitness = remove_duplicates(by_fitness)
        if prefer_short_circuits != prefer_long_circuits:
            if prefer_short_circuits:
                by_fitness = sorted(by_fitness, key=lambda genotype: genotype.get_fitness()/genotype.get_depth(), reverse=True)
            else:
                by_fitness = sorted(by_fitness, key=lambda genotype: genotype.get_fitness()*genotype.get_depth(), reverse=True)
        else:
            by_fitness = sorted(by_fitness, key=lambda genotype: genotype.get_fitness(), reverse=True)
        while by_fitness[-1].get_fitness() < min_fitness:
            by_fitness.pop(-1)
        return by_fitness
    
    def top_by_fitness(self, population, min_fitness=0, prefer_short_circuits=False, prefer_long_circuits=False, remove_dupe=True):
        """finds the best circuits in the population; top sample taken as well as uniform selection of remaining circuits"""
        by_fitness = Evolution.sort_by_fitness(population, min_fitness, prefer_short_circuits, prefer_long_circuits, remove_dupe)
        step = (len(by_fitness)-self.SAMPLE_SIZE)//(self.GENERATION_SIZE-self.SAMPLE_SIZE)
        step = 1 if step==0 else step
        end = (1-step)*self.SAMPLE_SIZE + step*self.GENERATION_SIZE
        return by_fitness[:self.SAMPLE_SIZE] + by_fitness[self.SAMPLE_SIZE:end:step]
        
        ### ---------- BASELINE ALGORITHMS ----------

    def random_search(self, min_length=30, max_length=45, falloff='linear', remove_duplicates=True,
                      output=True, plot_fitness=True, plot_depth=False):
        """returns final population and fitness trace"""
        fitness_trace = [[0] for i in range(self.SAMPLE_SIZE)]
        depth_trace = [[] for i in range(self.SAMPLE_SIZE)]
        population = []

        for generation in range(self.GENERATION_COUNT):
            for _ in range(self.GENERATION_SIZE):
                g = Genotype(self.metadata, min_length=min_length, max_length=max_length, falloff=falloff)
                g.get_fitness()
                population.append(g)
            # sort population by fitness, take top sample
            population = self.top_by_fitness(population, remove_dupe=remove_duplicates)
            # each run compares the new programs with the
            # sample carried forward from the previous generation
            if output:
                print(f'Generation {generation+1} best: {population[0].genotype_str}')
            if plot_fitness:
                for x in range(self.SAMPLE_SIZE):
                    try:
                        fitness_trace[x].append(population[x].fitness)
                    except:
                        fitness_trace[x].append(0)
            if plot_depth:
                for x in range(self.SAMPLE_SIZE):
                    try:
                        depth_trace[x].append(population[x].get_depth())
                    except:
                        depth_trace[x].append(0)

        if output:
            s = min(self.SAMPLE_SIZE, len(population))
            print(f'top {s}:')
            for i in range(s):
                print(population[i].genotype_str)
                print(population[i].fitness)

            print('best random circuit:')
            print(population[0].to_circuit())

            if plot_fitness:
                plot_list(fitness_trace, 'Generations', 'Circuit Fitness')
                plt.show()
                plot_list(fitness_trace, 'Generations', 'Circuit Fitness', False)
                plt.show()
            if plot_depth:
                plot_list(depth_trace, 'Generations', 'Genotype Length')#'Circuit Depth')
                plt.show()

        return population, fitness_trace
    
    def stochastic_hill_climb(self, min_length=30, max_length=45, falloff='linear', MINIMUM_FITNESS=0.0,
                              remove_duplicates=True, output=True, plot_fitness=True, plot_depth=False):
        """returns final population and fitness trace"""
        best_genotype = Genotype(self.metadata, '')
        best_genotype.fitness = MINIMUM_FITNESS
        fitness_trace = []
        depth_trace = []
        fitness_trace = [[0] for i in range(self.SAMPLE_SIZE)]
        depth_trace = [[] for i in range(self.SAMPLE_SIZE)]

        population = []
        for generation in range(self.GENERATION_COUNT):
            for _ in range(self.GENERATION_SIZE):
                g = Genotype(self.metadata, min_length=min_length, max_length=max_length, falloff=falloff)
                m = g.get_fitness()
                population.append(g)

            if len(population) > 0:
                population = self.top_by_fitness(population, remove_dupe=remove_duplicates)
                # select a random genotype, using the fitness improvements as weights
                try:
                    best_subset = list(filter(lambda g: g.get_fitness() - best_genotype.get_fitness() > 0, population))
                    best_genotype = random.choices(best_subset, weights=[(g.get_fitness() - best_genotype.get_fitness()) for g in best_subset], k=1)[0]
                except:
                    pass

            if output:
                print(f'Generation {generation+1} best: {best_genotype.genotype_str}')
            if plot_fitness:
                for x in range(self.SAMPLE_SIZE):
                    try:
                        fitness_trace[x].append(population[x].fitness)
                    except:
                        fitness_trace[x].append(0)
            if plot_depth:
                for x in range(self.SAMPLE_SIZE):
                    try:
                        depth_trace[x].append(population[x].get_depth())
                    except:
                        depth_trace[x].append(0)

        if output:
            print('best random circuit:')
            print(best_genotype.genotype_str)
            print(best_genotype.to_list())
            print(best_genotype.to_circuit())
            print(best_genotype.fitness)
            
            if plot_fitness:
                plot_list(fitness_trace, 'Generations', 'Circuit Fitness')
                plt.show()
            if plot_depth:
                plot_list(depth_trace, 'Generations', 'Genotype Length')#'Circuit Depth')
                plt.show()

        return population, fitness_trace
    
    ### ---------- EVOLUTIONARY SEARCH ----------

    def develop_circuits_uniform(self, inital_population, use_double_point_crossover=True):
        """use a prespecified distribution of search operators
        population should be sorted by fitness"""
        population = inital_population.copy()
        # crossover operation for every pair of genotypes in the sample
        for g_1_index in range(len(inital_population)):
            for g_2_index in range(g_1_index+1,len(inital_population)):
                for c in range(self.gamma):
                    g_3, g_4 = Genotype.single_crossover(population[g_1_index],population[g_2_index])
                    population.append(g_3)
                    population.append(g_4)
        if use_double_point_crossover==True:
            for g_1_index in range(len(inital_population)):
                for g_2_index in range(g_1_index+1,len(inital_population)):
                    for c in range(self.gamma):
                        g_3, g_4 = Genotype.double_crossover(population[g_1_index],population[g_2_index])
                        population.append(g_3)
                        population.append(g_4)
        # mutation operation for every genotype in the sample
        # insertion/deletion for each genotype in the sample
        for g_1_index in range(len(inital_population)):
            for a in range(self.alpha):
                g_2 = g_3 = g_4 = population[g_1_index]
                for b in range(self.beta):
                    g_2 = Genotype.mutation(g_2)
                    g_3 = Genotype.insertion(g_3)
                    g_4 = Genotype.deletion(g_4)
                    for g in [g_2, g_3, g_4]:
                        population.append(g)
        return population

    def develop_circuits_random(self, inital_population, operation_count, use_double_point_crossover=True,
                                crossover_proportion=0.5, insert_delete_proportion=0.1):
        """use a random assortment of search operators"""
        population = inital_population.copy()
        operations = ['crossover', 'mutation', 'insersion', 'deletion']
        w = [(1-insert_delete_proportion)*crossover_proportion,
             (1-insert_delete_proportion)*(1-crossover_proportion),
             insert_delete_proportion/2, insert_delete_proportion/2]
        while operation_count > 0:
            # randomly select from the search operators
            operation = random.choices(population=operations, weights=w, k=1)[0]
            # randomly select a genotype
            g_1 = random.choices(inital_population, weights=[g.get_fitness() for g in inital_population], k=1)[0]
            if operation == 'crossover':
                g_2 = g_1
                while g_2 == g_1:
                    g_2 = random.choices(inital_population, weights=[g.get_fitness() for g in inital_population], k=1)[0]
                if use_double_point_crossover==True:
                    for c in range(self.gamma):
                        g_3, g_4 = Genotype.double_crossover(g_1, g_2)
                        population.append(g_3)
                        population.append(g_4)
                else:
                    for c in range(self.gamma):
                        g_3, g_4 = Genotype.single_crossover(g_1, g_2)
                        population.append(g_3)
                        population.append(g_4)
                operation_count -= self.gamma*2
            else:
                for a in range(self.alpha):
                    g_2 = g_1
                    for b in range(self.beta):
                        if operation=='mutation':
                            g_2 = Genotype.mutation(g_2)
                        elif operation=='insersion':
                            g_2 = Genotype.insertion(g_2)
                        elif operation=='deletion':
                            g_2 = Genotype.deletion(g_2)
                        population.append(g_2)
                operation_count -= self.alpha*self.beta
        return population

    """DEPRECATED FOR PERFORMANCE REASONS
    def develop_circuits_combined(self, inital_population, operation_count=250, double_point_crossover=True,
                                  crossover_proportion=0.5, insert_delete_proportion=0.1): 
        #population_uniform = self.develop_circuits_uniform(inital_population, double_point_crossover)#[len(inital_population):]
        population_random = self.develop_circuits_random(inital_population, operation_count, double_point_crossover,
                                                         crossover_proportion, insert_delete_proportion)
        #return population_uniform + population_random
        return population_random
    """
    
    def evolutionary_search(self, min_length=None, max_length=None, falloff=None, remove_duplicates=True,
                            MINIMUM_FITNESS=0, crossover_proportion=0.5, insert_delete_proportion=0.1, 
                            output=True, plot_fitness=True, plot_depth=False,
                            random_sample_size=0, use_double_point_crossover=True, prefer_short_circuits=None):
        """generates random population, evolves over generation using input parameters
           returns final population and fitness trace"""
        fitness_trace = [[] for _ in range(self.SAMPLE_SIZE)]
        depth_trace = [[] for _ in range(self.SAMPLE_SIZE)]

        population = []
        while len(population) < self.SAMPLE_SIZE:
            for _ in range(self.GENERATION_SIZE):
                g = Genotype(self.metadata, min_length=min_length, max_length=max_length, falloff=falloff)
                g.get_fitness()
                population.append(g)
            population = self.top_by_fitness(population)
            if population[-1].get_fitness() >= MINIMUM_FITNESS:
                break
            else:
                for i in range(len(population)):
                    if population[i].get_fitness() < MINIMUM_FITNESS:
                        population = population[:i]
                        break
        if output:
            print(f'Generation 0 (initial) Best Genotype: {population[0].genotype_str}')
            print(f'Generation 0 (initial) Size: {len(population)}')
        if plot_fitness:
            for k in range(self.SAMPLE_SIZE):
                try:
                    fitness_trace[k].append(population[k].fitness)
                except:
                    fitness_trace[k].append(0)
        if plot_depth:
            for x in range(self.SAMPLE_SIZE):
                try:
                    depth_trace[k].append(population[k].get_depth())
                except:
                    depth_trace[k].append(0)

        start_time = time()
        for i in range(self.GENERATION_COUNT):
            if not output:
                if i!=1:
                    remaining_time = (time()-start_time) * (self.GENERATION_COUNT-i)/(i+1)
                    remaining_time = remaining_time_calc(remaining_time)
                    if remaining_time:
                        print(f"run progress: [{i*'#'}{(self.GENERATION_COUNT-i)*'_'}] "+
                              f"// estimated time remaining for run ~ {remaining_time}"+20*" ", end='\r')

            # added random sample
            for _ in range(random_sample_size):
                g = Genotype(self.metadata, min_length=min_length, max_length=max_length, falloff=falloff)
                g.get_fitness()
                population.append(g)

            # create new circuits
            population = self.develop_circuits_random(population, int(self.GENERATION_SIZE*(self.GENERATION_MULTIPLIER-1)),
                                                      use_double_point_crossover, crossover_proportion, insert_delete_proportion)
            for g in population:
                g.get_fitness()

            if output:
                print(f'Generation {i+1} Size (pre-selection): {len(population)}')
            
            if prefer_short_circuits!=None:
                population = self.top_by_fitness(population, min_fitness=MINIMUM_FITNESS, remove_dupe=remove_duplicates, prefer_short_circuits=prefer_short_circuits)
            else:
                population = self.top_by_fitness(population, min_fitness=MINIMUM_FITNESS, remove_dupe=remove_duplicates)

            if output:
                print(f'Generation {i+1} Best Genotype: {population[0].genotype_str}')
                print(f'Generation {i+1} Best Fitness: {population[0].fitness}')
            if plot_fitness:
                for k in range(self.SAMPLE_SIZE):
                    try:
                        fitness_trace[k].append(population[k].fitness)
                    except:
                        fitness_trace[k].append(0)
            if plot_depth:
                for x in range(self.SAMPLE_SIZE):
                    try:
                        depth_trace[k].append(population[k].get_depth())
                    except:
                        depth_trace[k].append(0)
                        
        if not output: print((80+self.GENERATION_COUNT)*" ", end='\r') 

        # output
        if output:
            print(f'Top {self.SAMPLE_SIZE} genotypes:')
            for i in range(self.SAMPLE_SIZE):
                print(population[i].genotype_str)
                print(population[i].fitness)
            print('best circuit:')
            print(population[0].to_circuit())

            if plot_fitness:
                plot_list(fitness_trace, 'Generations', 'Circuit Fitness')
                plt.show()
                plot_list(fitness_trace, 'Generations', 'Circuit Fitness', False)
                plt.show()
            if plot_depth:
                plot_list(depth_trace, 'Generations', 'Genotype Length')#'Circuit Depth')
                plt.show()

        return population, fitness_trace