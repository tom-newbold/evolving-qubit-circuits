from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from grid_search_old import remaining_time_calc

import matplotlib.pyplot as plt
import random, math
from time import time
import numpy as np

def encodeToLetter(n):
    '''26 (english) capitals , 26 (english) lower case,
       10 valid (greek) upper case, 18 valid (greek) lower case
       80 allowable symbols'''
    if n < 26:
        key = chr(ord('A')+n)
    elif n < 52:
        key = chr(ord('a')+n-26)
    elif n < 62:
        key = ['Γ','Δ','Θ','Λ','Ξ','Π','Σ','Φ','Ψ','Ω'][n-52]
    elif n < 80:
        key = ['α','β','γ','δ','ε','ζ','η','θ','λ','μ',
               'ξ','ρ','σ','τ','φ','χ','ψ','ω'][n-62]
    #elif n < 62:
    #    key = str(n-52)
    #elif n < 64:
    #    key = ['+','/'][n-62]
    else:
        return None
    return key

class Genotype:
    def __init__(self, problem_parameters, genotype_string=None, min_length=15, max_length=45, falloff='linear'):            
        self.genotype_str = genotype_string
        self.circuit = None
        self.metadata = problem_parameters
        self.fitness = None
        #self.source = ''
        if self.genotype_str==None:
            self.generate_random_genotype(min_length, max_length, falloff)#PARAMETERS

    ### ---------- GENOTYPE UTILS ----------

    def to_list(self):
        """takes the genotype and splits using the number of arguments per gate"""
        out = []
        i = 0
        while i<len(self.genotype_str):
            gate = self.genotype_str[i]
            j = i + self.metadata.gate_set[gate].num_qubits + 1
            #if 'parameters' in self.metadata.gate_set[int(gate)]:
            #    j += self.metadata.gate_set[int(gate)]['parameters']
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
            #print(self.to_list())
            circuit_instance = QuantumCircuit(self.metadata.qubit_count)
            for k in self.to_list():
                #gate = int(k[0])
                #if len(k)-1 != self.gate_set[gate]['inputs']:
                #    print('ERROR')
                try:
                    circuit_instance = self.construct_gate(k, circuit_instance)
                except ValueError:
                    print(self.genotype_str)
                    print(self.to_list())
                    raise ValueError
            self.circuit = circuit_instance
            return circuit_instance
    
    def construct_gate(self, genotype_string, c_instance):
        """constructs a single gate from a string and appends to the given ciruit"""
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
            raise ValueError()
        return c_instance
    
    #def generate_random_genotype(self, min_length=15, max_length=45, falloff='linear'):#, input_count_weighted=True):
    def generate_random_genotype(self, min_length, max_length, falloff):#, input_count_weighted=True):
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

    def get_fitness(self):
        """calculates fitness for genotype and stores"""
        if self.fitness == None:
            self.fitness = self.metadata.circuit_fitness(self.to_circuit())
        return self.fitness
    
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
                #if len(gate) != 1 + old_input_count + old_param_count:
                #    print('## ERROR')
                #print(f'old {old_input_count} new {new_input_count} input; old {old_param_count} new {new_param_count} param')

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
                        # print('inserting new input')
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



def list_to_state(x):
    return Statevector.from_int(x[2]*4+x[1]*2+x[0], 2**3)

def ansi(n=0):
    '''returns the ANSI escape code for n (used for text colouring)'''
    try:
        n = int(n)
        if n>=0 and n<10:
            return f'\033[0{str(n)}m'
        elif n>=10 and n<100:
            return f'\033[{str(n)}m'
        else:
            return ''
    except:
        return ''

from abc import ABC, abstractmethod

class ProblemParameters(ABC):
    def __init__(self, qubits, set_of_gates):
        """if set_of_gates is a dictionary, any invalid keys are remapped;
           if set_of_gates is a list, keys are assigned (single digit ints
           if there are less than 10 gate, otherwise a range of english and
           greek letters are used - the set can contain at most 79 gates)"""
        self.qubit_count = qubits
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
                        while i<79:
                            if encodeToLetter(i) not in set_of_gates:
                                if encodeToLetter(i) not in set_of_gates_dict:
                                    set_of_gates_dict[encodeToLetter(i)] = set_of_gates[old_key]
                                    break
                            i+=1
                        if i==79:
                            raise RuntimeError('Could not remap gate')
                else:
                    set_of_gates_dict[key] = set_of_gates[old_key]
            self.gate_set = set_of_gates_dict
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
                    set_of_gates_dict[encodeToLetter(i)] = set_of_gates[i]
            self.gate_set = set_of_gates_dict
        else:
            raise TypeError('set_of_gates is not a dictionary or list')
        print('{')
        for symbol in self.gate_set:
            print(f'    {ansi(92)}{symbol}{ansi()} : {ansi(96)}{self.gate_set[symbol].base_class.__name__}{ansi()}')
        print('}')
        self.all_gate_combinations = self.generate_gate_combinations()

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
    
    def msf(self, candidate_circuit, input_states, output_states):
        """mean square fidelity function over a set of input and output states"""
        M = Operator(candidate_circuit)
        fidelity_sum = 0
        # ADD BACK IN FOR ROBUSTNESS
        #if len(input_states)!=len(output_states):
        #    raise ValueError('Inconsistent size of input_states and output_states')
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
        return Genotype(self, '').get_fitness()

    # function to check correctness? unsure how to intergrate with specific and non-specific msf

class AppliedProblemParameters(ProblemParameters):
    def __init__(self, set_of_gates, input_states, output_states, N=3):
        """if output_states is a circuit object, uses to evaluate truth table;
           otherwise, assumed to be a precalulated list of states"""
        self.input_states = input_states
        try:
            M = Operator(output_states)
            self.output_states = [s.evolve(M) for s in input_states]
            super().__init__(output_states.num_qubits, set_of_gates)
        except:
            self.output_states = output_states
            super().__init__(N, set_of_gates)

    def circuit_fitness(self, candidate_circuit):
        """overrides with the required truth table"""
        return self.msf(candidate_circuit, self.input_states, self.output_states)
    
def matrix_difference_fitness(m_1, m_2, tolerance=0.05):
    '''takes the difference betweens two matricies and counts the
       proportion of entries which are zero (elements are identical
       in both matricies, up to a tolerance)'''
    difference_matrix = (m_1-m_2).data.flatten()
    count = 0
    for x in difference_matrix:
        if abs(x) < tolerance:
            #count += 1
            count += (tolerance-abs(x)) / tolerance
    return count / len(difference_matrix)

class ProblemParametersMatrix(ProblemParameters):
    def __init__(self, set_of_gates, target_behaviour_circuit, N=3):
        super().__init__(N, set_of_gates)
        self.M = Operator(target_behaviour_circuit)

    def circuit_fitness(self, candidate_circuit):
        return matrix_difference_fitness(self.M, Operator(candidate_circuit))

class ProblemParametersCombined(AppliedProblemParameters):
    def __init__(self, set_of_gates, input_states, target_behaviour_circuit, mdf_tolerance=0.05):
        """evaluates the truth table on the provided input_states using target_behaviour_circuit"""
        self.M = Operator(target_behaviour_circuit)
        output_states = [s.evolve(self.M) for s in input_states]
        super().__init__(set_of_gates, input_states, output_states, target_behaviour_circuit.num_qubits)
        self.tolerance = mdf_tolerance

    def mdf(self, circuit):
        '''matrix difference fitness for circuit'''
        return matrix_difference_fitness(self.M, Operator(circuit), self.tolerance)

    def circuit_fitness(self, candidate_circuit):
        '''re-overrides with combination of msf and mdf'''
        msf = self.msf(candidate_circuit, self.input_states, self.output_states)
        mdf = self.mdf(candidate_circuit)
        return msf*mdf

def plot_list(float_list, x_label=None, y_label=None):
    """plots a list of floats (between 0 and 1)"""
    if type(float_list[0])==list:
        x_axis = [i for i in range(len(float_list[0]))]
        for j in range(len(float_list)):
            plt.plot(x_axis, float_list[-(j+1)], linewidth=20/(20+len(float_list)))
    else:
        x_axis = [i+1 for i in range(len(float_list))]
        plt.plot(x_axis, float_list)
    
    while len(x_axis) > 20:
        x_axis = [i*5 for i in range(len(x_axis)//5+1)]
    plt.xticks([0]+x_axis)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    try:
        max_value = max(1, max(float_list))
    except:
        max_value = max([max(float_list[i]) for i in range(len(float_list))]+[1])
    plt.xlim([x_axis[0],x_axis[-1]])
    plt.ylim([0,max_value])
    plt.yticks([x/10 for x in range(1+math.ceil(10*max_value))])
    plt.grid()
    plt.show()

def remove_duplicates(genotype_list):
    '''efficient way to do this for non-hashable objects??'''
    seen_genotypes = []
    out = []
    for i in range(len(genotype_list)):
        if genotype_list[i].genotype_str not in seen_genotypes:
            seen_genotypes.append(genotype_list[i].genotype_str)
            out.append(genotype_list[i])
    return out


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

    @staticmethod
    def sort_by_fitness(population, min_fitness=0, prefer_short_circuits=False, prefer_long_circuits=False, remove_dupe=True, use_qiskit_depth=False):
        """sorts population by fitness, also removed duplicates / sorts by circuit depth if specified"""
        by_fitness = population.copy()
        if remove_dupe:
            by_fitness = remove_duplicates(by_fitness)
        if prefer_short_circuits != prefer_long_circuits:
            if use_qiskit_depth:
                by_fitness = sorted(by_fitness, key=lambda genotype: genotype.to_circuit().depth, reverse=prefer_long_circuits)
            else:
                by_fitness = sorted(by_fitness, key=lambda genotype: len(genotype.genotype_str), reverse=prefer_long_circuits)
        by_fitness = sorted(by_fitness, key=lambda genotype: genotype.fitness, reverse=True)
        while by_fitness[-1].fitness < min_fitness:
            by_fitness.pop(-1)
        return by_fitness
    
    def top_by_fitness(self, population, min_fitness=0, prefer_short_circuits=False, prefer_long_circuits=False, remove_dupe=True, use_qiskit_depth=False):
        """finds the best circuits in the population; top sample taken as well as uniform [CHANGE THIS TO RAMPED] selection of remaining circuits"""
        by_fitness = Evolution.sort_by_fitness(population, min_fitness, prefer_short_circuits, prefer_long_circuits, remove_dupe, use_qiskit_depth)
        step = (len(by_fitness)-self.SAMPLE_SIZE)//(self.GENERATION_SIZE-self.SAMPLE_SIZE)
        step = 1 if step==0 else step
        end = (1-step)*self.SAMPLE_SIZE + step*self.GENERATION_SIZE
        return by_fitness[:self.SAMPLE_SIZE] + by_fitness[self.SAMPLE_SIZE:end:step]
        
    def random_search(self, output=True, plot_fitness=True):
        fitness_trace = [[] for i in range(self.SAMPLE_SIZE)]
        population = []

        for generation in range(self.GENERATION_COUNT):
            for _ in range(self.GENERATION_SIZE):
                g = Genotype(self.metadata, min_length=30, max_length=45, falloff='linear')
                g.get_fitness()
                population.append(g)
            # sort population by fitness, take top 5
            population = self.top_by_fitness(population)
            # each run compares the 100 new programs with the
            # 5 carried forward from the previous generation
            if output:
                print(f'Generation {generation+1} best: {population[0].genotype_str}')
                if plot_fitness:
                    for x in range(self.SAMPLE_SIZE):
                        try:
                            fitness_trace[x].append(population[x].fitness)
                        except:
                            fitness_trace[x].append(0)

        if output:
            s = min(self.SAMPLE_SIZE, len(population))
            print(f'top {s}:')
            for i in range(s):
                print(population[i].genotype_str)
                print(population[i].fitness)

            print('best random circuit:')
            #population[0]['circuit'].draw(output='mpl',style='iqp')
            print(population[0].to_circuit())

            if plot_fitness:
                plot_list(fitness_trace, 'Generations', 'Circuit Fitness')
        return population
    
    def stochastic_hill_climb(self, output=True, plot_fitness=True):
        best_genotype = Genotype(self.metadata, '')
        best_genotype.fitness = 0.0
        fitness_trace = []

        for generation in range(self.GENERATION_COUNT):
            population = []
            for _ in range(self.GENERATION_SIZE):
                g = Genotype(self.metadata, min_length=30, max_length=45, falloff='linear')
                m = g.get_fitness()
                m_delta = m - best_genotype.fitness
                if m_delta > 0:
                    # only take better circuits
                    population.append(g)


            if len(population) > 0:
                population = self.top_by_fitness(population)
                # select a random genotype, using the fitness improvements as weights
                best_genotype = random.choices(population, weights=[(g.fitness - best_genotype.fitness) for g in population], k=1)[0]

            if output:
                print(f'Generation {generation+1} best: {best_genotype.genotype_str}')
                if plot_fitness:
                    try:
                        fitness_trace.append(best_genotype.fitness)
                    except:
                        fitness_trace.append(0)

        if output:
            print('best random circuit:')
            print(best_genotype.genotype_str)
            print(best_genotype.to_list())
            print(best_genotype.to_circuit())
            print(best_genotype.fitness)
            #best_genotype['circuit'].draw(output='mpl',style='iqp')
            if plot_fitness:
                plot_list(fitness_trace, 'Generations', 'Circuit Fitness')
        return population
    
    ### ---------- EVOLUTIONARY SEARCH ----------

    def develop_circuits_uniform(self, inital_population, use_double_point_crossover=True):
        '''use a prespecified distribution of search operators
        population should be sorted by fitness'''
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

    def develop_circuits_random(self, inital_population, operation_count, use_double_point_crossover=True):
        '''use a random assortment of search operators'''
        population = inital_population.copy()
        for o in range(operation_count):
            # randomly select from the search operators
            operation = random.choices(population=['crossover', 'mutation', 'insersion', 'deletion'], weights=[0.4,0.5,0.05,0.05], k=1)[0]
            # randomly select a genotype
            g_1 = random.choices(inital_population, weights=[g.fitness for g in inital_population], k=1)[0]
            if operation == 'crossover':
                g_2 = g_1
                while g_2 == g_1:
                    g_2 = random.choices(inital_population, weights=[g.fitness for g in inital_population], k=1)[0]
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
        return population

    def develop_circuits_combined(self, inital_population, operation_count=250, double_point_crossover=True):
        #population_uniform = self.develop_circuits_uniform(inital_population, double_point_crossover)#[len(inital_population):]
        #len(population_uniform)//10
        population_random = self.develop_circuits_random(inital_population, operation_count, double_point_crossover)
        #return population_uniform + population_random
        return population_random
    
    def evolutionary_search(self, min_length=30, max_length=60, falloff=None, remove_duplicates=False,
                            MINIMUM_FITNESS=0, output=True, plot_fitness=True, random_sample_size=0,
                            use_double_point_crossover=True):
        fitness_trace = [[] for _ in range(self.SAMPLE_SIZE)]

        population = []
        while len(population) < self.SAMPLE_SIZE:
            for _ in range(self.GENERATION_SIZE):
                g = Genotype(self.metadata, min_length=min_length, max_length=max_length, falloff=falloff)
                g.get_fitness()
                population.append(g)
            population = self.top_by_fitness(population)
            if population[-1].fitness >= MINIMUM_FITNESS:
                break
            else:
                for i in range(len(population)):
                    if population[i].fitness < MINIMUM_FITNESS:
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

        start_time = time()
        for i in range(self.GENERATION_COUNT):
            if (i-1)%5==0:
                if i!=1:
                    remaining_time = (time()-start_time) * (self.GENERATION_COUNT-i)/(i+1)
                    remaining_time = remaining_time_calc(remaining_time)
                    if remaining_time:
                        print(f"[ estimated time remaining for run ~ {remaining_time} ]")

            # added random sample
            for _ in range(random_sample_size):
                g = Genotype(self.metadata, min_length=min_length, max_length=max_length, falloff=falloff)
                g.get_fitness()
                population.append(g)

            population = self.develop_circuits_combined(population, operation_count=int(self.GENERATION_SIZE*self.GENERATION_MULTIPLIER),
                                                        double_point_crossover=use_double_point_crossover)
            for g in population:
                g.get_fitness()
                #population.append(g)

            if output:
                print(f'Generation {i+1} Size (pre-selection): {len(population)}')
            
            population = self.top_by_fitness(population, min_fitness=MINIMUM_FITNESS, remove_dupe=remove_duplicates)#, prefer_short_circuits=True)
            
            if output:
                print(f'Generation {i+1} Best Genotype: {population[0].genotype_str}')
                print(f'Generation {i+1} Best Fitness: {population[0].fitness}')
                if plot_fitness:
                    for k in range(self.SAMPLE_SIZE):
                        try:
                            fitness_trace[k].append(population[k].fitness)
                        except:
                            fitness_trace[k].append(0)
                        

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

        return population