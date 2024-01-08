from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

import matplotlib.pyplot as plt
import random, math
import numpy as np

class Genotype:
    def __init__(self, problem_parameters, genotype_string=None, min_length=15, max_length=45, falloff='linear'):            
        self.genotype_str = genotype_string
        self.circuit = None
        self.metadata = problem_parameters
        self.msf = None
        if self.genotype_str==None:
            self.generate_random_genotype(min_length, max_length, falloff)#PARAMETERS

    ### ---------- GENOTYPE UTILS ----------

    def to_list(self):
        """takes the genotype and splits using the number of arguments per gate"""
        out = []
        i = 0
        while i<len(self.genotype_str):
            gate = self.genotype_str[i]
            j = i + self.metadata.gate_set[int(gate)]['inputs'] + 1
            if 'parameters' in self.metadata.gate_set[int(gate)]:
                j += self.metadata.gate_set[int(gate)]['parameters']
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
                gate = int(k[0])
                #if len(k)-1 != self.gate_set[gate]['inputs']:
                #    print('ERROR')
                circuit_instance = self.construct_gate(circuit_instance)
            self.circuit = circuit_instance
            return circuit_instance
    
    def construct_gate(self, c_instance):
        """constructs a single gate from a string and appends to the given ciruit"""
        g_label = self.metadata.gate_set[int(self.genotype_str[0])]['label']
        
        if g_label=='not':
            c_instance.x(int(self.genotype_str[1]))
        elif g_label=='cnot':
            c_instance.cx(int(self.genotype_str[1]),int(self.genotype_str[2]))
        elif g_label=='had':
            c_instance.h(int(self.genotype_str[1]))
        elif g_label=='chad':
            c_instance.ch(int(self.genotype_str[1]),int(self.genotype_str[2]))
        elif g_label=='phase':
            c_instance.p(math.pi/int(self.genotype_str[2]),int(self.genotype_str[1]))
        elif g_label=='cphase':
            c_instance.cp(math.pi/int(self.genotype_str[3]),int(self.genotype_str[1]),int(self.genotype_str[2]))
        elif g_label=='t':
            c_instance.t(int(self.genotype_str[1]))
        elif g_label=='t_prime':
            c_instance.tdg(int(self.genotype_str[1]))
            
        return c_instance
    
    #def generate_random_genotype(self, min_length=15, max_length=45, falloff='linear'):#, input_count_weighted=True):
    def generate_random_genotype(self, min_length, max_length, falloff):#, input_count_weighted=True):
        gradient = -1/(max_length-min_length)
        intercept = -max_length*gradient
        g = ''
        while True:
            #if input_count_weighted:
            new_gate = random.choice(self.metadata.all_gate_combinations)
            g += new_gate
            #else:
            #    gate = random.randint(0,len(self.metadata.gate_set)-1)
            #    inputs = []
            #    while len(inputs) < self.metadata.gate_set[gate]['inputs']:
            #        x = str(random.randint(0,self.metadata.qubit_count-1))
            #        if x not in inputs:
            #            inputs.append(x)
            #    g += str(gate) + ''.join(inputs)
            if 'parameters' in self.metadata.gate_set[int(new_gate[0])]:
                for i in range(self.metadata.gate_set[int(new_gate[0])]['parameters']):
                    g += str(random.randint(1,9))

            if falloff=='linear':
                if random.random() > intercept + gradient*len(g):
                    break
            elif falloff=='logarithmic':
                if random.random() > math.log10(1-9*(len(g)-max_length)/(max_length-min_length)):
                    break
            elif falloff=='reciprocal':
                if random.random() > min_length/len(g):
                    break
            else:
                if len(g) > max_length:
                    break
        self.genotype_str = g
        self.circuit = None
    
    ### ---------- EVOLUTIONARY OPERATORS ----------
    
    @staticmethod
    def crossover(genotype_1, genotype_2, uniform=True):
        """computes the random crossover of two genotypes; the point is selected
           separately on each genotype, allowing for length variations"""
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
            return Genotype(''.join(new_genotype_1)), Genotype(''.join(new_genotype_2))
    
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
                # print('mutate gate')
                old_gate_index = int(gate[0])
                new_gate_index = random.randint(0,len(genotype.metadata.gate_set)-1)
                if new_gate_index==old_gate_index:
                    continue
                new_input_count = genotype.metadata.gate_set[new_gate_index]['inputs']
                old_param_count = 0
                new_param_count = 0
                if genotype.metadata.gate_set[old_gate_index]['parameters']:
                    old_param_count = genotype.metadata.gate_set[old_gate_index]['parameters']
                if genotype.metadata.gate_set[new_gate_index]['parameters']:
                    new_param_count = genotype.metadata.gate_set[new_gate_index]['parameters']
                prev_inputs = gate[1:-old_param_count]
                prev_params = gate[-old_param_count:]
                # adjust inputs
                if len(prev_inputs) > new_input_count:
                    prev_inputs = prev_inputs[:new_input_count]
                else:
                    while len(prev_inputs) < new_input_count:
                        # add inputs to satify new gate
                        i = str(random.randint(0,genotype.metadata.qubit_count-1))
                        if i not in prev_inputs:
                            prev_inputs += i
                # adjust params
                if len(prev_params) > new_param_count:
                    prev_params = prev_params[:new_param_count]
                else:
                    while len(prev_params) < new_param_count:
                        # add params
                        prev_params += str(random.randint(1,9))
                
                gate = str(new_gate_index) + prev_inputs + prev_params
                """# check length
                expected_len = genotype.metadata.gate_set[int(gate)]['inputs']+1
                if len(gate) > expected_len:
                    # print('mutate gate: truncating inputs')
                    # truncate extra inputs
                    gate = gate[:expected_len]
                else:
                    target_len = genotype.metadata.gate_set[new_gate]['inputs']+1
                    while len(gate) < target_len:
                        # add inputs to satify new gate
                        new_input = str(random.randint(0,genotype.metadata.qubit_count-1))
                        if str(new_input) not in gate[1:]:
                            gate += str(new_input)
                    while len(gate) < target_len+genotype.metadata.gate_set[int(gate)]['parameters']:
                        # add parameters if required
                        gate += str(random.randint(1,9))"""
            else:
                # mutate an input
                # print('mutate input')
                new_input = random.randint(0,genotype.metadata.qubit_count-1)
                if len(gate)==2:
                    gate = prev_gate[0] + str(new_input)
                else:
                    index_to_change = random.randint(1,len(gate)-1)
                    if str(new_input) not in prev_gate[1:index_to_change] + prev_gate[index_to_change+1:]:
                        # adds new input if not a duplicate
                        # print('inserting new input')
                        gate = prev_gate[:index_to_change] + str(new_input) + prev_gate[index_to_change+1:]

        genotype_list[mutation_point] = gate
        return ''.join(genotype_list)
    
    @staticmethod
    def insertion(genotype):
        """inserts a new random gate at a randomly chosen point in the genotype"""
        new_gate = random.randint(0,len(genotype.metadata.gate_set)-1)
        g_add = str(new_gate)
        inputs = []
        while len(inputs) < genotype.metadata.gate_set[new_gate]['inputs']:
            # generates the right number of inputs
            x = str(random.randint(0,genotype.metadata.qubit_count-1))
            if x not in inputs:
                inputs.append(x)
        params = []
        if genotype.metadata.gate_set[new_gate]['parameters']:
            while len(params) < genotype.metadata.gate_set[new_gate]['parameters']:
                params.append(str(random.randint(1,9)))
        g_add += ''.join(inputs) + ''.join(params)

        # insert at random position
        genotype_list = genotype.to_list()
        genotype_add_index = random.randint(0,len(genotype_list)-1)
        return ''.join(genotype_list[:genotype_add_index]) + g_add + ''.join(genotype_list[genotype_add_index:])

    @staticmethod
    def deletion(genotype):
        """removes a random gate from the genotype"""
        genotype_list = genotype.to_list()
        if len(genotype_list)<=1:
            return genotype
        genotype_remove_index = random.randint(0,len(genotype_list)-1)
        return ''.join(genotype_list[:genotype_remove_index] + genotype_list[genotype_remove_index+1:])



def list_to_state(x):
    return Statevector.from_int(x[2]*4+x[1]*2+x[0], 2**3)

from abc import ABC, abstractmethod

class ProblemParameters(ABC):
    def __init__(self, qubits, set_of_gates):
        self.qubit_count = qubits
        self.gate_set = set_of_gates
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
        for index, gate in enumerate(self.gate_set):
            if gate['inputs']==1:
                for q in range(self.qubit_count):
                    all_gates.append(str(index)+str(q))
            elif gate['inputs']==2:
                for q in double_input_combinations:
                    all_gates.append(str(index)+str(q))
        return all_gates
    
    def msf(self, candidate_circuit, input_states, output_states, test_all_states=True):
        """mean square fidelity function over a set of input and output states"""
        M = Operator(candidate_circuit)
        fidelity_sum = 0
        case_count = min(len(input_states),len(output_states))
        for i in range(case_count):
            state = input_states[i]
            calc_state = state.evolve(M)
            if calc_state==output_states[i]:
                fidelity_sum += 1.0 + 1/(2**(2*self.qubit_count))
            else:
                fidelity_sum += abs(np.inner(output_states[i].data, calc_state.data).item())**2
        return fidelity_sum/case_count
    
    @abstractmethod
    def specific_msf(self, candidate_circuit):
        pass


def plot_list(float_list, x_label=None, y_label=None):
    """plots a list of floats (between 0 and 1)"""
    if type(float_list[0])==list:
        x_axis = [i+1 for i in range(len(float_list[0]))]
        for j in range(len(float_list)):
            plt.plot(x_axis, float_list[-(j+1)], linewidth=20/(20+len(float_list)))
    else:
        x_axis = [i+1 for i in range(len(float_list) + 1)]
        plt.plot(x_axis, float_list)
    
    while len(x_axis) > 20:
        x_axis = [(i+1)*5 for i in range(len(x_axis)//5 + 1)]
    plt.xticks([0]+x_axis)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)

    try:
        max_value = max(1, max(float_list))
    except:
        max_value = max([max(float_list[i]) for i in range(len(float_list))]+[1])
    plt.ylim([0,max_value])
    plt.yticks([x/10 for x in range(1+math.ceil(10*max_value))])
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
    def __init__(self, problem_parameters, sample=20, number_of_generations=50, individuals_per_generation=100, alpha=2, beta=3, gamma=2):
        self.metadata = problem_parameters
        self.SAMPLE_SIZE = sample
        self.GENERATION_COUNT = number_of_generations
        self.GENERATION_SIZE = individuals_per_generation
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def top_by_fitness(self, population, prefer_short_circuits=False, prefer_long_circuits=False, remove_dupe=True):
        """finds the best circuits in the population"""
        if remove_dupe:
            population = remove_duplicates(population)
        by_fitness = sorted(population, key=lambda genotype: genotype.msf, reverse=True)
        if prefer_short_circuits == prefer_long_circuits:
            return by_fitness[:self.SAMPLE_SIZE]
        else:
            return sorted(by_fitness, key=lambda genotype: len(genotype.genotype_str), reverse=prefer_long_circuits)
        
    def random_search(self):
        return
    
    def stochastic_hill_climb(self):
        best_genotype = Genotype(self.metadata, '')
        best_genotype.msf = 0.0
        msf_trace = []

        for generation in range(self.GENERATION_COUNT):
            population = []
            for _ in range(self.GENERATION_SIZE):
                g = Genotype(self.metadata, min_length=30, max_length=45, falloff='linear')
                c = g.to_circuit()
                m = self.metadata.specific_msf(c)
                m_delta = m - best_genotype.msf
                if m_delta > 0:
                    # only take better circuits
                    population.append(g)


            if len(population) > 0:
                population = self.top_by_fitness(population)
                # select a random genotype, using the msf improvements as weights
                best_genotype = random.choices(population, weights=[(g.msf - best_genotype.msf) for g in population], k=1)[0]

            print(f'Generation {generation+1} best: {best_genotype.genotype_str}')
            msf_trace.append(best_genotype.msf)


        print('best random circuit:')
        print(best_genotype.to_circuit())
        print(best_genotype.msf)
        #best_genotype['circuit'].draw(output='mpl',style='iqp')

        plot_list(msf_trace, 'Generations', 'MSF')
        return population
    
    ### ---------- EVOLUTIONARY SEARCH ----------

    @staticmethod
    def develop_circuits_uniform(self, inital_population):
        '''use a prespecified distribution of search operators
        population should be sorted by msf'''
        population = [x['genotype'] for x in inital_population]
        # crossover operation for every pair of genotypes in the sample
        for g_1_index in range(len(inital_population)):
            for g_2_index in range(g_1_index+1,len(inital_population)):
                for c in range(self.gamma):
                    g_3, g_4 = Genotype.crossover(population[g_1_index],population[g_2_index], self.metadata.gate_set)
                    population.append(g_3)
                    population.append(g_4)
        # mutation operation for every genotype in the sample
        # insertion/deletion for each genotype in the sample
        for g_1_index in range(len(inital_population)):
            for a in range(self.alpha):
                g_2 = g_3 = g_4 = population[g_1_index]
                for b in range(self.beta):
                    g_2 = Genotype.mutation(g_2, self.metadata.gate_set)
                    g_3 = Genotype.insertion(g_3, self.metadata.gate_set)
                    g_4 = Genotype.deletion(g_4, self.metadata.gate_set)
                    for g in [g_2, g_3, g_4]:
                        population.append(g)
        return population

    @staticmethod
    def develop_circuits_random(self, inital_population, operation_count):
        '''use a random assortment of search operators'''
        population = [g.genotype_str for g in inital_population]
        for o in range(operation_count):
            # randomly select from the search operators
            operation = random.choices(population=['crossover', 'mutation', 'insersion', 'deletion'],weights=[0.4,0.5,0.05,0.05], k=1)[0]
            # randomly select a genotype
            g_1 = random.choices(inital_population, weights=[g.msf for g in inital_population], k=1)[0]['genotype']
            if operation == 'crossover':
                g_2 = g_1
                while g_2 == g_1:
                    g_2 = random.choices(inital_population, weights=[g.msf for g in inital_population], k=1)[0]['genotype']
                for c in range(self.gamma):
                    g_3, g_4 = Genotype.crossover(g_1, g_2, self.metadata.gate_set)
                    population.append(g_3)
                    population.append(g_4)
            else:
                for a in range(self.alpha):
                    g_2 = g_1
                    for b in range(self.beta):
                        if operation=='mutation':
                            g_2 = Genotype.mutation(g_2, self.metadata.gate_set)
                        elif operation=='insersion':
                            g_2 = Genotype.insertion(g_2, self.metadata.gate_set)
                        elif operation=='deletion':
                            g_2 = Genotype.deletion(g_2, self.metadata.gate_set)
                        population.append(g_2)
                '''
                if operation=='mutation':
                    for a in range(alpha):
                        g_2 = g_1
                        for b in range(beta):
                            g_2 = self.mutation(g_2, gate_set)
                            population.append(g_2)
                elif operation=='insersion':
                    for a in range(alpha):
                        g_2 = g_1
                        for b in range(beta):
                            g_2 = self.insertion(g_2, gate_set)
                            population.append(g_2)
                elif operation=='deletion':
                    for a in range(alpha):
                        g_2 = g_1
                        for b in range(beta):
                            g_2 = self.deletion(g_2, gate_set)
                            population.append(g_2)
                '''
        return population

    @staticmethod
    def develop_circuits_combined(self, inital_population, operation_count=250):
        population_uniform = self.develop_circuits_uniform(inital_population, self.metadata.gate_set)#[len(inital_population):]
        #print(2*len(population_uniform)//3)
        population_random = self.develop_circuits_random(inital_population, self.matadata.gate_set,
                                                         operation_count)[len(inital_population):]
        return population_uniform + population_random
    
    def evolutionary_search(self):
        return