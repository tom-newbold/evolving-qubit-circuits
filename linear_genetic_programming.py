from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

import random, math
import numpy as np

class Genotype:
    def __init__(self, genotype_string, problem_parameters):
        self.genotype_str = genotype_string
        self.metadata = problem_parameters

    ### ---------- GENOTYPE UTILS ----------

    def to_list(self):
        """takes the genotype and splits using the number of arguments per gate"""
        out = []
        i = 0
        while i<len(self.genotype_str):
            gate = self.genotype_str[i]
            j = i + self.metadata.gate_set[int(gate)]['inputs'] + 1
            if self.metadata.gate_set[int(gate)]['parameters']:
                j += self.metadata.gate_set[int(gate)]['parameters']
            k = self.genotype_str[i:j]
            i = j
            out.append(k)
        return out

    def to_circuit(self):
        """decodes a genotype string to circuit form"""
        circuit_instance = QuantumCircuit(self.qubit_count)
        for k in self.to_list():
            gate = int(k[0])
            #if len(k)-1 != self.gate_set[gate]['inputs']:
            #    print('ERROR')
            circuit_instance = self.construct_gate(circuit_instance)
        return circuit_instance
    
    def construct_gate(self, c_instance):
        """constructs a single gate from a string and appends to the given ciruit"""
        g_label = self.gate_set[int(self.genotype_str[0])]['label']
        
        if g_label=='not':
            c_instance.x(int(self.genotype_str[1]))
        elif g_label=='cnot':
            c_instance.cx(int(self.genotype_str[1]),int(self.genotype_str[2]))
        elif g_label=='had':
            c_instance.h(int(self.genotype_str[1]))
        elif g_label=='chad':
            c_instance.ch(int(self.genotype_str[1]),int(self.genotype_str[2]))
        elif g_label=='phase':
            c_instance.p(math.pi/self.genotype_str[2],int(self.genotype_str[1]))
        elif g_label=='cphase':
            c_instance.cp(math.pi/self.genotype_str[3],int(self.genotype_str[1]),int(self.genotype_str[2]))
        elif g_label=='t':
            c_instance.t(int(self.genotype_str[1]))
        elif g_label=='t_prime':
            c_instance.tdg(int(self.genotype_str[1]))
            
        return c_instance
    
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
        self.all_gate_combinations = self.generate_gate_combinations(set_of_gates)

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
        '''mean square fidelity function over a set of input and output states'''
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