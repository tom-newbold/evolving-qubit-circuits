from qiskit.circuit.library import QFT as QFT_blueprint
from qiskit.circuit import CircuitInstruction
from qiskit import transpile
#from qiskit.converters import circuit_to_dag
from quantum_fourier_transform import GATE_SET, QFTGeneration
from linear_genetic_programming import Genotype, AppliedProblemParameters

from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.visualization import dag_drawer
from qiskit.transpiler.passes import CommutationAnalysis
from qiskit.transpiler import PassManager

import math
from random import choice, randint

#del GATE_SET[3]
##from qiskit.circuit.library import SGate
#GATE_SET.append(SGate())

def unoptimiser(initial_circuit, app, N=3):
    '''produces an equivilent circuit (up to global phase) with 2 * N**2 extra gates.'''

    circuit = initial_circuit.copy()
    genotype = Genotype(app)
    genotype.from_circuit(circuit)

    identity_chains = []
    seen_gate_names = []
    for key in app.gate_set:
        gate = app.gate_set[key]
        if len(gate.params)>0:
            continue
        if gate.inverse() in app.gate_set.values() and gate.inverse().name not in seen_gate_names:
            seen_gate_names.append(gate.name)
            base_gate_index = next(gate_index for gate_index in app.gate_set if gate.name == app.gate_set[gate_index].name)
            inv_gate_index = next(gate_index for gate_index in app.gate_set if gate.inverse().name == app.gate_set[gate_index].name)
            for i in range(circuit.num_qubits):
                if gate.num_qubits==2:
                    j_range = list(range(circuit.num_qubits))
                    j_range.remove(i)
                    for j in j_range:
                        identity_chains.append([f'{base_gate_index}{i}{j}',f'{inv_gate_index}{i}{j}'])
                        #identity_chains[-1] = [f'{x}{j}' for x in identity_chains[-1]]
                else:
                    identity_chains.append([f'{base_gate_index}{i}',f'{inv_gate_index}{i}'])
                    identity_chains.append([f'{base_gate_index}{i}',f'{inv_gate_index}{i}']) # TWICE to balance gate probabilites with two-qubit gates
    #print(identity_chains)

    for _ in range(N**2):
        random_addition = choice(identity_chains)

        i = randint(0, len(genotype.to_list())-1)
        genotype.from_genotype(''.join(genotype.to_list()[:i]+random_addition+genotype.to_list()[i:]))
        circuit = genotype.to_circuit()

    #print(circuit)
        for i in range(N):
            #print(f'swapping: {genotype.get_fitness()}')
            pm = PassManager([CommutationAnalysis()])
            pm.run(circuit)
            wires = list(pm.property_set['commutation_set'].keys())[:circuit.num_qubits]
            '''
            for k in list(pm.property_set['commutation_set'].keys())[circuit.num_qubits:-2*circuit.num_qubits]:
                print(f'key: {k}')

            for wire in wires: print(wire)
            '''
            possible_commutations = []
            for wire in wires:
                try:
                    #print('wire')
                    for gate_list in pm.property_set['commutation_set'][wire]:
                        if len(gate_list)>1:
                            possible_commutations.append(gate_list)
                except:
                    pass
                    #print(wire)
                    #print(pm.property_set['commutation_set'][wire])

            # using dag circuit to swap
            #print(circuit)
            #or i in range(10):
            dag_form = circuit_to_dag(circuit)

            #dag_drawer(dag_form, scale=0.7, filename=None, style='color')

            to_swap = choice(possible_commutations)
            if len(to_swap) > 2:
                i = randint(0, len(to_swap)-2)
                to_swap = to_swap[i:i+2]

            #print(genotype.to_circuit())
            #print(genotype.to_list())
            #print(to_swap)

            try:
                dag_form.swap_nodes(*to_swap)
                if circuit_to_dag(dag_to_circuit(dag_form))==dag_form:
                    circuit = dag_to_circuit(dag_form, False)
                    genotype.from_circuit(circuit)
                    #print(circuit)
                    #print(f'> new fitness: {genotype.get_fitness()}')
                
                else:
                    #print('> error')
                    i -= 1
            except:
                pass
                #print('> couldnt swap')

            #print(f'identity size = {circuit_to_dag(dag_to_circuit(dag_form)).size()}')
    genotype.from_circuit(circuit)
    return circuit, genotype

if __name__=="__main__":
    N = 3
    qft = QFT_blueprint(N)
    qft = transpile(qft, basis_gates=[gate.name for gate in GATE_SET], optimization_level=0)

    qft_gen = QFTGeneration(GATE_SET, N)
    #genotype = Genotype(qft_gen)
    #genotype.from_circuit(qft)
    
    qft_gen.print_gate_set()


    #print(genotype.to_list())
    
    circuit, genotype = unoptimiser(qft, qft_gen, 3)
    print(circuit)
    print('^^ unoptimised')

    
    print(f'r_unopt = {circuit.depth()/qft.depth()}')
    # greater value means better unoptimisation

    qiskit_optimised = transpile(circuit, basis_gates=[gate.name for gate in GATE_SET], optimization_level=3)
    print(qiskit_optimised)
    while qiskit_optimised.depth()!=circuit.depth():
        circuit = qiskit_optimised.copy()
        qiskit_optimised = transpile(qiskit_optimised, basis_gates=[gate.name for gate in GATE_SET], optimization_level=3)
        print(qiskit_optimised)
    print(f'r_opt = {qiskit_optimised.depth()/qft.depth()}')
