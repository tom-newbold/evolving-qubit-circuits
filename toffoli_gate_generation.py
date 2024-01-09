from linear_genetic_programming import ProblemParameters, Evolution, list_to_state

class ToffoliGeneration(ProblemParameters):
    def __init__(self, set_of_gates):
        super().__init__(3, set_of_gates)

        self.toffoli_inputs = [[i//4 %2, i//2 %2, i%2] for i in range(8)]
        self.toffoli_outputs = []
        for i in range(8):
            x = self.toffoli_inputs[i].copy()
            if bool(x[0]) and bool(x[1]):
                x[2] = int(not bool(x[2]))
            self.toffoli_outputs.append(x)
    
    def specific_msf(self, candidate_circuit):
        return self.msf(candidate_circuit,
                        [list_to_state(x) for x in self.toffoli_inputs],
                        [list_to_state(y) for y in self.toffoli_outputs])
    

if __name__=="__main__":
    GATE_SET_SIMPLE = [{'label':'had','inputs':1},
                       {'label':'cnot','inputs':2},
                       {'label':'t','inputs':1},
                       {'label':'t_prime','inputs':1}]
                              
    GATE_SET = [{'label':'had','inputs':1},
                {'label':'not','inputs':1},
                {'label':'cnot','inputs':2},
                {'label':'phase','inputs':1,'parameters':1},
                {'label':'t','inputs':1},
                {'label':'t_prime','inputs':1},
                {'label':'chad','inputs':2},
                {'label':'cphase','inputs':2,'parameters':1}]
    
    TOFFOLI = ToffoliGeneration(GATE_SET)
    E = Evolution(TOFFOLI)#, number_of_generations=10)
    
    #population = E.random_search()
    #population = E.stochastic_hill_climb()
    population = E.evolutionary_search()

    ### --- check best circuit ---
    from qiskit.quantum_info import Operator
    import numpy as np

    input_states = [list_to_state(x) for x in TOFFOLI.toffoli_inputs]
    output_states = [list_to_state(y) for y in TOFFOLI.toffoli_outputs]

    c = population[0].to_circuit()
    differences = {'global':0,'local':0}
    M = Operator(c)
    for i in range(len(input_states)):
        state = input_states[i]
        calc_state = state.evolve(M)
        print(f"target: {output_states[i].draw(output='latex_source')} --> actual: {calc_state.draw(output='latex_source')}")
        #output_states[i].draw(output='latex')
        if calc_state==output_states[i]:
            #print('exacly correct')
            pass
        else:
            if abs(np.inner(output_states[i].data, calc_state.data).item())**2 == 1.0:
                print('state correct up to global phase')
                differences['global'] += 1
            else:
                #print('state incorrect')
                differences['local'] += 1

    print(f"exactly correct for {8-differences['global']-differences['local']} states")
    if differences['global']+differences['local'] != 0:
        print(f"correct (up to global phase) for {8-differences['local']} states")