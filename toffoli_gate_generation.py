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
        """overrides with the required truth table"""
        return self.msf(candidate_circuit,
                        [list_to_state(x) for x in self.toffoli_inputs],
                        [list_to_state(y) for y in self.toffoli_outputs])
    
def grid_search(evolution):
    """performs a grid search of the 'primary' parameters associated with genotype generation"""
    results = []
    lengths = ([0,15,30,45],[30,45,60])
    falloff = ['linear','logarithmic','reciprocal']

    i = 1
    total = len(lengths[1]) * (len(falloff)*len(lengths[0]) + 1)
    for max_len in lengths[1]:
        for f in falloff:
            for min_len in lengths[0]:
                if min_len>=max_len:
                    i+=1
                    continue
                print(f"TEST {i}/{total} - checking min:{min_len} max:{max_len} falloff:{f}")
                best_genotype = evolution.evolutionary_search(min_length=min_len, max_length=max_len, falloff=f, plot_msf=False)[0]
                results.append({'min':min_len, 'max':max_len, 'falloff':f, 'best':best_genotype})
                i+=1
        print(f"TEST {i}/{total} - checking max:{max_len} falloff:None")
        best_genotype = evolution.evolutionary_search(min_length=0, max_length=max_len, falloff=None, plot_msf=False)[0]
        results.append({'min':0, 'max':max_len, 'falloff':None, 'best':best_genotype})
        i+=1

    for r in results:
        print(f"min:{r['min']} max:{r['max']} falloff:{r['falloff']}")
        print(f"genotype: {r['best'].genotype_str}")
        print(f"msf: {r['best'].msf}")
    return results


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
    E = Evolution(TOFFOLI)
    
    #population = E.random_search()
    #population = E.stochastic_hill_climb()
    #population = E.evolutionary_search()
    grid_search(Evolution(TOFFOLI, sample=10, number_of_generations=20,
                          individuals_per_generation=50, alpha=1, beta=2))

    ### --- check best circuit ---
    """
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
    """