import os

from linear_genetic_programming import Redundancy


os.makedirs('models', exist_ok=True)

def analyse_gates(genotype):
    '''DEPRECATED'''
    distribution = dict(genotype.to_circuit().count_ops())
    distribution['total'] = sum([distribution[key] for key in distribution])
    reduced, redundancy = genotype.remove_redundant_gates()
    distribution['redundancy'] = redundancy
    return distribution


if __name__=="__main__":
    import matplotlib.pyplot as plt

    redundancy = Redundancy(N=3)
    
    target_filename = f'clifford-t-redundancy-model-{redundancy.TRAINING_SET_SIZE}samples-{redundancy.rand_gen.qubit_count}qubits'
    if len(redundancy.load_model(target_filename))==0:
        df, coef = redundancy.train()
        print(coef)

        '''
        plt.title('sample length range')
        plt.boxplot(df['total'].values)
        plt.ylim(0, 100)
        plt.show()
        '''

        redundancy.save_model(target_filename)
    
    
    square_error = redundancy.validate()

    plt.clf()
    plt.title('error')
    plt.boxplot(square_error)
    plt.ylim(0, max(square_error))
    plt.show()

    '''
    import time
    t_1 = 0
    t_2 = 0
    for _ in range(1000):
        rand_genotype = Genotype(app)
        start_time = time.time()
        rand_genotype.remove_redundant_gates()
        t_1 += time.time() - start_time
        start_time = time.time()
        genotype_dict = {gate:0 for gate in rand_genotype.metadata.all_gate_combinations}
        i = 0
        j = 1
        while j<len(rand_genotype.genotype_str):
            if rand_genotype.genotype_str[i:j] in genotype_dict:
                genotype_dict[rand_genotype.genotype_str[i:j]] += 1
                i = j
            j += 1
        values = [genotype_dict[gate] for gate in rand_genotype.metadata.all_gate_combinations]
        regr.predict([values])[0]
        t_2 += time.time() - start_time
    print(f'pure time: {t_1}')
    print(f'aproximation time: {t_2}')
    print(f'{t_1/t_2}x speedup')
    '''