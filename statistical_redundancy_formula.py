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
    from quantum_fourier_transform import GATE_SET
    redundancy = Redundancy(100000)#gate_set=GATE_SET)
    
    #target_filename = f'clifford-t-redundancy-model-{redundancy.TRAINING_SET_SIZE}samples-{redundancy.rand_gen.qubit_count}qubits'
    target_filename = 'neural-network-test-2'
    #target_filename = 'qft-redundancy-model'
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

        plt.clf()
        plt.plot(redundancy.model.loss_curve_)
        plt.show()
    
    
    square_error, points = redundancy.validate()
    avr = sum(points['pure'])/len(points['pure'])
    ss_tot = sum([(p-avr)**2 for p in points["pure"]])
    print(f'r2: {1 - sum(square_error)/ss_tot}')

    plt.clf()
    plt.title('error')
    plt.boxplot(square_error)
    plt.ylim(0, max(square_error))
    plt.show()

    plt.clf()
    plt.title('error')
    line = [min(0, max(min(points['pure']), min(points['predicted']))),
            max(1, min(max(points['pure']), max(points['predicted'])))]
    plt.plot(line, line)
    plt.scatter(points['pure'], points['predicted'])
    plt.xlabel('pure')
    plt.ylabel('predicted')
    plt.show()

    # testing individual variable relations
    import pandas
    import numpy as np
    from linear_genetic_programming import Genotype
    df = pandas.DataFrame(columns=redundancy.rand_gen.all_gate_combinations + ['total', 'redundancy', 'error', 'fitness'], dtype='Int64')
    error = []
    # generate datapoints
    for i in range(100):
        rand_genotype = Genotype(redundancy.rand_gen)
        distribution = Redundancy.count_gate_combinations(rand_genotype)
        distribution['redundancy'] = rand_genotype.remove_redundant_gates()[1]
        distribution['fitness'] = rand_genotype.get_fitness()
        distribution['error'] = redundancy.calc_error(rand_genotype)

        df.loc[i] = distribution
        
    plt.clf()
    plt.scatter(df['fitness'], df['error'])
    plt.xlabel('fitness')
    plt.ylabel('redundancy error')
    plt.show()

    '''
    for var in redundancy.rand_gen.all_gate_combinations:
        plt.clf()
        plt.title(var)
        plt.scatter(df[var], df['redundancy'])
        x = np.linspace(0, int(max(df[var])), 2)
        y = redundancy.model.coef_[redundancy.rand_gen.all_gate_combinations.index(var)] * x
        plt.plot(x, y)
        plt.ylabel('redundancy')
        plt.show()
    '''
    
    import time
    t_1 = 0
    t_2 = 0
    for _ in range(1000):
        rand_genotype = Genotype(redundancy.rand_gen)
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
        redundancy.model.predict([values])[0]
        t_2 += time.time() - start_time
    print(f'pure time: {t_1}')
    print(f'aproximation time: {t_2}')
    print(f'{t_1/t_2}x speedup')