from quantum_fourier_transform import QFTGeneration, GATE_SET, GATE_SET_SIMPLE
from linear_genetic_programming import Evolution, plot_many_averages
from grid_search import multiple_runs

from pandas import DataFrame

def run_algorithm_test(set, gen_multiplier=8, iterations=20):
    """performs multiple runs on each algorithm"""
    stats = {}
    to_plot = {}
    for algorithm in ['random','stochastic','evolution']:
        print(f'<{algorithm}>')
        QFT_GEN = QFTGeneration(set, 3)
        E = Evolution(QFT_GEN, sample_percentage=0.1, gen_mulpilier=gen_multiplier)

        to_plot[algorithm], stats[algorithm] = multiple_runs(E, method=algorithm, iterations=iterations, plot=False)
    return stats, to_plot

def run_gateset_test(sets, gen_multiplier=8, iterations=20):
    """performs multiple runs on each input gate set"""
    stats = {}
    to_plot = {}
    for set_name in sets:
        print(f'<{set_name}>')
        QFT_GEN = QFTGeneration(sets[set_name], 3)
        QFT_GEN.print_gate_set()
        E = Evolution(QFT_GEN, sample_percentage=0.1, gen_mulpilier=gen_multiplier)

        to_plot[set_name], stats[set_name] = multiple_runs(E, iterations=iterations, plot=False)
    return stats, to_plot

def run_qubitcount_test(set, gen_multiplier=8, iterations=20):
    """performs multiple runs on each circuit size"""
    stats = {}
    to_plot = {}
    for qubit_count in [3, 4, 5]:
        qubit_count_str = f'{qubit_count}qubits'
        print(f'<{qubit_count_str}>')
        QFT_GEN = QFTGeneration(set, qubit_count)
        E = Evolution(QFT_GEN, sample_percentage=0.1, gen_mulpilier=gen_multiplier)

        to_plot[qubit_count_str], stats[qubit_count_str] = multiple_runs(E, iterations=iterations, plot=False)
    return stats, to_plot

def run_distribution_test(set, gen_multiplier=8, iterations=20):
    """performs multiple runs on each random distribution"""
    stats = {}
    to_plot = {}
    for crossover in [3, 4, 5, 6, 7]:
        for ins_del in [1]:
            for x in ['single','double']:
                dist_str = f'crossover{crossover}insertdelete{ins_del}{x}'
                print(f'<{dist_str}>')
                QFT_GEN = QFTGeneration(set, 3)
                E = Evolution(QFT_GEN, sample_percentage=0.1, gen_mulpilier=gen_multiplier)

                to_plot[dist_str], stats[dist_str] = multiple_runs(E, crossover_proportion=crossover, 
                                                                   insert_delete_proportion=ins_del,
                                                                   use_double_point_crossover= x=='double',
                                                                   iterations=iterations, plot=False)
    return stats, to_plot

def output(p, s, test_param, multiplier):
    print(f'--{test_param}-- multiplier:{multiplier}')
    df = DataFrame.from_dict(s[test_param])
    print(df)
    with open(f'out/iter{ITERATIONS}_{test_param}_mult{multiplier}.csv','w') as file:
        file.write(DataFrame.to_csv(df))
        file.close()
    print(f'plotting...')
    plot_many_averages(p[test_param], 'Generations', 'Circuit Fitness', legend=False)

if __name__=="__main__":
    import os
    os.makedirs('out', exist_ok=True)

    TEST_FUNC = [run_gateset_test,run_algorithm_test,run_qubitcount_test,run_distribution_test]
    ITERATIONS = 10
    test_multipliers = [5]#[2,4,8]
    to_plot = []
    all_stats = []
    
    sets = {'reduced':GATE_SET_SIMPLE,'overcomplete':GATE_SET}
    
    for multiplier in test_multipliers:
        print(f'\n\nmultiplier:{multiplier}')
        t_func = TEST_FUNC[3]
        if t_func == run_gateset_test:
            s, p = t_func(sets, multiplier, ITERATIONS)
        else:
            s, p = t_func(GATE_SET, multiplier, ITERATIONS)
        to_plot.append(p)
        all_stats.append(s)

    for i in range(len(to_plot)):
        p = to_plot[i]
        s = all_stats[i]
        multiplier = test_multipliers[i]

        for test_param in list(s.keys()):
            output(p, s, test_param, multiplier)