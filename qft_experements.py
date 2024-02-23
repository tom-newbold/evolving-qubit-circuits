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
        E = Evolution(QFT_GEN, sample_percentage=0.1, gen_mulpilier=gen_multiplier)

        to_plot[set_name], stats[set_name] = multiple_runs(E, iterations=iterations, plot=False)
    return stats, to_plot

def run_qubitcount_test(set, gen_multiplier=8, iterations=20):
    """performs multiple runs on each circuit size"""
    stats = {}
    to_plot = {}
    for qubit_count in ['3qubits','4qubits','5qubits']:
        print(f'<{qubit_count}>')
        QFT_GEN = QFTGeneration(set, int(qubit_count[0]))
        E = Evolution(QFT_GEN, sample_percentage=0.1, gen_mulpilier=gen_multiplier)

        to_plot[qubit_count], stats[qubit_count] = multiple_runs(E, iterations=iterations, plot=False)
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

    TEST_FUNC = [run_gateset_test,run_algorithm_test,run_qubitcount_test][2]
    ITERATIONS = 5
    test_multipliers = [2]#[2,4,8]
    to_plot = []
    all_stats = []
    
    sets = {'reduced':GATE_SET_SIMPLE,'overcomplete':GATE_SET}
    
    for multiplier in test_multipliers:
        print(f'\n\nmultiplier:{multiplier}')
        #s, p = run_gateset_test(sets, multiplier, ITERATIONS)
        #s, p = run_algorithm_test(GATE_SET, multiplier, ITERATIONS)
        #s, p = run_qubitcount_test(GATE_SET, multiplier, ITERATIONS)
        s, p = TEST_FUNC(GATE_SET, multiplier, ITERATIONS)
        to_plot.append(p)
        all_stats.append(s)

    for i in range(len(to_plot)):
        p = to_plot[i]
        s = all_stats[i]
        multiplier = test_multipliers[i]

        #for set_name in list(s.keys()):
        #    output(p, s, set_name, multiplier)
        #for algorithm in list(s.keys()):
        #    output(p, s, algorithm, multiplier)
        #for qubit_count in list(s.keys()):
        #    output(p, s, qubit_count, multiplier)
        for test_param in list(s.keys()):
            output(p, s, test_param, multiplier)