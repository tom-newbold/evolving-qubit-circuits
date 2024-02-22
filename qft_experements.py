from quantum_fourier_transform import QFTGeneration, GATE_SET, GATE_SET_SIMPLE
from linear_genetic_programming import Evolution, plot_many_averages
from grid_search import multiple_runs

from pandas import DataFrame

def run_test(sets, gen_multiplier=8, iterations=20):
    stats = {}
    to_plot = {}
    for set_name in sets:
        QFT_GEN = QFTGeneration(sets[set_name], 3)
        E = Evolution(QFT_GEN, sample_percentage=0.1, gen_mulpilier=gen_multiplier)

        to_plot[set_name], stats[set_name] = multiple_runs(E, iterations=iterations, plot=False)
    
    #for set_name in sets:
    #    print(f'--{set_name}--')
    #    print(DataFrame.from_dict(stats[set_name]))
    #    plot_many_averages(to_plot[set_name], 'Generations', 'Circuit Fitness', legend=False)
    return stats, to_plot


if __name__=="__main__":
    import os
    os.makedirs('out', exist_ok=True)

    ITERATIONS = 10
    sets = {'reduced':GATE_SET_SIMPLE,'overcomplete':GATE_SET}
    test_multipliers = [2,4,8]
    to_plot = []
    all_stats = []
    
    for multiplier in test_multipliers:
        print(f'\n\nmultiplier:{multiplier}')
        s, p = run_test(sets, multiplier, ITERATIONS)
        to_plot.append(p)
        all_stats.append(s)

    for i in range(len(to_plot)):
        p = to_plot[i]
        s = all_stats[i]
        multiplier = test_multipliers[i]
        for set_name in sets:
            print(f'--{set_name}-- multiplier:{multiplier}')
            df = DataFrame.from_dict(s[set_name])
            print(df)
            with open(f'out/iter{ITERATIONS}_{set_name}_mult{multiplier}.csv','w') as file:
                file.write(DataFrame.to_csv(df))
                file.close()
            print(f'plotting...')
            plot_many_averages(p[set_name], 'Generations', 'Circuit Fitness', legend=False)