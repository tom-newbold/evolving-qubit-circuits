from quantum_fourier_transform import QFTGeneration, GATE_SET, GATE_SET_SIMPLE
from linear_genetic_programming import Evolution
from linear_genetic_programming_utils import plot_many_averages
from grid_search import multiple_runs
import matplotlib.pyplot as plt

from pandas import DataFrame
import os

class Experiements:
    def __init__(self, iterations=10, multipliers=[2,4,8], save_filepath='out'):
        os.makedirs(save_filepath, exist_ok=True)
        self.ITERATIONS = iterations
        self.test_multipliers = multipliers
        self.base_filepath = save_filepath

    def run_algorithm_test(self, set, gen_multiplier=8, iterations=20):
        """performs multiple runs on each algorithm"""
        stats = {}
        to_plot = {}
        for algorithm in ['random','stochastic','evolution']:
            print(f'<{algorithm}>')
            QFT_GEN = QFTGeneration(set, 3)
            E = Evolution(QFT_GEN, sample_percentage=0.1, gen_mulpilier=gen_multiplier)

            to_plot[algorithm], stats[algorithm] = multiple_runs(E, method=algorithm, iterations=iterations, plot=False)
        return stats, to_plot

    def run_gateset_test(self, sets, gen_multiplier=8, iterations=20):
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

    def run_qubitcount_test(self, set, gen_multiplier=8, iterations=20):
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

    def run_distribution_test(self, set, gen_multiplier=8, iterations=20):
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

    def output(self, p, s, test_param, multiplier, save=True):
        print(f'--{test_param}-- multiplier:{multiplier}')
        df = DataFrame.from_dict(s[test_param])
        print(df)
        with open(self.base_filepath+f'/{test_param}_mult{multiplier}_boxplot.csv','w') as file:
            file.write(DataFrame.to_csv(df))
            file.close()
        print(f'plotting...')
        plot_many_averages(p[test_param], 'Generations', 'Circuit Fitness', legend=False)
        # SAVE FIGURE
        if save:
            plt.savefig(self.base_filepath+f'/{test_param}_mult{multiplier}_graph.png')
        else:
            plt.show()

    def run_test(self, test_name):
        # initialise dictionary of test functions
        test_functions = {'gateset':self.run_gateset_test,'algorithm':self.run_algorithm_test,
                          'qubit':self.run_qubitcount_test,'distribution':self.run_distribution_test}
        if test_name not in test_functions:
            raise ValueError('Invalid test_name')
        t_func = test_functions[test_name]

        to_plot = []
        all_stats = []
        for multiplier in self.test_multipliers:
            print(f'\n\nmultiplier:{multiplier}')
            if test_name=='gateset':
                s, p = t_func({'reduced':GATE_SET_SIMPLE,'overcomplete':GATE_SET}, multiplier, self.ITERATIONS)
            else:
                s, p = t_func(GATE_SET, multiplier, self.ITERATIONS)
            to_plot.append(p)
            all_stats.append(s)

        with open(self.base_filepath+'/params.txt','w') as file:
            # save parameters to allow easy csv reading
            file.write(f'{self.ITERATIONS}\n{",".join([str(m) for m in self.test_multipliers])}\n{",".join(all_stats[0])}')
            file.close()

        # iterate through all tests and plot / save csv
        for i in range(len(to_plot)):
            p = to_plot[i]
            s = all_stats[i]
            multiplier = self.test_multipliers[i]

            for test_param in list(s.keys()):
                self.output(p, s, test_param, multiplier)

if __name__=="__main__":
    experiment_instance = Experiements(save_filepath='out/autosave_test_2',iterations=25, multipliers=[5])

    experiment_instance.run_test('gateset')