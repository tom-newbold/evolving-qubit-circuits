import os
import matplotlib.pyplot as plt
from pandas import DataFrame

from quantum_fourier_transform import QFTGeneration, GATE_SET, GATE_SET_SIMPLE
from linear_genetic_programming import Evolution
from linear_genetic_programming_utils import plot_many_averages
from grid_search import multiple_runs
from box_plot import boxplot_from_folder

class Experiements:
    def __init__(self, problem_parameters, iterations=20, multipliers=[2,4,8], save_filepath='out', generation_count=50, default_sample_percent=0.1):
        os.makedirs(save_filepath, exist_ok=True)
        self.prob_params = problem_parameters
        self.ITERATIONS = iterations
        self.test_multipliers = multipliers
        self.base_filepath = save_filepath
        self.default_sample_percent = default_sample_percent
        self.gen_count = generation_count

    def run_algorithm_test(self, gen_multiplier=8):
        """performs multiple runs on each algorithm"""
        stats = {}
        to_plot = {}
        for algorithm in ['random','stochastic','evolution']:
            print(f'<{algorithm}>') # unique identifier used to name output files
            E = Evolution(self.prob_params, number_of_generations=self.gen_count,
                          sample_percentage=self.default_sample_percent, gen_mulpilier=gen_multiplier)

            to_plot[algorithm], stats[algorithm] = multiple_runs(E, method=algorithm, iterations=self.ITERATIONS, plot=False)
        return stats, to_plot

    def run_gateset_test(self, sets, gen_multiplier=8):
        """performs multiple runs on each input gate set"""
        stats = {}
        to_plot = {}
        current_gate_set = self.prob_params.gate_set
        for set_name in sets: # iterate over provided dictionary of sets
            print(f'<{set_name}>') # unique identifier used to name output files
            self.prob_params.set_gate_set(sets[set_name])
            self.prob_params.print_gate_set()
            E = Evolution(self.prob_params, number_of_generations=self.gen_count,
                          sample_percentage=self.default_sample_percent, gen_mulpilier=gen_multiplier)

            to_plot[set_name], stats[set_name] = multiple_runs(E, iterations=self.ITERATIONS, plot=False)
        self.prob_params.set_gate_set(current_gate_set)
        return stats, to_plot

    def run_qubitcount_test(self, circuit_constructor, gen_multiplier=8):
        """performs multiple runs on each circuit size"""
        stats = {}
        to_plot = {}
        for qubit_count in [3, 4, 5]:
            qubit_count_str = f'{qubit_count}qubits'
            # unique identifier used to name output files
            print(f'<{qubit_count_str}>')
            self.prob_params.set_new_circuit(circuit_constructor(qubit_count)) # qubit count varied
            E = Evolution(self.prob_params, number_of_generations=self.gen_count,
                          sample_percentage=self.default_sample_percent, gen_mulpilier=gen_multiplier)

            to_plot[qubit_count_str], stats[qubit_count_str] = multiple_runs(E, iterations=self.ITERATIONS, plot=False)
        self.prob_params.set_new_circuit(circuit_constructor(3))
        return stats, to_plot

    def run_distribution_test(self, gen_multiplier=8):
        """performs multiple runs on each random distribution"""
        stats = {}
        to_plot = {}
        for crossover in [3, 4, 5, 6, 7]:
            for x in ['single','double']:
                dist_str = f'crossover{crossover}{x}'
                # unique identifier used to name output files
                print(f'<{dist_str}>')
                E = Evolution(self.prob_params, number_of_generations=self.gen_count,
                              sample_percentage=self.default_sample_percent, gen_mulpilier=gen_multiplier)

                to_plot[dist_str], stats[dist_str] = multiple_runs(E, crossover_proportion=crossover/10,
                                                                use_double_point_crossover= x=='double',
                                                                iterations=self.ITERATIONS, plot=False)
        return stats, to_plot
    
    def run_multiobjective_test(self, gen_multiplier=8):
        """performs multiple runs on each circuit preference"""
        stats = {}
        to_plot = {}
        for circuit_preference in [None, True, False]: # no sorting, shortest first, longest first
            multobj_str = "none" if circuit_preference==None else ("short" if circuit_preference else "long")
            multobj_str = 'preferlength'+multobj_str
            # unique identifier used to name output files
            print(f'<{multobj_str}>')
            E = Evolution(self.prob_params, number_of_generations=self.gen_count,
                          sample_percentage=self.default_sample_percent, gen_mulpilier=gen_multiplier)

            to_plot[multobj_str], stats[multobj_str] = multiple_runs(E, iterations=self.ITERATIONS, short_circuit_preference=circuit_preference, plot=False)
        return stats, to_plot
    
    def run_elitism_test(self, gen_multiplier=8):
        """performs multiple runs using different sample (elitism) percentages"""
        stats = {}
        to_plot = {}
        for elitism_percent in [0.05,0.1,0.15,0.2,0.25]:
            elite_str = f'elitepercent{int(elitism_percent*100)}'
            # unique identifier used to name output files
            print(f'<{elite_str}>')
            E = Evolution(self.prob_params, number_of_generations=self.gen_count,
                          sample_percentage=elitism_percent, gen_mulpilier=gen_multiplier)

            to_plot[elite_str], stats[elite_str] = multiple_runs(E, iterations=self.ITERATIONS, plot=False)
        return stats, to_plot

    def output(self, p, s, test_param, multiplier, save=True):
        """writes stats to dataframe, plots graph of averages, and saves when required"""
        print(f'--{test_param}-- multiplier:{multiplier}')
        df = DataFrame.from_dict(s[test_param])
        #print(df)
        with open(self.base_filepath+f'/{test_param}_mult{multiplier}.csv','w') as file:
            # writes dataframe to unique file, statistical analysis and further plots can be carried out externally
            file.write(DataFrame.to_csv(df))
            file.close()
        print(f'plotting...')
        plot_many_averages(p[test_param], 'Generations', 'Circuit Fitness', legend=False)
        if save: # saves figure if specified
            plt.savefig(self.base_filepath+f'/{test_param}_mult{multiplier}_graph.png')
        else:
            plt.show()

    def run_test(self, test_name, circuit_constructor=None):
        # initialise dictionary of test functions
        test_functions = {'gateset':self.run_gateset_test,'algorithm':self.run_algorithm_test,
                          'qubit':self.run_qubitcount_test,'distribution':self.run_distribution_test,
                          'multiobjective':self.run_multiobjective_test,'elitism':self.run_elitism_test}
        if test_name not in test_functions:
            raise ValueError('Invalid test_name')
        t_func = test_functions[test_name]

        to_plot = []
        all_stats = []
        for multiplier in self.test_multipliers:
            print(f'\n\nmultiplier:{multiplier}')
            if test_name=='gateset':
                s, p = t_func({'reduced':GATE_SET_SIMPLE,'overcomplete':GATE_SET}, multiplier)
            elif test_name=='qubit':
                if circuit_constructor==None:
                    raise ValueError('No circuit constructor provided')
                else:
                    s, p = t_func(circuit_constructor, multiplier)
            else:
                s, p = t_func(multiplier)
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

ALL_TESTS = ['algorithm','gateset','qubit','distribution','multiobjective','elitism']

if __name__=="__main__":
    #experiment_instance = Experiements(save_filepath=f'out/autosave_test_2',iterations=10, multipliers=[2])
    #experiment_instance.run_test('elitism')
    from qiskit.circuit.library import QFT as QFT_blueprint

    QFT_GEN = QFTGeneration(GATE_SET, 3)
    for test in ALL_TESTS:
        print(f'__{test.upper()}__')
        experiment_instance = Experiements(QFT_GEN, save_filepath=f'out/final/{test}_test',iterations=25, multipliers=[2,5])
        experiment_instance.run_test(test, QFT_blueprint)
        boxplot_from_folder(f'out/final/{test}_test')