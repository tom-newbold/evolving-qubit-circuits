import os
import matplotlib.pyplot as plt
from pandas import DataFrame

from linear_genetic_programming import Evolution
from linear_genetic_programming_utils import plot_many_averages, list_avr
from bulk_runs import multiple_runs

class Experiments:
    def __init__(self, problem_parameters, iterations=20, multipliers=[2,4,8], save_filepath='out',
                 generation_count=50, default_sample_percent=0.1,
                 test_gate_sets={}):
        self.set_save_dir(save_filepath)
        self.prob_params = problem_parameters
        self.ITERATIONS = iterations
        self.test_multipliers = multipliers
        self.default_sample_percent = default_sample_percent
        self.gen_count = generation_count
        self.test_gate_sets = test_gate_sets

    def set_save_dir(self, save_filepath):
        os.makedirs(save_filepath, exist_ok=True)
        self.base_filepath = save_filepath

    def run_algorithm_test(self, gen_multiplier=8, algorithms=['random','stochastic','evolution']):
        """performs multiple runs on each algorithm"""
        stats = {}
        to_plot = {}
        for algorithm in algorithms:
            print(f'<{algorithm}>') # unique identifier used to name output files
            E = Evolution(self.prob_params, number_of_generations=self.gen_count,
                          sample_percentage=self.default_sample_percent, gen_mulpilier=gen_multiplier)

            to_plot[algorithm], stats[algorithm] = multiple_runs(E, method=algorithm, iterations=self.ITERATIONS,
                                                                 plot=False, save_dir=self.base_filepath+'/')
        return stats, to_plot

    def run_gateset_test(self, sets, gen_multiplier=8):
        """performs multiple runs on each input gate set"""
        if len(sets)==0:
            return ValueError('No gate sets provided')
        stats = {}
        to_plot = {}
        current_gate_set = self.prob_params.gate_set
        for set_name in sets: # iterate over provided dictionary of sets
            print(f'<{set_name}>') # unique identifier used to name output files
            self.prob_params.set_gate_set(sets[set_name])
            self.prob_params.print_gate_set()
            E = Evolution(self.prob_params, number_of_generations=self.gen_count,
                          sample_percentage=self.default_sample_percent, gen_mulpilier=gen_multiplier)

            to_plot[set_name], stats[set_name] = multiple_runs(E, iterations=self.ITERATIONS, plot=False, save_dir=self.base_filepath+'/')
        self.prob_params.set_gate_set(current_gate_set)
        return stats, to_plot

    def run_qubitcount_test(self, circuit_constructor=None, gen_multiplier=8):
        """performs multiple runs on each circuit size"""
        if circuit_constructor==None:
            raise ValueError('No circuit constructor provided')
        stats = {}
        to_plot = {}
        for qubit_count in [3, 4, 5]:
            qubit_count_str = f'{qubit_count}qubits'
            # unique identifier used to name output files
            print(f'<{qubit_count_str}>')
            self.prob_params.set_new_circuit(circuit_constructor(qubit_count)) # qubit count varied
            E = Evolution(self.prob_params, number_of_generations=self.gen_count, individuals_per_generation=100*2**(qubit_count-3),
                          sample_percentage=self.default_sample_percent, gen_mulpilier=gen_multiplier)
            # gen_size is modified to account for larger search space

            to_plot[qubit_count_str], stats[qubit_count_str] = multiple_runs(E, iterations=self.ITERATIONS, plot=False, save_dir=self.base_filepath+'/')
        self.prob_params.set_new_circuit(circuit_constructor(3))
        return stats, to_plot

    def run_distribution_test(self, gen_multiplier=8):
        """performs multiple runs on each random distribution"""
        stats = {}
        to_plot = {}
        for crossover in [3, 5, 7]:
            #for x in ['single','double']:
            #dist_str = f'{x}crossover{crossover}'
            dist_str = f'crossover{crossover}'
            # unique identifier used to name output files
            print(f'<{dist_str}>')
            E = Evolution(self.prob_params, number_of_generations=self.gen_count,
                            sample_percentage=self.default_sample_percent, gen_mulpilier=gen_multiplier)

            to_plot[dist_str], stats[dist_str] = multiple_runs(E, crossover_proportion=crossover/10,
                                                            #use_double_point_crossover= x=='double',
                                                            iterations=self.ITERATIONS, plot=False,
                                                            save_dir=self.base_filepath+'/')
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

            to_plot[multobj_str], stats[multobj_str] = multiple_runs(E, iterations=self.ITERATIONS, short_circuit_preference=circuit_preference,
                                                                     plot=False, save_dir=self.base_filepath+'/')
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

            to_plot[elite_str], stats[elite_str] = multiple_runs(E, iterations=self.ITERATIONS, plot=False, save_dir=self.base_filepath+'/')
        return stats, to_plot
    
    def run_sorting_test(self, gen_multiplier=8, omega=100):
        """performs multiple runs using different sorting functions"""
        stats = {}
        to_plot = {}
        avr_gate_string_length = list_avr([len(s) for s in self.prob_params.all_gate_combinations])
        print(f'avr symbols/gate : {avr_gate_string_length}')
        functions = {
            'base':None,
            'length': lambda genotype: omega*genotype.get_fitness() - len(genotype.genotype_str)/avr_gate_string_length,
            'count': lambda genotype: omega*genotype.get_fitness() - len(genotype.to_circuit().data),
            'depth': lambda genotype: omega*genotype.get_fitness() - genotype.to_circuit().depth()
        }
        E = Evolution(self.prob_params)
        from qiskit import transpile
        from circuit_unoptimiser import unoptimiser
        transpiled = transpile(self.prob_params.target_circuit, basis_gates=[self.prob_params.gate_set[gate_key].name for gate_key in self.prob_params.gate_set], optimization_level=0)

        circuit_population = []
        for i in range(self.ITERATIONS):
            print(f'unoptimising: {"#"*(i+1)}{"-"*(self.ITERATIONS-i-1)}', end='\r')
            circuit_population.append(unoptimiser(transpiled, self.prob_params, self.prob_params.qubit_count))
        print('')

        qiskit_optimised = [transpile(c[0].copy(), basis_gates=[self.prob_params.gate_set[gate_key].name for gate_key in self.prob_params.gate_set],
                                      optimization_level=3) for c in circuit_population]
        stats['qiskit'] = {}
        stats['qiskit']["peak_fitness"] = [self.prob_params.circuit_fitness(c) for c in qiskit_optimised]
        stats['qiskit']["r_opt_depth"] = [c.depth()/transpiled.depth() for c in qiskit_optimised]
        stats['qiskit']["r_unopt_depth"] = [c[0].depth()/transpiled.depth() for c in circuit_population]
        stats['qiskit']["r_opt_length"] = [len(c.data)/len(transpiled.data) for c in qiskit_optimised]
        stats['qiskit']["r_unopt_length"] = [len(c[0].data)/len(transpiled.data) for c in circuit_population]
        # save runtime??

        for func_name in functions:
            # unique identifier used to name output files
            omega_func = f'{func_name}_omega{omega}'
            print(f'<{omega_func}>')
            E = Evolution(self.prob_params, number_of_generations=self.gen_count, gen_mulpilier=gen_multiplier, sorting_function_override=functions[func_name])

            to_plot[omega_func], stats[omega_func] = multiple_runs(E, iterations=self.ITERATIONS, method='optimisation', plot=False, save_dir=self.base_filepath+'/',
                                                                 circuit_population=circuit_population, insert_delete_proportion=0.5)
            #print(f'absolute r_opt: {list_avr(stats[func_name]["best_genotype_depth"])/transpiled.depth()}')

            print([d==c[0].depth() for d, c in zip(stats[omega_func]["best_genotype_depth"],circuit_population)])
            stats[omega_func]["r_opt_depth"] = [d/transpiled.depth() for d in stats[omega_func]["best_genotype_depth"]]
            stats[omega_func]["r_unopt_depth"] = [c[0].depth()/transpiled.depth() for c in circuit_population]
            stats[omega_func]["r_opt_length"] = [l/len(transpiled.data) for l in stats[omega_func]["best_genotype_length"]]
            stats[omega_func]["r_unopt_length"] = [len(c[0].data)/len(transpiled.data) for c in circuit_population]
        return stats, to_plot

    def output(self, p, s, test_param, multiplier, save=True):
        """writes stats to dataframe, plots graph of averages, and saves when required"""
        df = DataFrame.from_dict(s[test_param])
        with open(self.base_filepath+f'{test_param}_mult{multiplier}.csv','w') as file:
            # writes dataframe to unique file, statistical analysis and further plots can be carried out externally
            file.write(DataFrame.to_csv(df))
            file.close()
        if 'qubits' in test_param:
            n = int(test_param[0])
        else:
            n = self.prob_params.qubit_count
        try:
            plot_many_averages(p[test_param], 'Generations', 'Circuit Fitness', legend=False, reference_line=(2**n-1)/(2**n))
            #plot_many_averages(p[test_param], 'Generations', 'Circuit Fitness', legend=False)
            if save: # saves figure if specified
                plt.savefig(self.base_filepath+f'/{test_param}_mult{multiplier}_graph.png')
            else:
                plt.show()
        except:
            if test_param!='qiskit':
                print(f'error ploting for test parameter {test_param}')
            pass

    def run_test(self, test_name, circuit_constructor=None, omega=None):
        # initialise dictionary of test functions
        test_functions = {'gateset':self.run_gateset_test,'algorithm':self.run_algorithm_test,
                          'qubit':self.run_qubitcount_test,'distribution':self.run_distribution_test,
                          'multiobjective':self.run_multiobjective_test,'elitism':self.run_elitism_test,
                          'sorting':self.run_sorting_test}
        if test_name not in test_functions:
            raise ValueError('Invalid test_name')
        t_func = test_functions[test_name]

        to_plot = []
        all_stats = []
        for multiplier in self.test_multipliers:
            print(f'\n\nmultiplier:{multiplier}')
            if test_name=='gateset':
                s, p = t_func(self.test_gate_sets, multiplier)
            elif test_name=='qubit':
                s, p = t_func(circuit_constructor, multiplier)
            elif test_name=='sorting' and omega!=None:
                s, p = t_func(gen_multiplier=multiplier, omega=omega)
            else:
                s, p = t_func(multiplier)
            to_plot.append(p)
            all_stats.append(s)
    
        with open(self.base_filepath+'/params.txt','w') as file:
            # save parameters to allow easy csv reading
            test_param_list = list(all_stats[0].keys())
            test_param_list.remove('qiskit')
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