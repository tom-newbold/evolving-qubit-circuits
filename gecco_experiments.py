from qiskit.circuit.library import QFT as QFT_blueprint

from experiments import Experiments
from linear_genetic_programming import AppliedProblemParameters
from toffoli_gate_generation import genericToffoliConstructor
from quantum_fourier_transform import GATE_SET

from box_plot import boxplot_from_folder

def run_alg_test_reduced(e_instance,m=6):
    #stats, plot = e_instance.run_algorithm_test(['random','evolution'],m)
    stats, plot = e_instance.run_algorithm_test(gen_multiplier=m)
    # 
    with open(e_instance.base_filepath+'/params.txt','w') as file:
        file.write(f'{ITERATIONS}\n{m}\n{",".join(list(stats.keys()))}')
        file.close()
    for test_param in list(stats.keys()):
        e_instance.output(plot, stats, test_param, m)

ITERATIONS = 50
GENERATIONS = 100
BOUNDS = [15,45]
MULTIPLIER = 8
if __name__=="__main__":
    base_filepath="out/gecco_local"
    # toffoli
    for qubit_count in [3,4,5]:
        print(f'toffoli{qubit_count}')
        PROB_PARAM = AppliedProblemParameters(GATE_SET, genericToffoliConstructor(qubit_count),
                                              genotype_len_bounds=BOUNDS, genotype_length_falloff='linear')
        experiment_instance = Experiments(PROB_PARAM,iterations=ITERATIONS,save_filepath=f'{base_filepath}/toffoli_{qubit_count}qubit',
                                          generation_count=GENERATIONS,multipliers=[MULTIPLIER])
        #experiment_instance.run_test('algorithm')
        run_alg_test_reduced(experiment_instance, MULTIPLIER)
        boxplot_from_folder(f'{base_filepath}/toffoli_{qubit_count}qubit', fitness_reference=(2**qubit_count-1)/(2**qubit_count))
    # qft
    for qubit_count in [2,3,4,5]:
        print(f'qft{qubit_count}')
        PROB_PARAM = AppliedProblemParameters(GATE_SET, QFT_blueprint(qubit_count),
                                              genotype_len_bounds=BOUNDS, genotype_length_falloff='linear')
        experiment_instance = Experiments(PROB_PARAM,iterations=ITERATIONS,save_filepath=f'{base_filepath}/qft_{qubit_count}qubit',
                                          generation_count=GENERATIONS,multipliers=[MULTIPLIER])
        #experiment_instance.run_test('algorithm')
        run_alg_test_reduced(experiment_instance, MULTIPLIER)
        boxplot_from_folder(f'{base_filepath}/qft_{qubit_count}qubit', fitness_reference=(2**qubit_count-1)/(2**qubit_count))