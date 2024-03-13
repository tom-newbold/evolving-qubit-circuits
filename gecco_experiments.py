from qiskit.circuit.library import QFT as QFT_blueprint

from experiments import Experiments
from linear_genetic_programming import AppliedProblemParameters
from toffoli_gate_generation import genericToffoliConstructor
from quantum_fourier_transform import GATE_SET

from box_plot import boxplot_from_folder

def run_alg_test_reduced(e_instance,m=6):
    stats, plot = e_instance.run_algorithm_test(['random','evolution'],m)
    # 
    with open(e_instance.base_filepath+'/params.txt','w') as file:
        file.write(f'{ITERATIONS}\n{m}\n{",".join(list(stats.keys()))}')
        file.close()
    for test_param in list(stats.keys()):
        e_instance.output(plot, stats, test_param, m)

ITERATIONS = 10
GENERATIONS = 50
if __name__=="__main__":
    #['random','evolution']
    # toffoli
    for qubit_count in [3]:#[3,4,5,6]:
        print(f'toffoli{qubit_count}')
        PROB_PARAM = AppliedProblemParameters(GATE_SET, genericToffoliConstructor(qubit_count),
                                              genotype_len_bounds=[15,45], genotype_length_falloff='linear')
        experiment_instance = Experiments(PROB_PARAM,iterations=ITERATIONS,save_filepath=f'out/gecco/toffoli_{qubit_count}qubit',
                                          generation_count=GENERATIONS,multipliers=[6])
        #experiment_instance.run_test('algorithm')
        run_alg_test_reduced(experiment_instance, 6)
        boxplot_from_folder(f'out/gecco/toffoli_{qubit_count}qubit', fitness_reference=(2**qubit_count-1)/(2**qubit_count))
    # qft
    for qubit_count in [3]:#[2,3,4,5]:
        print(f'qft{qubit_count}')
        PROB_PARAM = AppliedProblemParameters(GATE_SET, QFT_blueprint(qubit_count),
                                              genotype_len_bounds=[15,45], genotype_length_falloff='linear')
        experiment_instance = Experiments(PROB_PARAM,iterations=ITERATIONS,save_filepath=f'out/gecco/qft_{qubit_count}qubit',
                                          generation_count=GENERATIONS,multipliers=[6])
        #experiment_instance.run_test('algorithm')
        run_alg_test_reduced(experiment_instance, 6)
        boxplot_from_folder(f'out/gecco/qft_{qubit_count}qubit', fitness_reference=(2**qubit_count-1)/(2**qubit_count))