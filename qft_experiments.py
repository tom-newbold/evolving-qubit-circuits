from qiskit.circuit.library import QFT as QFT_blueprint

from experiments import Experiments, ALL_TESTS
from box_plot import boxplot_from_folder

from quantum_fourier_transform import QFTGeneration, GATE_SET, GATE_SET_SIMPLE

if __name__=="__main__":
    folder = 'qft/' # should end in slash, or be empty

    QFT_GEN = QFTGeneration(GATE_SET, 3)
    experiment_instance = Experiments(QFT_GEN,iterations=25,multipliers=[3,6],generation_count=100,
                                      test_gate_sets={'reduced':GATE_SET_SIMPLE,'overcomplete':GATE_SET})
    
    for test in ALL_TESTS[1:]:
        print(f'__{test.upper()}__')
        experiment_instance.set_save_dir(f'out/{folder}{test}_test')
        experiment_instance.run_test(test, QFT_blueprint)
        if test=='qubit':
            boxplot_from_folder(f'out/{folder}{test}_test')
        else:
            n = experiment_instance.prob_params.qubit_count
            boxplot_from_folder(f'out/{folder}{test}_test', (2**n-1)/(2**n))
    
    # separate algorithm test for each qubit count
    test = 'algorithm'
    for qubit_count in [3,4,5]:
        print(f'\n\n__{qubit_count}qubit_ALGORITHM__')
        experiment_instance.set_save_dir(f'out/{folder}{qubit_count}qubit{test}_test')
        experiment_instance.prob_params = QFTGeneration(GATE_SET, qubit_count)
        experiment_instance.run_test(test)
        boxplot_from_folder(f'out/{folder}{qubit_count}qubit{test}_test', (2**qubit_count-1)/(2**qubit_count))
    