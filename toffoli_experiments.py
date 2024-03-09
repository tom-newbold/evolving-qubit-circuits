from qiskit.circuit.library import *
from qft_experiments import Experiments, ALL_TESTS
from toffoli_gate_generation import ToffoliGeneration, genericToffoliConstructor, GATE_SET
from box_plot import boxplot_from_folder

if __name__=="__main__":
    folder = 'temp/' # should end in slash, or be empty

    QFT_GEN = ToffoliGeneration(GATE_SET, 3)
    experiment_instance = Experiments(QFT_GEN,iterations=5,multipliers=[3],
                                      test_gate_sets={'reduced':[HGate(), CXGate(), TGate(), TdgGate()],
                                                      'overcomplete':GATE_SET})
    for test in ALL_TESTS[1:]:
        print(f'__{test.upper()}__')
        experiment_instance.set_save_dir(f'out/{folder}{test}_test')
        experiment_instance.run_test(test, genericToffoliConstructor)
        boxplot_from_folder(f'out/{folder}{test}_test')
    
    test = 'algorithm' # ALL_TESTS[0]
    for qubit_count in [3,4,5]:
        print(f'\n\n__{qubit_count}qubit_ALGORITHM__')
        experiment_instance.set_save_dir(f'out/{folder}{qubit_count}qubit{test}_test')
        experiment_instance.prob_params = ToffoliGeneration(GATE_SET, qubit_count)
        experiment_instance.run_test(test)
        boxplot_from_folder(f'out/{folder}{qubit_count}qubit{test}_test')
    