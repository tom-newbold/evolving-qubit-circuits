from experiments import Experiments
from quantum_fourier_transform import QFTGeneration, GATE_SET
from toffoli_gate_generation import ToffoliGeneration
from box_plot import boxplot_from_folder

import sys

if __name__=="__main__":
    #print(sys.argv)
    # sys.argv = [filepath, algorithm, qubits, output_directory]
    if len(sys.argv)>=3:
        try:
            N = int(sys.argv[2])
            if N > 0 and N < 10:
                if sys.argv[1]=='qft':
                    APP = QFTGeneration(GATE_SET, N)
                elif sys.argv[1]=='toffoli':
                    APP = ToffoliGeneration(GATE_SET, N)
                else:
                    raise ValueError('first argument (algorithm) specified incorrectly')
        except:
            raise ValueError('first argument (algorithm) specified incorrectly')
        try:
            folder = sys.argv[3]
        except:
            print('folder not specified, using default directory')
            folder = f'out/epsrc_{sys.argv[1]}_{N}qubits'
    else:
        print('no valid parameters provided, running with pre-specified parameters')
        APP = QFTGeneration(GATE_SET, 3)
        folder = 'out/eprsc_optimisers'

    #QFT_GEN = QFTGeneration(GATE_SET, 3)
    experiment_instance = Experiments(APP,iterations=20,multipliers=[8],generation_count=100,
                                      test_gate_sets={'overcomplete':GATE_SET}, save_filepath=f'{folder}')
    
    experiment_instance.run_test('sorting')
    boxplot_from_folder(f'{folder}', fitness_reference=(2**APP.qubit_count-1)/(2**APP.qubit_count))