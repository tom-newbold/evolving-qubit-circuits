from experiments import Experiments
from quantum_fourier_transform import QFTGeneration, GATE_SET
from toffoli_gate_generation import ToffoliGeneration
from box_plot import boxplot_from_folder
from opt_ratio_scatter_plot import plot_scatters
from opt_ratio_box_plot import plot_box_plots

import sys

if __name__=="__main__":
    # sys.argv = [filepath, algorithm, qubits, output_directory]
    if len(sys.argv)>=3:
        try:
            N = int(sys.argv[2])
            if N > 0 and N < 10:
                problem = sys.argv[1]
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
            folder = f'out/epsrc_{sys.argv[1]}{N}/'
    else:
        print('no valid parameters provided, running with pre-specified parameters')
        APP = QFTGeneration(GATE_SET, 3)
        folder = 'out/epsrc_qft3/'
        problem = 'qft'

    experiment_instance = Experiments(APP,iterations=25,multipliers=[6],generation_count=100,
                                      test_gate_sets={'overcomplete':GATE_SET}, save_filepath=f'{folder}')
    
    omega_test = [10, 100, 250, 500, 1000]
    #omega_test = [100, 250]
    for omega in omega_test:
        print(f'OMEGA: {omega}')
        experiment_instance.run_test('sorting', omega=omega)
         
        with open(experiment_instance.base_filepath+'/params.txt','w') as file:
            # save parameters to allow easy csv reading
            test_param_list = [f'{test_param}_omega{omega}' for test_param in ['qiskit', 'base', 'length', 'count', 'depth'] for omega in omega_test]
            file.write(f'{experiment_instance.ITERATIONS}\n{",".join([str(m) for m in experiment_instance.test_multipliers])}\n{",".join(test_param_list)}')
            file.close()
    

    boxplot_from_folder(f'{folder}', fitness_reference=(2**APP.qubit_count-1)/(2**APP.qubit_count))
    plot_scatters(folder.strip('/'), fitness_threshold=(2**APP.qubit_count-1)/(2**APP.qubit_count))
    plot_box_plots('/'.join(folder.strip('/').split('/')[:-1]) + '/', problem, fitness_threshold=(2**APP.qubit_count-1)/(2**APP.qubit_count))