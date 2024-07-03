from experiments import Experiments
from quantum_fourier_transform import QFTGeneration, GATE_SET
from box_plot import boxplot_from_folder

if __name__=="__main__":
    folder = 'out/eprsc'

    QFT_GEN = QFTGeneration(GATE_SET, 3)
    experiment_instance = Experiments(QFT_GEN,iterations=5,multipliers=[6],generation_count=100,
                                      test_gate_sets={'overcomplete':GATE_SET}, save_filepath=f'{folder}')
    
    experiment_instance.run_test('sorting')
    boxplot_from_folder(f'{folder}', fitness_reference=(2**3-1)/(2**3))