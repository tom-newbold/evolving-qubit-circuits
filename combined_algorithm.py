from bulk_runs import multiple_runs
from quantum_fourier_transform import QFTGeneration
from linear_genetic_programming import Evolution
from experiments import Experiments

from box_plot import boxplot_from_folder
from opt_ratio_scatter_plot import plot_scatters

QFT_GEN = QFTGeneration()
evolution = Evolution(QFT_GEN, number_of_generations=100, gen_mulpilier=5)
ITERATIONS = 5

experiment_instance = Experiments(QFT_GEN,iterations=ITERATIONS,multipliers=[evolution.GENERATION_MULTIPLIER],
                                  generation_count=evolution.GENERATION_COUNT, save_filepath='out/combined_algorithm/')

to_plot = {}
stats= {}
to_plot['combined'], stats['combined'] = multiple_runs(evolution, iterations=ITERATIONS, method='combined', plot=False, save_dir='out/combined_algorithm/')

experiment_instance.output(to_plot, stats, 'combined', evolution.GENERATION_MULTIPLIER)
with open(experiment_instance.base_filepath+'/params.txt','w') as file:
    # save parameters to allow easy csv reading
    file.write(f'{experiment_instance.ITERATIONS}\n{",".join([str(m) for m in experiment_instance.test_multipliers])}\ncombined')
    file.close()

boxplot_from_folder('out/combined_algorithm/', fitness_reference=(2**QFT_GEN.qubit_count-1)/(2**QFT_GEN.qubit_count))
#plot_scatters('out/combined_algorithm', fitness_threshold=(2**QFT_GEN.qubit_count-1)/(2**QFT_GEN.qubit_count))