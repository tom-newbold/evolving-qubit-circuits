import matplotlib.pyplot as plt
from time import time

from linear_genetic_programming_utils import plot_many_averages, plot_list

def remaining_time_calc(remaining_time):
    if remaining_time > 0.001:
        if remaining_time < 60:
            remaining_time = f'{remaining_time:.01f} seconds'
        elif remaining_time < 3600:
            remaining_time /= 60
            remaining_time = f'{int(remaining_time)} minuites {int((remaining_time-int(remaining_time))*60)} seconds'
        else:
            remaining_time /= 3600
            remaining_time = f'{int(remaining_time)} hours {int((remaining_time-int(remaining_time))*60)} minuites'
        return remaining_time

def multiple_runs(evolution, iterations=10, method='evolution', min_length=None, max_length=None, MINIMUM_FITNESS=0,
                  crossover_proportion=0.5, insert_delete_proportion=0.1, remove_duplicates=True,
                  use_double_point_crossover=True, short_circuit_preference=None, output=True, plot=True, legend=True, save_dir='out/'):
    if min_length==None or max_length==None:
        if evolution.metadata.genotype_length_bounds!=None and len(evolution.metadata.genotype_length_bounds)==2:
            min_length, max_length = evolution.metadata.genotype_length_bounds
        else:
            raise ValueError('Genotype generation parameters not fully specified')
    peak_fitness_non_global = (len(evolution.metadata.input_states) - 1) / len(evolution.metadata.input_states)
    start_time = time()
    to_plot = []
    out = []
    stats = {'peak_fitness':[],'runtime':[], 'generations_taken_to_converge':[], 'best_genotype_length':[], 'best_genotype_depth':[]}
    for i in range(iterations):
        # run with desired algoirithm
        if method=='evolution':
            population, fitness_trace = evolution.evolutionary_search(min_length, max_length, MINIMUM_FITNESS=MINIMUM_FITNESS,
                                                                    remove_duplicates=remove_duplicates,
                                                                    use_double_point_crossover=use_double_point_crossover,
                                                                    crossover_proportion=crossover_proportion,
                                                                    insert_delete_proportion=insert_delete_proportion,
                                                                    output=False, prefer_short_circuits=short_circuit_preference)
        elif method=='random':
            population, fitness_trace = evolution.random_search(min_length, max_length, remove_duplicates=remove_duplicates, output=False)
        elif method=='stochastic':
            population, fitness_trace = evolution.stochastic_hill_climb(min_length, max_length, MINIMUM_FITNESS=MINIMUM_FITNESS,
                                                                        remove_duplicates=remove_duplicates, output=False)
        else:
            raise ValueError('Invalid method parameter')
        to_plot.append(fitness_trace)
        if population[0].get_fitness() > peak_fitness_non_global:
            out.append((i, population))
        delta_time = time()-start_time
        print(f'{(i+1)*"█"}{(iterations-i-1)*"░"} runtime = {remaining_time_calc(delta_time)}')
        start_time = time()

        # stats to return
        stats['peak_fitness'].append(fitness_trace[0][-1])
        stats['runtime'].append(delta_time)
        for i in range(len(fitness_trace[0])):
            if fitness_trace[0][i]==fitness_trace[0][-1]:
                stats['generations_taken_to_converge'].append(i)
                break
        stats['best_genotype_length'].append(len(population[0].genotype_str))
        stats['best_genotype_depth'].append(population[0].to_circuit().depth())

    if plot:
        plot_many_averages(to_plot, 'Generations', 'Circuit Fitness', legend=legend)
        plt.show()

    if output:
        print(f'{len(out)} runs found \"optimal\" circuits')
        for run, pop in out:
            if plot:
                plot_list(to_plot[run], 'Generations', 'Circuit Fitness', False)
                plt.show()
            # save optimal circuits to file
            with open(f'{save_dir}optimal_circuits.txt','a+',encoding='utf-8') as file:
                file.write(f'Run {run+1}: Top {evolution.SAMPLE_SIZE} genotypes:\n')
                for i in range(evolution.SAMPLE_SIZE):
                    file.write(f'{pop[i].genotype_str}\n{pop[i].get_fitness()}\n')
                file.write(f'best circuit:\n{str(pop[0].to_circuit().draw("text"))}\n\n')
                file.close()

    return to_plot, stats