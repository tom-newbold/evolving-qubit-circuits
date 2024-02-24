import matplotlib.pyplot as plt
from time import time

from linear_genetic_programming import ProblemParametersCombined
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

def run_with_params(evolution, x, iterations, i, total, start_time, min_fitness, random_sample_size, sample_percentage, remove_duplicates, tolerance):
    # update sample_percent
    evolution.sample_percentage = sample_percentage
    evolution.SAMPLE_SIZE = int(evolution.GENERATION_SIZE*sample_percentage)
    # update tolerance
    if tolerance:
        evolution.metadata.tolerance = tolerance
    print(f"LOOP {x+1}/{iterations} TEST {(i-1)%total + 1}/{total} - checking min:{min_fitness} random:{random_sample_size} "+
          f"sample-percent:{sample_percentage} no-dupe:{remove_duplicates} tolerance:{tolerance}")
    run_start = time()
    if i!=1:
        estimated_total_time = (run_start-start_time)*total*iterations/(i-1)
        remaining_time = estimated_total_time*(total*iterations-(i-1))/(total*iterations)
        estimated_total_time = remaining_time_calc(estimated_total_time)
        if estimated_total_time:
            print(f"expected total runtime = {estimated_total_time}")
        remaining_time = remaining_time_calc(remaining_time)
        if remaining_time:
            print(f"expected remaining runtime = {remaining_time}")
    best_genotype = evolution.evolutionary_search(MINIMUM_FITNESS=min_fitness, random_sample_size=random_sample_size, remove_duplicates=remove_duplicates, output=False)[0]
    print(f"actual runtime = {remaining_time_calc(time()-run_start)}")
    print(f"[{x*'0'}{(iterations-x)*'.'}] [{(i%total)*'#'}{(total-(i%total))*'_'}]")
    return {'min_fitness':min_fitness, 'random_sample_size':random_sample_size, 'sample_percentage':sample_percentage, 'remove_duplicates':remove_duplicates, 'best':best_genotype}

def grid_search(evolution, iterations=1, minimum_fitnesses=[0], random_sample_sizes=[0], sample_percentages=[0.05], tolerances=[0.05]):
    """performs a grid search of the 'primary' parameters associated with genotype generation"""
    results = []

    start_time = time()
    i = 1
    total = len(minimum_fitnesses)*len(random_sample_sizes)*len(sample_percentages)#*2
    if type(evolution.metadata)==ProblemParametersCombined:
        total *= len(tolerances)
    else:
        tolerances = [None]

    for x in range(iterations):
        #for remove_duplicates in [True, False]:
        for r_sample in random_sample_sizes:
            for min_fitness in minimum_fitnesses:
                for sample_percent in sample_percentages:
                    for t in tolerances:
                        results.append(run_with_params(evolution, x, iterations, i, total,
                                                    start_time, min_fitness, r_sample,
                                                    sample_percent, True, t))
                        i+=1
                        print(f"peak fitness for run: {results[-1]['best'].fitness}")
    print(f"[{iterations*'.'}] [{total*'#'}]")

    results = sorted(results, key=lambda result: result['best'].fitness, reverse=True)
    
    for j, r in enumerate(results):
        print(f'## {j+1}')
        print(' '.join([f'{key}:{r[key]}' for key in r][:-1]))
        print(f"genotype: {r['best'].genotype_str}")
        print(f"fitness: {r['best'].fitness}")
        print(r['best'].to_circuit())
    return results

def multiple_runs(evolution, iterations=10, method='evolution', min_length=10, max_length=25, MINIMUM_FITNESS=0,
                  crossover_proportion=0.5, insert_delete_proportion=0.1,
                  remove_duplicates=True, use_double_point_crossover=True, output=True, plot=True, legend=True):
    peak_fitness_non_global = (len(evolution.metadata.input_states) - 1) / len(evolution.metadata.input_states)
    start_time = time()
    to_plot = []
    out = []
    stats = {'peak_fitness':[],'runtime':[], 'generations_taken_to_converge':[], 'best_genotype_length_and_depth':[]}
    for i in range(iterations):
        if method=='evolution':
            population, fitness_trace = evolution.evolutionary_search(min_length, max_length, MINIMUM_FITNESS=MINIMUM_FITNESS,
                                                                    remove_duplicates=remove_duplicates,
                                                                    use_double_point_crossover=use_double_point_crossover,
                                                                    crossover_proportion=crossover_proportion,
                                                                    insert_delete_proportion=insert_delete_proportion,
                                                                    output=False)
        elif method=='random':
            population, fitness_trace = evolution.random_search(min_length, max_length, remove_duplicates=remove_duplicates, output=False)
        elif method=='stochastic':
            population, fitness_trace = evolution.stochastic_hill_climb(min_length, max_length, MINIMUM_FITNESS=MINIMUM_FITNESS,
                                                                        remove_duplicates=remove_duplicates, output=False)
        else:
            print('invalid method parameter')
            return
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
        stats['best_genotype_length_and_depth'].append((len(population[0].genotype_str), population[0].to_circuit().depth()))

    if plot:
        plot_many_averages(to_plot, 'Generations', 'Circuit Fitness', legend=legend)
        plt.show()

    if output:
        print(f'{len(out)} runs found \"optimal\" circuits')
        for run, pop in out:
            plot_list(to_plot[run], 'Generations', 'Circuit Fitness', False)
            plt.show()
            print(f'Run {run+1}: Top {evolution.SAMPLE_SIZE} genotypes:')
            for i in range(evolution.SAMPLE_SIZE):
                print(pop[i].genotype_str)
                print(pop[i].get_fitness())
            print('best circuit:')
            print(population[0].to_circuit())

    return to_plot, stats