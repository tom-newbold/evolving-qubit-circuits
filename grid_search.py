from time import time
from linear_genetic_programming import ProblemParametersCombined

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
    total = iterations*len(minimum_fitnesses)*len(random_sample_sizes)*len(sample_percentages)#*2
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