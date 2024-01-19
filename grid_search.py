from time import time
import matplotlib as plt

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

def run_with_params(evolution, x, iterations, i, total, start_time, min_len, max_len, falloff, min_fitness, random_sample_size, remove_duplicates, _time_estimate_plot):
    print(f"LOOP {x+1}/{iterations} TEST {(i-1)%total + 1}/{total} - checking min:{min_len} max:{max_len} falloff:{falloff}")
    run_start = time()
    if i!=1:
        estimated_total_time = (run_start-start_time)*total*iterations/(i-1)
        _time_estimate_plot.append(estimated_total_time)
        remaining_time = estimated_total_time*(total*iterations-(i-1))/(total*iterations)
        estimated_total_time = remaining_time_calc(estimated_total_time)
        if estimated_total_time:
            print(f"expected total runtime = {estimated_total_time}")
        remaining_time = remaining_time_calc(remaining_time)
        if remaining_time:
            print(f"expected remaining runtime = {remaining_time}")
    best_genotype = evolution.evolutionary_search(min_length=min_len, max_length=max_len, falloff=falloff, MINIMUM_FITNESS=min_fitness,
                                                  random_sample_size=random_sample_size, remove_duplicates=remove_duplicates, output=False)[0]
    print(f"actual runtime = {remaining_time_calc(time()-run_start)}")
    print(f"[{x*'0'}{(iterations-x)*'.'}] [{(i%total)*'#'}{(total-(i%total))*'_'}]")
    return {'min':min_len, 'max':max_len, 'falloff':falloff, 'best':best_genotype}

def grid_search(evolution, lengths=([0],[30]), falloff=[], iterations=1, MINIMUM_FITNESS=0, random_sample_size=0, remove_duplicates=False):
    """performs a grid search of the 'primary' parameters associated with genotype generation"""
    _time_estimate_plot = []
    results = []

    start_time = time()
    i = 1
    total = 0
    for l_max in lengths[1]:
        for l_min in lengths[0]:
            if l_max>l_min:
                total += 1
    total *= len(falloff)+1
    for x in range(iterations):
        for max_len in lengths[1][::-1]: # reversed for more accurate time estimates
            for f in falloff:
                for min_len in lengths[0]:
                    if min_len>=max_len:
                        i+=1
                        continue
                    results.append(run_with_params(evolution, x, iterations, i, total,
                                                   start_time, min_len, max_len, f,
                                                   MINIMUM_FITNESS, random_sample_size,
                                                   remove_duplicates, _time_estimate_plot))
                    i+=1
            results.append(run_with_params(evolution, x, iterations, i, total,
                                           start_time, 0, max_len, None,
                                           MINIMUM_FITNESS, random_sample_size,
                                           remove_duplicates, _time_estimate_plot))
            i+=1

    ### ---- time ----
    time_taken = time()-start_time
    print(f"total time taken = {remaining_time_calc(time_taken)}")
    plt.plot([x/60 for x in _time_estimate_plot])
    plt.ylabel('minuites')
    plt.show()

    results = sorted(results, key=lambda result: result['best'].msf, reverse=True)
    
    for j, r in enumerate(results):
        print(f'## {j+1}')
        print(f"min:{r['min']} max:{r['max']} falloff:{r['falloff']}")
        print(f"genotype: {r['best'].genotype_str}")
        print(f"msf: {r['best'].msf}")
    return results