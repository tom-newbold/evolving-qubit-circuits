import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pandas import read_csv

from opt_ratio_box_plot import labels_handled, save

from linear_genetic_programming import Genotype
from linear_genetic_programming_utils import list_avr
from quantum_fourier_transform import QFTGeneration, QFT_blueprint, GATE_SET
from qiskit import transpile

def labels_handled(test_params):
    labels = []
    for t in test_params:
        try:
            t = t.split('_')
            labels.append(f'{t[0]}\n$\\omega={t[1][5:]}$')
        except:
            labels.append(t[0])
    return labels

def final_plots(folder, problem, sort_params=False):

    subfolders = [name for name in os.listdir(folder) if os.path.isdir(folder+name)]

    qubit_counts = []
    for subdir in subfolders:
        if subdir[:-1] == f'epsrc_{problem}':
            qubit_counts.append(int(subdir[-1]))


    with open(f'{folder}epsrc_{problem}{qubit_counts[0]}/params.txt','r') as file:
        # fetches run parameters in order to consruct csv filenames
        lines = [l.strip('\n') for l in file.readlines()]
        ITERATIONS = int(lines[0])
        multipliers = [int(m) for m in lines[1].split(',')]
        test_params = lines[2].split(',')
        omega_list = [int(x) for x in lines[3].split(',')]

    fsize = (len(test_params),5)

    if sort_params:
        test_params = sorted(test_params)

    # removing qiskit from plot
    q_csv = list(filter(lambda key: 'qiskit' in key, test_params))
    for key in q_csv:
        test_params.remove(key)
    csv_to_plot = [f'{tp}_mult{m}.csv' for tp in test_params for m in multipliers]

    dataframes = {q:[read_csv(f'{folder}/epsrc_{problem}{q}/{csv_filename}') for csv_filename in csv_to_plot] for q in qubit_counts}

    ## peak fitness
    for q in qubit_counts:
        os.makedirs(f'{folder}epsrc_{problem}{q}/plots', exist_ok=True)

        labels = labels_handled(test_params)
        plt.clf()
        data = [d[f"peak_fitness"] for d in dataframes[q]]
        labels = labels_handled(test_params)

        plt.axhline((2**q-1)/(2**q), c='r', linewidth=0.8, linestyle='dashed')
        plt.boxplot(data, labels=labels, widths=0.8)

        plt.title('Peak fitness')
        plt.ylabel('fitness')
        plt.xlabel('method and $\omega$')

        plt.ylim([0,1])
        
        save(f'{folder}epsrc_{problem}{q}/plots/{q}qubits_peak_fitness')

    ## percent of runs generating ideal circuits
    for q in qubit_counts:
        # filter non-ideal circuits
        fitness_threshold = (2**q - 1)/2**q
        for i, df in enumerate(dataframes[q]):
            dataframes[q][i] = df[df['peak_fitness']>=fitness_threshold]

        data = [100*len(d['peak_fitness'])/ITERATIONS for d in dataframes[q]]
        labels = labels_handled(test_params)
        plt.clf()
        plt.bar(labels, data)

        plt.title('Percentage of circuits over ideal fitness threshold')
        plt.ylabel('$\%$')
        plt.xlabel('method and $\omega$')
        plt.ylim([0,100])
        
        save(f'{folder}epsrc_{problem}{q}/plots/{q}qubits_ideal_percent')


    ## complexity
    for q in qubit_counts:
        plt.clf()
        a = plt.subplots(figsize=fsize)[1]

        # plot reference line
        if problem=='qft':
            qft = QFT_blueprint(q)
            qft = transpile(qft, basis_gates=[gate.name for gate in GATE_SET], optimization_level=0)
            g = Genotype(QFTGeneration(N=q))
            g.from_circuit(qft)
            plt.axhline(len(g.genotype_str)/len(qft.data), c='r', linewidth=0.8, linestyle='dashed')

        data = [d["best_genotype_length"]/d["best_genotype_gate_count"] for d in dataframes[q]]
        labels = labels_handled(test_params)
        plt.boxplot(data, labels=labels, widths=0.8)
        plt.title(f'Complexity of ideal circuits')
        plt.xlabel('method and $\omega$')
        plt.ylabel('string length / gate count')
        save(f'{folder}epsrc_{problem}{q}/plots/{q}qubits_complexity_box')


    ## size
    for q in qubit_counts:
        plt.clf()
        a = plt.subplots(figsize=fsize)[1]
        

        # plot reference line
        if problem=='qft':
            qft = QFT_blueprint(q)
            qft = transpile(qft, basis_gates=[gate.name for gate in GATE_SET], optimization_level=0)
            plt.axhline(qft.depth() * len(qft.data), c='r', linewidth=0.8, linestyle='dashed')

        i = next(csv_to_plot.index(csv) for csv in csv_to_plot if 'base' in csv)
        base = dataframes[q][i]
        data = [d["best_genotype_depth"]*d["best_genotype_gate_count"] for d in dataframes[q]]
        data.pop(i)
        labels = labels_handled(test_params)
        labels.remove('base')
        base['scaled_size'] = base["best_genotype_depth"]*base["best_genotype_gate_count"]
        stats = base.describe()
        for percentile in [25,50,75]:
            plt.axhline(stats['scaled_size'][f'{percentile}%'], c='b', linewidth=0.75, linestyle='dashed')

        plt.boxplot(data, labels=labels, widths=0.8)
        plt.title(f'Size of ideal circuits')
        plt.xlabel('method and $\omega$')
        plt.ylabel('circuit depth * gate count')
        save(f'{folder}epsrc_{problem}{q}/plots/{q}qubits_size_box')

    ## size (scaled by runtime)
    for q in qubit_counts:
        plt.clf()
        a = plt.subplots(figsize=fsize)[1]
        
        i = next(csv_to_plot.index(csv) for csv in csv_to_plot if 'base' in csv)
        base = dataframes[q][i]
        data = [d["best_genotype_depth"]*d["best_genotype_gate_count"]*d["runtime"] for d in dataframes[q]]
        data.pop(i)
        labels = labels_handled(test_params)
        labels.remove('base')
        base['scaled_size'] = base["best_genotype_depth"]*base["best_genotype_gate_count"]*base["runtime"]
        stats = base.describe()
        for percentile in [25,50,75]:
            plt.axhline(stats['scaled_size'][f'{percentile}%'], c='b', linewidth=0.75, linestyle='dashed')

        plt.boxplot(data, labels=labels, widths=0.8)

        '''xy = (a.transData + a.transAxes.inverted()).transform((0,stats['scaled_size']['50%']))
        print(xy)
        plt.gcf().text(1, xy[1], '$Q_2$', c='orange', fontsize=12)'''

        plt.title(f'Size of ideal circuits (runtime scaled)')
        plt.xlabel('method and $\omega$')
        plt.ylabel('circuit depth * gate count * runtime')
        save(f'{folder}epsrc_{problem}{q}/plots/{q}qubits_size_box_scaled')


    ## bloat ratio across qubit counts

    test_params.remove('base') # TODO
    dataframes = {t:d for t, d in zip(test_params, [read_csv(f'{folder}/epsrc_{problem}{q}/{csv_filename}') for csv_filename in csv_to_plot])}
    for omega in omega_list:
        plt.clf()
        a = plt.subplots(figsize=fsize)[1]
        labels = list(filter(lambda l: int(l.split('_')[1][5:])==int(omega), test_params))
        
        x_ticks = []
        
        colours = ['red', 'green', 'blue']
        
        for i, q in enumerate(qubit_counts):
            if problem=='qft':
                qft = QFT_blueprint(q)
                qft = transpile(qft, basis_gates=[gate.name for gate in GATE_SET], optimization_level=0)
                g = Genotype(QFTGeneration(N=q))
                g.from_circuit(qft)
                optimal_size = qft.depth() * len(qft.data)

            
            fitness_threshold = (2**q - 1)/2**q            
            data = []
            for key in labels:
                d = dataframes[key]
                d = d[d['peak_fitness']>=fitness_threshold]
                data.append(d["best_genotype_depth"]*d["best_genotype_gate_count"]/optimal_size)
            positions = [1+i*len(labels)+j for j in range(len(labels))]
            
            x_ticks.append(list_avr(positions))
            
            #boxplot = plt.boxplot(data, labels=labels, positions=positions)
            for d, l, p, c in zip(data, labels, positions, colours):
                boxplot = plt.boxplot([d], labels=[l], positions=[p], widths=[0.8])
                for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
                    plt.setp(boxplot[item], color=c)

        plt.axhline(1, c='r', linewidth=0.8, linestyle='dashed')


        plt.xticks(x_ticks, qubit_counts)

        plt.title('$\omega = $'+f'{omega}')
        plt.xlabel('qubit count')
        plt.ylabel('size ratio (produced / ideal)')

        legend_def = [Patch(facecolor=c, edgecolor=c, label=l)
                      for c,l in zip(colours, [l.split('_')[0] for l in labels])]
        plt.legend(handles=legend_def, loc='upper left', prop={'size': 'small'})
        save(f'{folder}omega_{omega}_size_box')
        #
    ### TODO SCALE BY RUNTIME

if __name__=="__main__":
    folder = 'out/'
    problem = 'qft'

    final_plots(folder, problem, True)
