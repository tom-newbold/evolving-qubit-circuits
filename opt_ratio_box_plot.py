import os
import matplotlib.pyplot as plt
from pandas import read_csv

def labels_handled(test_params):
    labels = []
    for t in test_params:
        try:
            t = t.split('_')
            labels.append(f'{t[0]}\n$\\omega={t[1][5:]}$')
        except:
            labels.append(t[0])
    return labels

def save(filename):
    plt.tight_layout()
    for file_extension in ['png','pdf']:
        plt.savefig(f'{filename}.{file_extension}')

def plot_box_plots(folder, problem, sort_params=False):

    subfolders = [name for name in os.listdir(folder) if os.path.isdir(folder+name)]

    qubit_counts = []
    for subdir in subfolders:
        if subdir[:-1] == f'epsrc_{problem}':
            qubit_counts.append(int(subdir[-1]))


    with open(f'{folder}epsrc_{problem}{qubit_counts[0]}/params.txt','r') as file:
        # fetches run parameters in order to consruct csv filenames
        lines = [l.strip('\n') for l in file.readlines()]
        multipliers = [int(m) for m in lines[1].split(',')]
        test_params = lines[2].split(',')

    fsize = (len(test_params)/2,5)

    if sort_params:
        test_params = sorted(test_params)

    # removing qiskit from plot
    q_csv = list(filter(lambda key: 'qiskit' in key, test_params))
    for key in q_csv:
        test_params.remove(key)
    csv_to_plot = [f'{tp}_mult{m}.csv' for tp in test_params for m in multipliers]



    # per qubit count
    for metric in ['depth', 'length']:
        for q in qubit_counts:
            os.makedirs(f'{folder}epsrc_{problem}{q}/plots', exist_ok=True)
            plt.clf()
            a = plt.subplots(figsize=fsize)[1]
            #a.set_aspect(3)
            dataframes = [read_csv(f'{folder}/epsrc_{problem}{q}/{csv_filename}') for csv_filename in csv_to_plot]
            fitness_threshold = (2**q - 1)/2**q
            for i, df in enumerate(dataframes):
                # filter non-ideal circuits
                dataframes[i] = df[df['peak_fitness']>=fitness_threshold]

            plt.axhline(1, c='r', linewidth=0.5, linestyle='dashed')
            data = [d[f"r_unopt_{metric}"]/d[f"r_opt_{metric}"] for d in dataframes]
            labels = labels_handled(test_params)
            plt.boxplot(data, labels=labels, widths=0.8)
            plt.title(f'compression ratio ({metric})')
            plt.xlabel('method and $\omega$')
            if metric[0]=='d':
                plt.ylabel('$d_{unopt}/d_{opt}$')
            if metric[0]=='l':
                plt.ylabel('$len_{unopt}/len_{opt}$')
            save(f'{folder}epsrc_{problem}{q}/plots/{q}qubits_{metric}_ratio_box')


    # per qubit count (opt)
    for metric in ['depth', 'length']:
        for q in qubit_counts:
            #os.makedirs(f'{folder}epsrc_{problem}{q}/plots', exist_ok=True)
            plt.clf()
            a = plt.subplots(figsize=fsize)[1]
            #a.set_aspect(3)
            dataframes = [read_csv(f'{folder}/epsrc_{problem}{q}/{csv_filename}') for csv_filename in csv_to_plot]
            fitness_threshold = (2**q - 1)/2**q
            for i, df in enumerate(dataframes):
                # filter non-ideal circuits
                dataframes[i] = df[df['peak_fitness']>=fitness_threshold]

            plt.axhline(1, c='r', linewidth=0.5, linestyle='dashed')
            data = [d[f"r_opt_{metric}"] for d in dataframes]
            labels = labels_handled(test_params)
            plt.boxplot(data, labels=labels, widths=0.8)
            plt.title(f'absolute r_opt ({metric})')
            plt.xlabel('method and $\omega$')
            if metric[0]=='d':
                plt.ylabel('$d_{opt}$')
            if metric[0]=='l':
                plt.ylabel('$len_{opt}$')
            save(f'{folder}epsrc_{problem}{q}/plots/{q}qubits_{metric}_r_opt_box')

    # as above, scaled by fitness
    for metric in ['depth', 'length']:
        for q in qubit_counts:
            os.makedirs(f'{folder}epsrc_{problem}{q}/plots', exist_ok=True)
            plt.clf()
            a = plt.subplots(figsize=fsize)[1]
            #a.set_aspect(3)
            dataframes = [read_csv(f'{folder}/epsrc_{problem}{q}/{csv_filename}') for csv_filename in csv_to_plot]
            fitness_threshold = (2**q - 1)/2**q
            for i, df in enumerate(dataframes):
                # filter non-ideal circuits
                dataframes[i] = df[df['peak_fitness']>=fitness_threshold]

            plt.axhline(1, c='r', linewidth=0.5, linestyle='dashed')
            data = [d[f"r_opt_{metric}"]/d["peak_fitness"] for d in dataframes]
            labels = labels_handled(test_params)
            plt.boxplot(data, labels=labels, widths=0.8)
            plt.title(f'r_opt / fitness ({metric})')
            plt.xlabel('method and $\omega$')
            if metric[0]=='d':
                plt.ylabel('$d_{opt}$')
            if metric[0]=='l':
                plt.ylabel('$len_{opt}$')
            save(f'{folder}epsrc_{problem}{q}/plots/{q}qubits_{metric}_r_opt_box_scaled')

    # grouped
    # TODO

    # scaled by runtime
    for metric in ['depth', 'length']:
        for q in qubit_counts:
            os.makedirs(f'{folder}epsrc_{problem}{q}/plots', exist_ok=True)
            plt.clf()
            a = plt.subplots(figsize=fsize)[1]
            #a.set_aspect(3)
            dataframes = [read_csv(f'{folder}/epsrc_{problem}{q}/{csv_filename}') for csv_filename in csv_to_plot]
            fitness_threshold = (2**q - 1)/2**q
            for i, df in enumerate(dataframes):
                # filter non-ideal circuits
                dataframes[i] = df[df['peak_fitness']>=fitness_threshold]

            data = [d[f"r_unopt_{metric}"]/(d[f"r_opt_{metric}"]*d["runtime"]) for d in dataframes]
            labels = labels_handled(test_params)
            plt.boxplot(data, labels=labels, widths=0.8)
            plt.title(f'compression ratio ({metric}) - runtime scaling')
            plt.xlabel('method and $\omega$')
            if metric[0]=='d':
                plt.ylabel('$d_{unopt}/(d_{opt}*runtime)$')
            if metric[0]=='l':
                plt.ylabel('$len_{unopt}/(len_{opt}*runtime)$')
            save(f'{folder}epsrc_{problem}{q}/plots/{q}qubits_{metric}_ratio_box_scaled')


if __name__=="__main__":
    folder = 'out/'
    problem = 'qft'

    plot_box_plots(folder, problem, True)