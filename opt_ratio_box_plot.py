import os
import matplotlib.pyplot as plt
from pandas import read_csv

def plot_box_plots(folder, problem):

    subfolders = [name for name in os.listdir(folder) if os.path.isdir(folder+name)]

    qubit_counts = []
    for subdir in subfolders:
        if subdir[:-1] == f'epsrc_{problem}':
            qubit_counts.append(subdir[-1])


    with open(f'{folder}epsrc_{problem}{qubit_counts[0]}/params.txt','r') as file:
        # fetches run parameters in order to consruct csv filenames
        lines = [l.strip('\n') for l in file.readlines()]
        ITERATIONS = int(lines[0])
        multipliers = [int(m) for m in lines[1].split(',')]
        test_params = lines[2].split(',')

    test_params.remove('qiskit')
    csv_to_plot = [f'{tp}_mult{m}.csv' for tp in test_params for m in multipliers]


    # per qubit count
    for metric in ['depth', 'length']:
        for q in qubit_counts:
            os.makedirs(f'{folder}epsrc_{problem}{q}/plots', exist_ok=True)
            plt.clf()
            dataframes = [read_csv(f'{folder}/epsrc_{problem}{q}/{csv_filename}') for csv_filename in csv_to_plot]
            data = [d[f"r_unopt_{metric}"]/d[f"r_opt_{metric}"] for d in dataframes]
            plt.boxplot(data, labels=test_params)
            plt.title(f'compression ratio ({metric})')
            plt.xlabel('$r_{unopt}$')
            if metric[0]=='d':
                plt.ylabel('$d_{unopt}/d_{opt}$')
            if metric[0]=='l':
                plt.ylabel('$len_{unopt}/len_{opt}$')
            plt.tight_layout()
            plt.savefig(f'{folder}epsrc_{problem}{q}/plots/{q}qubits_{metric}_ratio_box.png')
            plt.savefig(f'{folder}epsrc_{problem}{q}/plots/{q}qubits_{metric}_ratio_box.pdf')


    # grouped
    # TODO

    # scaled by runtime
    for metric in ['depth', 'length']:
        for q in qubit_counts:
            os.makedirs(f'{folder}epsrc_{problem}{q}/plots', exist_ok=True)
            plt.clf()
            dataframes = [read_csv(f'{folder}/epsrc_{problem}{q}/{csv_filename}') for csv_filename in csv_to_plot]
            data = [d[f"r_unopt_{metric}"]/(d[f"r_opt_{metric}"]*d["runtime"]) for d in dataframes]
            plt.boxplot(data, labels=test_params)
            plt.title(f'compression ratio ({metric}) - runtime scaling')
            plt.xlabel('$r_{unopt}$')
            if metric[0]=='d':
                plt.ylabel('$d_{unopt}/(d_{opt}*runtime)$')
            if metric[0]=='l':
                plt.ylabel('$len_{unopt}/(len_{opt}*runtime)$')
            plt.tight_layout()
            plt.savefig(f'{folder}epsrc_{problem}{q}/plots/{q}qubits_{metric}_ratio_box_scaled.png')
            plt.savefig(f'{folder}epsrc_{problem}{q}/plots/{q}qubits_{metric}_ratio_box_scaled.pdf')


if __name__=="__main__":
    folder = 'out/'
    problem = 'qft'

    plot_box_plots(folder, problem)