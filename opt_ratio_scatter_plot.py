import os
import matplotlib.pyplot as plt
from pandas import read_csv
from linear_genetic_programming_utils import list_avr


def plot_scatters(filepath, fitness_threshold=None):
    os.makedirs(f'{filepath}/plots', exist_ok=True)

    with open(filepath+'/params.txt','r') as file:
        # fetches run parameters in order to consruct csv filenames
        lines = [l.strip('\n') for l in file.readlines()]
        multipliers = [int(m) for m in lines[1].split(',')]
        test_params = lines[2].split(',')

    # removing qiskit from plot
    q_csv = list(filter(lambda key: 'qiskit' in key, test_params))
    for key in q_csv:
        test_params.remove(key)
    csv_to_plot = [f'{tp}_mult{m}.csv' for tp in test_params for m in multipliers]
    dataframes = [read_csv(filepath+'/'+csv_filename) for csv_filename in csv_to_plot]

    if fitness_threshold != None:
        for i, df in enumerate(dataframes):
            # filter non-ideal circuits
            dataframes[i] = df[df['peak_fitness']>=fitness_threshold]

    # compression ratio against unopt
    '''
    for metric in ['depth', 'length']:
        for d_i, dataframe in enumerate(dataframes):
            if len(dataframe[f"r_opt_{metric}"])==0:
                continue
            plt.clf()
            plt.title(f'{csv_to_plot[d_i][:-4].split("_")[0]} - {metric} ratios')
            min_max = [min(dataframe[f"r_unopt_{metric}"]), max(dataframe[f"r_unopt_{metric}"])]
            plt.plot(min_max, min_max, linestyle='dashed')
            plt.scatter(dataframe[f"r_unopt_{metric}"], dataframe[f"r_opt_{metric}"], s=10)
            plt.xlabel('$r_{unopt}$')
            if metric[0]=='d':
                plt.ylabel('$d_{opt}$')
            if metric[0]=='l':
                plt.ylabel('$len_{opt}$')
            plt.tight_layout()
            plt.savefig(f'{filepath}/plots/{csv_to_plot[d_i][:-4]}_{metric}_ratio_scatter.png')
            plt.savefig(f'{filepath}/plots/{csv_to_plot[d_i][:-4]}_{metric}_ratio_scatter.pdf')
    '''

    # all
    for metric in ['depth', 'length']:
        plt.clf()
        min_max = [min([min(df[f"r_unopt_{metric}"]) for df in dataframes]), max([max(df[f"r_unopt_{metric}"]) for df in dataframes])]
        plt.plot(min_max, min_max, linestyle='dashed', label='reference')
        plt.title(f'all {metric} ratios')
        for d_i, dataframe in enumerate(dataframes):
            if len(dataframe[f"r_opt_{metric}"])==0:
                continue
            gradient = list_avr((dataframe[f"r_opt_{metric}"]/dataframe[f"r_unopt_{metric}"]).to_list())
            plt.plot(min_max, [y*gradient for y in min_max], linestyle='dashed', label=csv_to_plot[d_i][:-4]+' avr comp. ratio (inv.)')
            plt.scatter(dataframe[f"r_unopt_{metric}"], dataframe[f"r_opt_{metric}"], s=2, label=csv_to_plot[d_i][:-4])
        plt.xlabel('$r_{unopt}$')
        if metric[0]=='d':
            plt.ylabel('$d_{opt}$')
        if metric[0]=='l':
            plt.ylabel('$len_{opt}$')
        
        plt.tight_layout()
        plt.savefig(f'{filepath}/plots/all_{metric}_ratio_scatter_nolegend.png')
        plt.savefig(f'{filepath}/plots/all_{metric}_ratio_scatter_nolegend.pdf')
        plt.legend(loc='lower right', prop={'size': 'small'})
        plt.savefig(f'{filepath}/plots/all_{metric}_ratio_scatter.png')
        plt.savefig(f'{filepath}/plots/all_{metric}_ratio_scatter.pdf')

    # scaled by runtime
    '''
    for metric in ['depth', 'length']:
        for d_i, dataframe in enumerate(dataframes):
            if 'qiskit' in csv_to_plot[d_i]:
                continue
            plt.clf()
            plt.title(f'{csv_to_plot[d_i][:-4].split("_")[0]} - {metric} ratios (scaled by runtime)')
            #ratio = [a/b for a,b in zip(dataframe[f"r_unopt_{metric}"],dataframe[f"r_opt_{metric}"])]
            #scaled = [a/b for a,b in zip(ratio, dataframe["runtime"])]
            plt.scatter(dataframe[f"r_unopt_{metric}"], dataframe[f"r_opt_{metric}"]*dataframe["runtime"], s=10)
            plt.xlabel('$r_{unopt}$')
            if metric[0]=='d':
                plt.ylabel('$d_{opt}*runtime$')
            if metric[0]=='l':
                plt.ylabel('$len_{opt}*runtime$')
            plt.tight_layout()
            plt.savefig(f'{filepath}/plots/{csv_to_plot[d_i][:-4]}_{metric}_ratio_scatter_scaled.png')
            plt.savefig(f'{filepath}/plots/{csv_to_plot[d_i][:-4]}_{metric}_ratio_scatter_scaled.pdf')
    '''

    for metric in ['depth', 'length']:
        plt.clf()
        plt.title(f'all {metric} ratios (scaled by runtime)')
        for d_i, dataframe in enumerate(dataframes):
            if 'qiskit' in csv_to_plot[d_i]:
                continue
            #ratio = [a/b for a,b in zip(dataframe[f"r_unopt_{metric}"],dataframe[f"r_opt_{metric}"])]
            #scaled = [a/b for a,b in zip(ratio, dataframe["runtime"])]
            plt.scatter(dataframe[f"r_unopt_{metric}"], dataframe[f"r_opt_{metric}"]*dataframe["runtime"], s=2, label=csv_to_plot[d_i][:-4])
        plt.xlabel('$r_{unopt}$')
        if metric[0]=='d':
            plt.ylabel('$d_{opt}*runtime$')
        if metric[0]=='l':
            plt.ylabel('$len_{opt}*runtime$')
        plt.tight_layout()
        
        plt.savefig(f'{filepath}/plots/all_{metric}_ratio_scatter_scaled_nolegend.png')
        plt.savefig(f'{filepath}/plots/all_{metric}_ratio_scatter_scaled_nolegend.pdf')
        plt.legend(loc='lower right', prop={'size': 'small'})
        plt.savefig(f'{filepath}/plots/all_{metric}_ratio_scatter_scaled.png')
        plt.savefig(f'{filepath}/plots/all_{metric}_ratio_scatter_scaled.pdf')


import sys

if __name__=="__main__":
    if len(sys.argv)>1:
        filepath = sys.argv[1]
    else:
        print('no filepath input, using prespecified')
        filepath = 'out/epsrc_qft3'

    try:
        n = int(filepath.strip('/')[-1])
        threshold = (2**n - 1)/2**n
    except:
        threshold = None

    plot_scatters(filepath, threshold)

    #out/epsrc_qft3_new